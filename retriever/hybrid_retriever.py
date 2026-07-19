"""Hybrid retriever: BM25 + Chroma Dense + RRF + FlashRank second-stage rerank.

Pipeline (per query)
--------------------
1. **BM25** over Vietnamese-stripped tokens (rank_bm25) → top ``k_per_retriever``.
2. **Dense** via Chroma collection query (cosine space) → top ``k_per_retriever``.
3. **RRF fusion** of the two ranked lists (k=60 default) → top ``rerank_pool``
   candidates (default 20).  This is the "Hybrid Search" stage.
4. **FlashRank cross-encoder rerank** on the ``rerank_pool`` candidates →
   final ``top_k`` results.  FlashRank uses ``ranker_ms_marco_viT_5_1`` under
   the hood — it is tiny (~120 MB) and runs on CPU in <50 ms for 20 docs.

If FlashRank is unavailable (e.g. import failed at runtime), the retriever
falls back gracefully to RRF-only output.  This keeps the system runnable in
constrained environments while still respecting the "RRF + FlashRank" design
whenever the dependency is present.
"""

from __future__ import annotations

import json
import logging
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_DB_DIR = Path(__file__).resolve().parents[1] / "data" / "vector_db"
DEFAULT_RRF_K = 60
DEFAULT_COLLECTION_NAME = "uet_handbook"
DEFAULT_FLASHRANK_MODEL = "ms-marco-MiniLM-L-12-v2"


# --------------------------------------------------------------------------- #
# Vietnamese text utilities
# --------------------------------------------------------------------------- #
_DIAKRITIC_MAP = str.maketrans(
    "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
    "ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ",
    "a" * 67 + "A" * 67,
)


def remove_diacritics(text: str) -> str:
    return text.translate(_DIAKRITIC_MAP)


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str, *, lower: bool = True, strip_diacritics: bool = True) -> List[str]:
    """Tokenise for BM25."""
    if lower:
        text = text.lower()
    if strip_diacritics:
        text = remove_diacritics(text)
    return _TOKEN_RE.findall(text)


# --------------------------------------------------------------------------- #
# BM25 index (unchanged)
# --------------------------------------------------------------------------- #
class _BM25Index:
    """Lightweight wrapper around rank_bm25.BM25Okapi."""

    def __init__(self, corpus_tokens: List[List[str]]):
        from rank_bm25 import BM25Okapi

        self.bm25 = BM25Okapi(corpus_tokens)

    def get_scores(self, query_tokens: List[str]) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(query_tokens), dtype=np.float32)


# --------------------------------------------------------------------------- #
# FlashRank reranker (lazy, optional)
# --------------------------------------------------------------------------- #
class _FlashRanker:
    """Wrap the ``flashrank.Ranker`` cross-encoder.

    The model is downloaded on first use (~120 MB) and cached on disk.  We
    instantiate it lazily so that environments without FlashRank installed
    still work — they just skip the rerank step.
    """

    def __init__(self, model_name: str = DEFAULT_FLASHRANK_MODEL):
        self.model_name = model_name
        self._ranker = None
        self._available: Optional[bool] = None

    def _ensure_loaded(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            from flashrank import Ranker  # type: ignore

            logger.info("Loading FlashRank model '%s' (CPU, ~120 MB) ...", self.model_name)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._ranker = Ranker(model_name=self.model_name, cache_dir="/tmp/.flashrank_cache")
            self._available = True
        except ImportError:
            logger.warning(
                "flashrank not installed — falling back to RRF-only (no second-stage rerank). "
                "Install with: pip install flashrank"
            )
            self._available = False
        except Exception as exc:  # pragma: no cover
            logger.warning("FlashRank load failed (%s); RRF-only fallback.", exc)
            self._available = False
        return self._available

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        *,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Re-order ``candidates`` and return the top ``top_k``.

        ``candidates`` must each contain a ``text`` field.  The returned list
        has the same dict shape, with an added ``rerank_score`` field.
        """
        if not candidates or top_k <= 0:
            return []
        if not self._ensure_loaded():
            # No rerank available — return as-is, truncated.
            return candidates[:top_k]

        from flashrank import RerankRequest  # type: ignore

        passages = [
            {"id": str(i), "text": c.get("text", "")[:1500]}
            for i, c in enumerate(candidates)
        ]
        try:
            request = RerankRequest(query=query, passages=passages)
            results = self._ranker.rerank(request)
        except Exception as exc:  # pragma: no cover
            logger.warning("FlashRank.rerank failed (%s); returning RRF order.", exc)
            return candidates[:top_k]

        # results is a list of dicts with at least "id" and "score".
        out: List[Dict[str, Any]] = []
        for r in results[:top_k]:
            idx = int(r["id"])
            chunk = dict(candidates[idx])
            chunk["rerank_score"] = float(r["score"])
            out.append(chunk)
        return out


# --------------------------------------------------------------------------- #
# Hybrid retriever
# --------------------------------------------------------------------------- #
class HybridRetriever:
    """BM25 + Chroma Dense + RRF + FlashRank.

    Parameters
    ----------
    db_dir
        Directory containing the ``chroma/`` subdirectory and ``chunks.jsonl``.
    embedder
        A pre-loaded ``_Embedder`` instance. If ``None`` we lazily create one.
    rrf_k
        RRF constant (default 60).
    use_flashrank
        ``True`` (default) to enable FlashRank rerank.  Set to ``False`` to
        force RRF-only.
    flashrank_model
        FlashRank model name (default ``ranker_ms_marco_viT_5_1``).
    """

    def __init__(
        self,
        db_dir: str | Path = DEFAULT_DB_DIR,
        *,
        embedder: Optional[Any] = None,
        rrf_k: int = DEFAULT_RRF_K,
        device: str = "auto",
        use_flashrank: bool = True,
        flashrank_model: str = DEFAULT_FLASHRANK_MODEL,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ):
        db_dir = Path(db_dir)
        self.db_dir = db_dir
        self.rrf_k = rrf_k
        self.collection_name = collection_name

        # ---- chunks (source of truth for BM25 + return values) ----------
        chunks_path = db_dir / "chunks.jsonl"
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"chunks.jsonl not found in {db_dir}. Run ingestion.build_index first."
            )
        self.chunks: List[Dict[str, Any]] = []
        with open(chunks_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self.chunks.append(json.loads(line))
        if not self.chunks:
            raise RuntimeError(f"No chunks loaded from {chunks_path}")

        # ---- BM25 ---------------------------------------------------------
        corpus_tokens = [tokenize(c["text"]) for c in self.chunks]
        self.bm25 = _BM25Index(corpus_tokens)

        # ---- Chroma client + collection ----------------------------------
        try:
            import chromadb  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "chromadb is required for the Chroma-backed retriever. "
                "Install with: pip install chromadb"
            ) from exc

        persist_dir = db_dir / "chroma"
        if not persist_dir.exists():
            raise FileNotFoundError(
                f"Chroma directory not found at {persist_dir}. Run ingestion.build_index first."
            )
        self._chroma_client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        if self._collection.count() == 0:
            raise RuntimeError(
                f"Chroma collection '{collection_name}' is empty. Run ingestion.build_index first."
            )

        # ---- embedder -----------------------------------------------------
        if embedder is None:
            from ..ingestion.embedder import load_embedder

            embedder = load_embedder(device=device)
        self.embedder = embedder

        # ---- FlashRank (lazy) --------------------------------------------
        self.use_flashrank = use_flashrank
        self._flashrank = _FlashRanker(model_name=flashrank_model) if use_flashrank else None

    # ------------------------------------------------------------------ #
    # Public search
    # ------------------------------------------------------------------ #
    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        k_per_retriever: int = 20,
        rerank_pool: int = 20,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """Hybrid search → RRF → FlashRank rerank → top_k results.

        Parameters
        ----------
        top_k
            Final number of chunks to return.
        k_per_retriever
            Number of chunks each branch (BM25 / Dense) contributes to RRF.
        rerank_pool
            Number of RRF-fused candidates to feed into FlashRank.  Should be
            ≥ ``top_k`` and ≤ ``k_per_retriever`` (clamped internally).
        bm25_weight, dense_weight
            RRF branch weights (default 1.0 / 1.0).
        """
        if not query.strip():
            return []

        bm25_hits = self._bm25_search(query, k=k_per_retriever)
        dense_hits = self._dense_search(query, k=k_per_retriever)

        # ---- RRF fusion -------------------------------------------------
        rrf_scores: Dict[int, float] = {}
        bm25_ranks: Dict[int, int] = {}
        dense_ranks: Dict[int, int] = {}
        bm25_raw: Dict[int, float] = {}
        dense_raw: Dict[int, float] = {}

        for rank, (idx, score) in enumerate(bm25_hits, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + bm25_weight / (self.rrf_k + rank)
            bm25_ranks[idx] = rank
            bm25_raw[idx] = float(score)

        for rank, (idx, score) in enumerate(dense_hits, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + dense_weight / (self.rrf_k + rank)
            dense_ranks[idx] = rank
            dense_raw[idx] = float(score)

        ranked = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)
        pool_size = max(top_k, min(rerank_pool, len(ranked)))
        pool_idx = [idx for idx, _ in ranked[:pool_size]]

        # Build candidate dicts (deep-ish copy so we can mutate).
        candidates: List[Dict[str, Any]] = []
        for idx in pool_idx:
            chunk = dict(self.chunks[idx])
            chunk.update(
                {
                    "rrf_score": float(rrf_scores[idx]),
                    "bm25_rank": bm25_ranks.get(idx),
                    "dense_rank": dense_ranks.get(idx),
                    "bm25_score": bm25_raw.get(idx, 0.0),
                    "dense_score": dense_raw.get(idx, 0.0),
                    "row_idx": idx,
                }
            )
            candidates.append(chunk)

        # ---- Second-stage rerank (FlashRank) ----------------------------
        if self._flashrank is not None:
            reranked = self._flashrank.rerank(query, candidates, top_k=top_k)
            # Annotate final score: prefer rerank_score if available.
            for c in reranked:
                c["score"] = c.get("rerank_score", c.get("rrf_score", 0.0))
                c["reranked"] = True
        else:
            reranked = candidates[:top_k]
            for c in reranked:
                c["score"] = c.get("rrf_score", 0.0)
                c["reranked"] = False

        return reranked

    # ------------------------------------------------------------------ #
    # Internal: BM25
    # ------------------------------------------------------------------ #
    def _bm25_search(self, query: str, *, k: int) -> List[Tuple[int, float]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        k = min(k, len(scores))
        if k <= 0:
            return []
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]

    # ------------------------------------------------------------------ #
    # Internal: Dense (Chroma)
    # ------------------------------------------------------------------ #
    def _dense_search(self, query: str, *, k: int) -> List[Tuple[int, float]]:
        qv = self.embedder.encode([query], show_progress=False)
        qv = qv.astype(np.float32)
        k = min(k, self._collection.count())
        if k <= 0:
            return []

        # Chroma expects a list of floats.
        query_embedding = qv[0].tolist()
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["metadatas", "distances", "documents"],
        )

        # Map back to local chunk indices via the "idx" metadata we stored.
        out: List[Tuple[int, float]] = []
        for dist, meta in zip(result["distances"][0], result["metadatas"][0]):
            idx = int(meta.get("idx", -1))
            if idx < 0:
                continue
            # Chroma cosine space returns distance = 1 - cos_sim.
            cos_sim = 1.0 - float(dist)
            out.append((idx, cos_sim))
        return out