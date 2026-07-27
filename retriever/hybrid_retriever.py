"""Bộ truy hồi lai: BM25 + Chroma Dense + RRF + FlashRank tái xếp hạng bước hai.

Quy trình (cho mỗi truy vấn)
---------------------------
1. **BM25** trên các token tiếng Việt đã bỏ dấu (rank_bm25) → ``k_per_retriever`` đầu.
2. **Dense** qua truy vấn collection Chroma (không gian cosine) → ``k_per_retriever`` đầu.
3. **Hợp nhất RRF** hai danh sách xếp hạng (k=60 mặc định) → các ứng viên
   ``rerank_pool`` đầu (mặc định 20). Đây là bước "Tìm kiếm lai".
4. **Tái xếp hạng bằng cross-encoder FlashRank** trên các ứng viên
   ``rerank_pool`` → ``top_k`` kết quả cuối. Bên trong FlashRank dùng
   ``ranker_ms_marco_viT_5_1`` — rất nhỏ (~120 MB) và chạy trên CPU trong
   <50 ms với 20 tài liệu.

Nếu FlashRank không khả dụng (ví dụ nhập thất bại khi chạy), bộ truy hồi sẽ
chuyển nhẹ nhàng sang đầu ra chỉ dùng RRF. Điều này giúp hệ thống vẫn chạy
trong môi trường hạn chế, đồng thời tuân thủ thiết kế "RRF + FlashRank" khi
có đủ phần phụ thuộc.
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
# Tiện ích văn bản tiếng Việt
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
    """Tách token cho BM25."""
    if lower:
        text = text.lower()
    if strip_diacritics:
        text = remove_diacritics(text)
    return _TOKEN_RE.findall(text)


# --------------------------------------------------------------------------- #
# Chỉ mục BM25 (không đổi)
# --------------------------------------------------------------------------- #
class _BM25Index:
    """Lớp bao nhẹ quanh rank_bm25.BM25Okapi."""

    def __init__(self, corpus_tokens: List[List[str]]):
        from rank_bm25 import BM25Okapi

        self.bm25 = BM25Okapi(corpus_tokens)

    def get_scores(self, query_tokens: List[str]) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(query_tokens), dtype=np.float32)


# --------------------------------------------------------------------------- #
# Bộ tái xếp hạng FlashRank (nạp lười, tùy chọn)
# --------------------------------------------------------------------------- #
class _FlashRanker:
    """Bao bọc cross-encoder ``flashrank.Ranker``.

    Mô hình được tải trong lần dùng đầu tiên (~120 MB) và lưu đệm trên đĩa.
    Khởi tạo theo kiểu nạp lười để môi trường chưa cài FlashRank vẫn hoạt động
    — chỉ bỏ qua bước tái xếp hạng.
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
        """Sắp xếp lại ``candidates`` và trả về ``top_k`` mục đầu.

        Mỗi phần tử trong ``candidates`` phải có trường ``text``. Danh sách trả
        về có cùng cấu trúc dict và được thêm trường ``rerank_score``.
        """
        if not candidates or top_k <= 0:
            return []
        if not self._ensure_loaded():
            # Không có bộ tái xếp hạng — cắt ngắn và trả về nguyên thứ tự.
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

        # results là danh sách dict có ít nhất "id" và "score".
        out: List[Dict[str, Any]] = []
        for r in results[:top_k]:
            idx = int(r["id"])
            chunk = dict(candidates[idx])
            chunk["rerank_score"] = float(r["score"])
            out.append(chunk)
        return out


# --------------------------------------------------------------------------- #
# Bộ truy hồi lai
# --------------------------------------------------------------------------- #
class HybridRetriever:
    """BM25 + Chroma Dense + RRF + FlashRank.

    Tham số
    -------
    db_dir
        Thư mục chứa thư mục con ``chroma/`` và ``chunks.jsonl``.
    embedder
        Đối tượng ``_Embedder`` đã nạp. Nếu là ``None``, sẽ tạo theo kiểu nạp lười.
    rrf_k
        Hằng số RRF (mặc định 60).
    use_flashrank
        ``True`` (mặc định) để bật tái xếp hạng FlashRank. Đặt ``False`` để
        buộc chỉ dùng RRF.
    flashrank_model
        Tên mô hình FlashRank (mặc định ``ranker_ms_marco_viT_5_1``).
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

        # ---- các đoạn (nguồn chuẩn cho BM25 + giá trị trả về) ------------
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

        # ---- client + collection Chroma ----------------------------------
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

        # ---- bộ embedding -------------------------------------------------
        if embedder is None:
            from ..ingestion.embedder import load_embedder

            embedder = load_embedder(device=device)
        self.embedder = embedder

        # ---- FlashRank (nạp lười) ----------------------------------------
        self.use_flashrank = use_flashrank
        self._flashrank = _FlashRanker(model_name=flashrank_model) if use_flashrank else None

    # ------------------------------------------------------------------ #
    # Tìm kiếm công khai
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
        """Tìm kiếm lai → RRF → FlashRank tái xếp hạng → top_k kết quả.

        Tham số
        -------
        top_k
            Số đoạn cuối cùng cần trả về.
        k_per_retriever
            Số đoạn mỗi nhánh (BM25 / Dense) đóng góp vào RRF.
        rerank_pool
            Số ứng viên đã hợp nhất RRF đưa vào FlashRank. Nên ≥ ``top_k`` và
            ≤ ``k_per_retriever`` (được giới hạn nội bộ).
        bm25_weight, dense_weight
            Trọng số các nhánh RRF (mặc định 1.0 / 1.0).
        """
        if not query.strip():
            return []

        bm25_hits = self._bm25_search(query, k=k_per_retriever)
        dense_hits = self._dense_search(query, k=k_per_retriever)

        # ---- Hợp nhất RRF ------------------------------------------------
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

        # Tạo các dict ứng viên (sao chép đủ sâu để có thể thay đổi).
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

        # ---- Tái xếp hạng bước hai (FlashRank) ---------------------------
        if self._flashrank is not None:
            reranked = self._flashrank.rerank(query, candidates, top_k=top_k)
            # Ghi điểm cuối: ưu tiên rerank_score nếu có.
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
    # Nội bộ: BM25
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
    # Nội bộ: Dense (Chroma)
    # ------------------------------------------------------------------ #
    def _dense_search(self, query: str, *, k: int) -> List[Tuple[int, float]]:
        qv = self.embedder.encode([query], show_progress=False)
        qv = qv.astype(np.float32)
        k = min(k, self._collection.count())
        if k <= 0:
            return []

        # Chroma yêu cầu một danh sách số thực.
        query_embedding = qv[0].tolist()
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["metadatas", "distances", "documents"],
        )

        # Ánh xạ về chỉ số đoạn cục bộ qua siêu dữ liệu "idx" đã lưu.
        out: List[Tuple[int, float]] = []
        for dist, meta in zip(result["distances"][0], result["metadatas"][0]):
            idx = int(meta.get("idx", -1))
            if idx < 0:
                continue
            # Không gian cosine của Chroma trả về khoảng cách = 1 - cos_sim.
            cos_sim = 1.0 - float(dist)
            out.append((idx, cos_sim))
        return out
