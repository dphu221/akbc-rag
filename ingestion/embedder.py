"""Embed chunks with BAAI/bge-m3 and build a FAISS index.

This module is the **only** place in the codebase that touches the embedding
model, so swapping the back-end later (ONNX, SentenceTransformers, dedicated
inference server, ...) only requires editing this file.

Design decisions
----------------
* Use ``FlagEmbedding`` (the official BGE library) when available because it
  supports multi-lingual + dense + sparse + colbert vectors in one call.
* Fall back to ``sentence-transformers`` if FlagEmbedding is missing — the
  dense vector is still correct, only the sparse/colbert heads are skipped.
* Run on GPU when CUDA is available; otherwise CPU (slow but works for the
  small UET corpus of a few thousand chunks).
* Persist three artefacts side-by-side under ``data/vector_db/``:
    - ``faiss.index``       : the FAISS index (Flat IP for cosine after L2 norm)
    - ``chunks.jsonl``      : the chunk metadata in the same order as the index
    - ``embedder_meta.json``: model name, dimension, normalisation flag
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-m3"
EMBED_DIM = 1024  # bge-m3 dense output dimension
DEFAULT_DB_DIR = Path(__file__).resolve().parents[1] / "data" / "vector_db"


# --------------------------------------------------------------------------- #
# Lazy model loader
# --------------------------------------------------------------------------- #
class _Embedder:
    """Thin wrapper that hides the FlagEmbedding / sentence-transformers split."""

    def __init__(self, device: str = "auto"):
        self.device = self._resolve_device(device)
        self._backend: Optional[str] = None
        self._model = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def load(self) -> None:
        if self._model is not None:
            return

        # 1) Prefer FlagEmbedding (BGEM3FlagModel)
        try:
            from FlagEmbedding import BGEM3FlagModel  # type: ignore

            logger.info("Loading %s via FlagEmbedding on %s", MODEL_NAME, self.device)
            self._model = BGEM3FlagModel(
                MODEL_NAME,
                use_fp16=(self.device == "cuda"),
                device=self.device,
            )
            self._backend = "flag"
            return
        except ImportError:
            logger.info("FlagEmbedding not installed, falling back to sentence-transformers.")
        except Exception as exc:  # pragma: no cover
            logger.warning("FlagEmbedding load failed (%s); trying sentence-transformers.", exc)

        # 2) Fallback: sentence-transformers
        from sentence_transformers import SentenceTransformer  # type: ignore

        logger.info("Loading %s via sentence-transformers on %s", MODEL_NAME, self.device)
        self._model = SentenceTransformer(MODEL_NAME, device=self.device)
        self._backend = "st"

    def encode(self, texts: List[str], *, batch_size: int = 8, show_progress: bool = True) -> np.ndarray:
        """Return an ``(N, 1024)`` float32 matrix, L2-normalised."""
        self.load()
        if self._backend == "flag":
            out = self._model.encode(
                texts,
                batch_size=batch_size,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
                show_progress=show_progress,
            )
            vecs = np.asarray(out["dense_vecs"], dtype=np.float32)
        else:
            vecs = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)

        # Ensure L2 normalised (so inner product == cosine similarity).
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms
        return vecs


def load_embedder(device: str = "auto") -> _Embedder:
    return _Embedder(device=device)


# --------------------------------------------------------------------------- #
# FAISS index build / load
# --------------------------------------------------------------------------- #
def build_faiss_index(
    chunks_path: str | Path,
    *,
    out_dir: str | Path = DEFAULT_DB_DIR,
    device: str = "auto",
    batch_size: int = 8,
) -> dict:
    """Embed every chunk and write a FAISS index next to the chunks file.

    Returns a small summary dict ``{n_chunks, dim, index_path, chunks_path}``.
    """
    import faiss  # local import: faiss is heavy and optional during dev

    chunks_path = Path(chunks_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks: List[dict] = []
    with open(chunks_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    if not chunks:
        raise RuntimeError(f"No chunks found in {chunks_path}")

    texts = [c["text"] for c in chunks]
    logger.info("Embedding %d chunks with %s ...", len(texts), MODEL_NAME)
    embedder = load_embedder(device=device)
    vectors = embedder.encode(texts, batch_size=batch_size, show_progress=True)
    n, dim = vectors.shape
    logger.info("Embedding shape: %s", (n, dim))

    # Flat inner-product index (vectors are L2-normalised => cosine).
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    logger.info("FAISS index size: %d", index.ntotal)

    index_path = out_dir / "faiss.index"
    chunks_out = out_dir / "chunks.jsonl"
    meta_out = out_dir / "embedder_meta.json"
    faiss.write_index(index, str(index_path))

    with open(chunks_out, "w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    meta = {
        "model": MODEL_NAME,
        "dim": dim,
        "n_chunks": n,
        "normalised": True,
        "metric": "cosine (IndexFlatIP on L2-normalised vectors)",
        "backend": embedder._backend,
    }
    meta_out.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "n_chunks": n,
        "dim": dim,
        "index_path": str(index_path),
        "chunks_path": str(chunks_out),
        "meta_path": str(meta_out),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _cli() -> None:  # pragma: no cover
    import argparse

    p = argparse.ArgumentParser(description="Build FAISS index from chunks.jsonl")
    p.add_argument("--chunks", required=True, help="Path to chunks.jsonl")
    p.add_argument("--out", default=str(DEFAULT_DB_DIR), help="Output dir for vector DB")
    p.add_argument("--device", default="auto", help="cuda | cpu | auto")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    summary = build_faiss_index(
        args.chunks,
        out_dir=args.out,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    _cli()
