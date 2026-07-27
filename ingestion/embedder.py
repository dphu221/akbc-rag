"""Embedding các đoạn bằng BAAI/bge-m3 và tạo chỉ mục Chroma.

Đây là nơi **duy nhất** trong mã nguồn làm việc với mô hình embedding, vì vậy
việc thay backend sau này (ONNX, SentenceTransformers, máy chủ suy luận riêng,
...) chỉ yêu cầu sửa tệp này.

Quyết định thiết kế
-------------------
* Dùng ``FlagEmbedding`` (thư viện BGE chính thức) khi có vì hỗ trợ vector đa
  ngôn ngữ + dense + sparse + colbert trong một lần gọi.
* Dự phòng bằng ``sentence-transformers`` nếu thiếu FlagEmbedding — vector
  dense vẫn chính xác, chỉ bỏ qua các đầu sparse/colbert.
* Chạy trên GPU khi có CUDA; nếu không thì dùng CPU (chậm nhưng vẫn phù hợp
  với tập dữ liệu UET nhỏ gồm vài nghìn đoạn).
* Lưu bền vững ba thành phần cạnh nhau trong ``data/vector_db/``:
    - ``chroma/``           : thư mục chỉ mục Chroma SQLite & HNSW
    - ``chunks.jsonl``      : siêu dữ liệu đoạn theo cùng thứ tự với chỉ mục
    - ``embedder_meta.json``: tên mô hình, số chiều, cờ chuẩn hóa
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
EMBED_DIM = 1024  # số chiều đầu ra dense của bge-m3
DEFAULT_DB_DIR = Path(__file__).resolve().parents[1] / "data" / "vector_db"


# --------------------------------------------------------------------------- #
# Trình nạp mô hình lười
# --------------------------------------------------------------------------- #
class _Embedder:
    """Lớp bao mỏng che giấu khác biệt giữa FlagEmbedding và sentence-transformers."""

    def __init__(self, device: str = "auto"):
        self.device = self._resolve_device(device)
        self._backend: Optional[str] = None
        self._model = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "cuda":
            return "cuda:0"
        if device != "auto":
            return device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda:0"
        except Exception:
            pass
        return "cpu"

    def load(self) -> None:
        if self._model is not None:
            return

        # 1) Ưu tiên FlagEmbedding (BGEM3FlagModel)
        try:
            from FlagEmbedding import BGEM3FlagModel  # type: ignore

            logger.info("Loading %s via FlagEmbedding on %s", MODEL_NAME, self.device)
            self._model = BGEM3FlagModel(
                MODEL_NAME,
                use_fp16=("cuda" in self.device),
                devices=self.device,
            )
            self._backend = "flag"
            return
        except ImportError:
            logger.info("FlagEmbedding not installed, falling back to sentence-transformers.")
        except Exception as exc:  # pragma: no cover
            logger.warning("FlagEmbedding load failed (%s); trying sentence-transformers.", exc)

        # 2) Dự phòng: sentence-transformers
        from sentence_transformers import SentenceTransformer  # type: ignore

        logger.info("Loading %s via sentence-transformers on %s", MODEL_NAME, self.device)
        self._model = SentenceTransformer(MODEL_NAME, device=self.device)
        self._backend = "st"

    def encode(self, texts: List[str], *, batch_size: int = 8, show_progress: bool = True) -> np.ndarray:
        """Trả về ma trận float32 ``(N, 1024)`` đã chuẩn hóa L2."""
        self.load()
        if self._backend == "flag":
            out = self._model.encode(
                texts,
                batch_size=batch_size,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
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

        # Bảo đảm chuẩn hóa L2 (để tích vô hướng == độ tương đồng cosine).
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms
        return vecs


def load_embedder(device: str = "auto") -> _Embedder:
    return _Embedder(device=device)


# --------------------------------------------------------------------------- #
# Tạo chỉ mục Chroma
# --------------------------------------------------------------------------- #
def build_chroma_index(
    chunks_path: str | Path,
    *,
    out_dir: str | Path = DEFAULT_DB_DIR,
    device: str = "auto",
    batch_size: int = 8,
    collection_name: str = "uet_handbook",
) -> dict:
    """Embedding mọi đoạn và ghi chỉ mục Chroma cạnh tệp các đoạn.

    Trả về dict tóm tắt nhỏ ``{n_chunks, dim, chroma_dir, chunks_path}``.
    """
    import chromadb  # nhập cục bộ

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

    # Khởi tạo PersistentClient của Chroma
    chroma_dir = out_dir / "chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))

    # Lấy hoặc tạo collection dùng độ tương đồng cosine
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Đặt lại collection nếu nó đã chứa tài liệu
    if collection.count() > 0:
        logger.info("Collection '%s' already exists and is not empty. Resetting it...", collection_name)
        chroma_client.delete_collection(collection_name)
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # Chuẩn bị các danh sách để chèn theo lô
    ids = [f"chunk_{i}" for i in range(n)]
    embeddings = vectors.tolist()
    
    metadatas = []
    for i, c in enumerate(chunks):
        meta = {
            "chunk_id": c.get("chunk_id", f"chunk_{i}"),
            "article_id": str(c.get("article_id", "")),
            "chapter": str(c.get("chapter", "")),
            "source": c.get("source", ""),
            "source_url": c.get("source_url", ""),
            "idx": i,
        }
        metadatas.append(meta)

    # Chèn các mục theo lô
    chroma_batch_size = 500
    for start_idx in range(0, n, chroma_batch_size):
        end_idx = min(start_idx + chroma_batch_size, n)
        collection.add(
            ids=ids[start_idx:end_idx],
            embeddings=embeddings[start_idx:end_idx],
            metadatas=metadatas[start_idx:end_idx],
            documents=texts[start_idx:end_idx],
        )
    logger.info("Chroma collection size: %d", collection.count())

    chunks_out = out_dir / "chunks.jsonl"
    meta_out = out_dir / "embedder_meta.json"

    with open(chunks_out, "w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    meta = {
        "model": MODEL_NAME,
        "dim": dim,
        "n_chunks": n,
        "normalised": True,
        "metric": "cosine (hnsw:space = cosine in Chroma)",
        "backend": embedder._backend,
    }
    meta_out.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "n_chunks": n,
        "dim": dim,
        "chroma_dir": str(chroma_dir),
        "chunks_path": str(chunks_out),
        "meta_path": str(meta_out),
    }


# --------------------------------------------------------------------------- #
# Giao diện dòng lệnh
# --------------------------------------------------------------------------- #
def _cli() -> None:  # pragma: no cover
    import argparse

    p = argparse.ArgumentParser(description="Build Chroma index from chunks.jsonl")
    p.add_argument("--chunks", required=True, help="Path to chunks.jsonl")
    p.add_argument("--out", default=str(DEFAULT_DB_DIR), help="Output dir for vector DB")
    p.add_argument("--device", default="auto", help="cuda | cpu | auto")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--collection-name", default="uet_handbook", help="Chroma collection name")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    summary = build_chroma_index(
        args.chunks,
        out_dir=args.out,
        device=args.device,
        batch_size=args.batch_size,
        collection_name=args.collection_name,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    _cli()
