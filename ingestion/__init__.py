"""Gói nhập liệu: chuyển văn bản thô thành các đoạn được lập chỉ mục bằng Chroma."""

from .chunker import chunk_document, chunk_corpus  # noqa: F401
from .embedder import build_chroma_index, load_embedder  # noqa: F401

# Giữ bí danh tương thích ngược cho mã bên ngoài vẫn gọi tên hàm cũ
# từ thời còn sử dụng FAISS.
build_faiss_index = build_chroma_index

__all__ = [
    "chunk_document",
    "chunk_corpus",
    "build_chroma_index",
    "build_faiss_index",
    "load_embedder",
]
