"""Ingestion package: turn raw text into Chroma-indexed chunks."""

from .chunker import chunk_document, chunk_corpus  # noqa: F401
from .embedder import build_chroma_index, load_embedder  # noqa: F401

# Backwards-compatible alias kept for any external code still calling the
# old FAISS-era function name.
build_faiss_index = build_chroma_index

__all__ = [
    "chunk_document",
    "chunk_corpus",
    "build_chroma_index",
    "build_faiss_index",
    "load_embedder",
]