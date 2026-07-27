"""Gói truy hồi: tìm kiếm lai (BM25 + Dense bge-m3) với hợp nhất RRF."""

from .hybrid_retriever import HybridRetriever  # noqa: F401

__all__ = ["HybridRetriever"]
