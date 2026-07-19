"""Retriever package: Hybrid Search (BM25 + Dense bge-m3) with RRF fusion."""

from .hybrid_retriever import HybridRetriever  # noqa: F401

__all__ = ["HybridRetriever"]