"""Crawler module for the UET Student Handbook (https://handbook.uet.vnu.edu.vn/).

This package is responsible for:
  1. Discovering all internal pages of the handbook site (BFS over local hrefs).
  2. Downloading every linked PDF / DOCX / DOC file (both internal and external).
  3. Extracting raw text from HTML pages and from downloaded documents.
  4. Writing everything to ``data/raw/`` so the ingestion module can chunk it.

Public entry-point: :func:`crawler.crawl.run_crawl`.
"""

from .crawl import run_crawl  # noqa: F401
from .pdf_docx_extractor import extract_text_from_file  # noqa: F401

__all__ = ["run_crawl", "extract_text_from_file"]
