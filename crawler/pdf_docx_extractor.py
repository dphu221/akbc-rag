"""Extract plain text from PDF / DOCX / DOC files.

We try multiple back-ends (in this order):
  * PDF  -> ``pdfplumber`` (best for Vietnamese), fallback ``PyPDF2``, fallback ``pymupdf``.
  * DOCX -> ``python-docx``.
  * DOC  -> ``antiword`` (system binary) if available, otherwise skip.

The function returns a *list of pages* (each page is a string). For DOCX the
whole document is returned as a single "page" because DOCX has no real page
concept. Callers that do not care about page boundaries can simply
``"\n\n".join(pages)``.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def _extract_pdf(path: str) -> List[str]:
    pages: List[str] = []
    # 1) pdfplumber
    try:
        import pdfplumber  # type: ignore

        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                pages.append(txt)
        if any(p.strip() for p in pages):
            return pages
    except Exception as exc:  # pragma: no cover - best effort
        logger.debug("pdfplumber failed on %s: %s", path, exc)

    # 2) PyPDF2
    try:
        from PyPDF2 import PdfReader  # type: ignore

        reader = PdfReader(path)
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        if any(p.strip() for p in pages):
            return pages
    except Exception as exc:  # pragma: no cover
        logger.debug("PyPDF2 failed on %s: %s", path, exc)

    # 3) pymupdf (fitz)
    try:
        import fitz  # type: ignore

        doc = fitz.open(path)
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        if any(p.strip() for p in pages):
            return pages
    except Exception as exc:  # pragma: no cover
        logger.debug("pymupdf failed on %s: %s", path, exc)

    logger.warning("All PDF back-ends failed for %s", path)
    return pages or [""]


def _extract_docx(path: str) -> List[str]:
    try:
        from docx import Document  # type: ignore

        doc = Document(path)
        paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        # Also pull text from tables (regulations often use tables).
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    txt = cell.text.strip()
                    if txt:
                        paragraphs.append(txt)
        return ["\n".join(paragraphs)]
    except Exception as exc:  # pragma: no cover
        logger.warning("python-docx failed on %s: %s", path, exc)
        return [""]


def _extract_doc(path: str) -> List[str]:
    """Extract text from legacy .doc using antiword if available."""
    try:
        result = subprocess.run(
            ["antiword", path], capture_output=True, timeout=60, check=False
        )
        if result.returncode == 0:
            txt = result.stdout.decode("utf-8", errors="ignore")
            if txt.strip():
                return [txt]
    except FileNotFoundError:
        logger.warning("antiword not installed - skipping .doc file: %s", path)
    except Exception as exc:  # pragma: no cover
        logger.warning("antiword failed on %s: %s", path, exc)
    return [""]


def extract_text_from_file(path: str) -> List[str]:
    """Return a list of page-text strings for the given document.

    Returns ``[""]`` (one empty page) when extraction fails entirely, so callers
    can iterate without special-casing.
    """
    p = Path(path)
    if not p.exists():
        logger.warning("File does not exist: %s", path)
        return [""]

    ext = p.suffix.lower()
    if ext == ".pdf":
        return _extract_pdf(path)
    if ext == ".docx":
        return _extract_docx(path)
    if ext == ".doc":
        return _extract_doc(path)
    logger.warning("Unsupported file extension: %s", path)
    return [""]


def extract_text(path: str) -> str:
    """Convenience wrapper that returns a single concatenated string."""
    return "\n\n".join(extract_text_from_file(path))