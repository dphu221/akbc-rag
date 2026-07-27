"""End-to-end ingestion driver: manifest -> chunks -> FAISS index.

Run as a module:  ``python -m ingestion.build_index``
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .chunker import chunk_corpus
from .embedder import build_chroma_index, DEFAULT_DB_DIR

logger = logging.getLogger(__name__)


def run(
    manifest_path: str | Path,
    *,
    chunks_path: str | Path | None = None,
    db_dir: str | Path = DEFAULT_DB_DIR,
    device: str = "auto",
    batch_size: int = 8,
) -> dict:
    """Run the full ingestion pipeline.

    Parameters
    ----------
    manifest_path
        Path to ``data/raw/manifest.jsonl`` produced by the crawler.
    chunks_path
        Optional explicit output path for ``chunks.jsonl``.  Defaults to
        ``data/processed/chunks.jsonl`` next to the manifest's parent.
    db_dir
        Output directory for the FAISS index + chunk metadata.
    device
        ``"cuda"``, ``"cpu"``, or ``"auto"``.
    batch_size
        Embedding batch size.  Lower this on small GPUs.
    """
    manifest_path = Path(manifest_path)
    if chunks_path is None:
        chunks_path = manifest_path.parent.parent / "processed" / "chunks.jsonl"
    chunks_path = Path(chunks_path)

    logger.info("Step 1/2 - chunking ...")
    chunks = chunk_corpus(manifest_path, out_path=chunks_path)
    logger.info("  -> %d chunks written to %s", len(chunks), chunks_path)

    logger.info("Step 2/2 - embedding + Chroma index ...")
    summary = build_chroma_index(
        chunks_path,
        out_dir=db_dir,
        device=device,
        batch_size=batch_size,
    )
    summary["chunks_path"] = str(chunks_path)
    summary["n_chunks"] = len(chunks)
    return summary


