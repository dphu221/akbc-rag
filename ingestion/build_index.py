"""Trình điều khiển nhập liệu đầu-cuối: manifest -> các đoạn -> chỉ mục Chroma.

Chạy dưới dạng mô-đun: ``python -m ingestion.build_index``
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
    """Chạy toàn bộ quy trình nhập liệu.

    Tham số
    -------
    manifest_path
        Đường dẫn đến ``data/raw/manifest.jsonl`` do trình thu thập tạo ra.
    chunks_path
        Đường dẫn đầu ra tùy chọn cho ``chunks.jsonl``. Mặc định là
        ``data/processed/chunks.jsonl`` cạnh thư mục cha của manifest.
    db_dir
        Thư mục đầu ra cho chỉ mục Chroma và siêu dữ liệu của các đoạn.
    device
        ``"cuda"``, ``"cpu"`` hoặc ``"auto"``.
    batch_size
        Kích thước lô embedding. Hãy giảm giá trị này trên GPU nhỏ.
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


def _cli() -> None:  # pragma: no cover
    p = argparse.ArgumentParser(description="Run ingestion: manifest -> chunks -> Chroma")
    p.add_argument(
        "--manifest",
        default=str(Path(__file__).resolve().parents[1] / "data" / "raw" / "manifest.jsonl"),
    )
    p.add_argument("--chunks", default=None)
    p.add_argument("--db", default=str(DEFAULT_DB_DIR))
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    import json

    summary = run(
        manifest_path=args.manifest,
        chunks_path=args.chunks,
        db_dir=args.db,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))



if __name__ == "__main__":  # pragma: no cover
    _cli()
