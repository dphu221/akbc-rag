"""Trình phân đoạn nhận biết cấu trúc cho các quy định đại học Việt Nam.

Tài liệu sổ tay là văn bản kiểu pháp lý được tổ chức như sau:

    Chương 1: NHỮNG QUY ĐỊNH CHUNG
    Điều 1. Phạm vi điều chỉnh
    1. Quy chế này quy định ...
    2. Sinh viên ... phải tuân thủ ...

Cần tạo **một đoạn cho mỗi Điều** — không dùng cửa sổ trượt có độ dài cố định —
vì mỗi Điều là một đơn vị ngữ nghĩa độc lập. Do đó trình phân đoạn:

1. Chuẩn hóa khoảng trắng và Unicode.
2. Chia tài liệu thành các khối "Chương" (regex cố gắng tối đa).
3. Trong mỗi Chương, tìm mọi mốc "Điều <n>." và gom toàn bộ văn bản đến mốc
   "Điều" tiếp theo thành một đoạn.
4. Nếu không tìm thấy mốc "Điều" (ví dụ trang thông tin HTML), chuyển sang
   phân đoạn theo đoạn văn với giới hạn ký tự tối đa.
5. Mỗi đoạn được lưu dưới dạng dict gồm:
   - ``chunk_id``   : mã ổn định tạo từ hàm băm (nguồn + chương + điều)
   - ``article_id`` : "Điều <n>" hoặc "PB-<k>" khi dự phòng theo đoạn văn
   - ``chapter``    : tiêu đề chương cha (nếu có)
   - ``source``     : tên tệp của tài liệu gốc
   - ``source_url`` : URL của tài liệu gốc (nếu biết)
   - ``text``       : nội dung đoạn
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Các mẫu regex
# --------------------------------------------------------------------------- #
# Khớp "Chương 1", "Chương I", "Chương 12:", "CHƯƠNG 3.", "Chương 4 - Tên"
CHAPTER_RE = re.compile(
    r"""^\s*chương\s+          # tiền tố
        ([0-9IVXLCDM]+)         # số (Ả Rập hoặc La Mã)
        \s*[:\.\-–—]?\s*        # dấu phân cách tùy chọn
        ([^\n]*)                # tiêu đề tùy chọn
    $""",
    re.IGNORECASE | re.VERBOSE | re.MULTILINE,
)

# Khớp "Điều 1.", "Điều 12.", "ĐIỀU 3.", "Điều 4:" (có thể kèm tiêu đề)
ARTICLE_RE = re.compile(
    r"""(?m)^\s*điều\s+        # tiền tố (phải ở đầu dòng)
        (\d{1,3})               # số điều (1-3 chữ số)
        \s*[:\.\-–—]?\s*        # dấu phân cách tùy chọn
        ([^\n]*)                # tiêu đề tùy chọn (phần còn lại của dòng)
    """,
    re.IGNORECASE | re.VERBOSE,
)

MAX_FALLBACK_CHARS = 1200
FALLBACK_OVERLAP = 150


# --------------------------------------------------------------------------- #
# Các hàm tiện ích
# --------------------------------------------------------------------------- #
def _normalize_text(text: str) -> str:
    """Thu gọn khoảng trắng liên tiếp và chuẩn hóa dấu ngắt dòng."""
    # Thay dấu xuống dòng kiểu Windows.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Loại bỏ ký tự độ rộng bằng không / gạch nối mềm xuất hiện khi trích PDF.
    text = text.replace("\u200b", "").replace("\u00ad", "")
    # Xóa khoảng trắng cuối mỗi dòng.
    lines = [ln.rstrip() for ln in text.splitlines()]
    # Thu gọn từ 3 dòng trống liên tiếp trở lên còn 1.
    cleaned: List[str] = []
    blank_run = 0
    for ln in lines:
        if ln.strip() == "":
            blank_run += 1
            if blank_run <= 1:
                cleaned.append("")
        else:
            blank_run = 0
            cleaned.append(ln)
    return "\n".join(cleaned).strip()


def _stable_id(source: str, article_id: str) -> str:
    raw = f"{source}::{article_id}".encode("utf-8")
    h = hashlib.md5(raw).hexdigest()[:8]
    safe_source = re.sub(r"[^A-Za-z0-9_\-.\u00C0-\u024F\u1E00-\u1EFF]", "_", source)
    return f"{safe_source}__{article_id}__{h}"


def _build_chunk(
    *,
    source: str,
    source_url: Optional[str],
    chapter: str,
    article_id: str,
    article_title: str,
    body: str,
) -> Optional[Dict[str, object]]:
    body = body.strip()
    if not body:
        return None
    # Thêm tiêu đề điều vào trước nội dung để embedding nhận biết chủ đề.
    full_text = f"{article_id}. {article_title}\n{body}".strip() if article_title else f"{article_id}.\n{body}"
    chunk = {
        "chunk_id": _stable_id(source, article_id),
        "article_id": article_id,
        "article_title": article_title.strip(),
        "chapter": chapter.strip(),
        "source": source,
        "source_url": source_url or "",
        "text": full_text,
    }
    return chunk


# --------------------------------------------------------------------------- #
# Phân đoạn theo Điều
# --------------------------------------------------------------------------- #
def _split_by_chapter(text: str) -> List[Tuple[str, str]]:
    """Chia ``text`` thành các cặp (tiêu đề chương, nội dung chương).

    Nếu không phát hiện chương, trả về một cặp duy nhất có tiêu đề trống.
    """
    matches = list(CHAPTER_RE.finditer(text))
    if not matches:
        return [("", text)]

    pairs: List[Tuple[str, str]] = []
    for i, m in enumerate(matches):
        num = m.group(1).strip()
        title = m.group(2).strip()
        chapter_title = f"Chương {num}" + (f": {title}" if title else "")
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            pairs.append((chapter_title, body))

    # Giữ phần văn bản trước chương đầu tiên thành một khối riêng.
    head = text[: matches[0].start()].strip()
    if head:
        pairs.insert(0, ("", head))
    return pairs


def _chunk_by_articles(
    body: str,
    *,
    source: str,
    source_url: Optional[str],
    chapter: str,
) -> List[Dict[str, object]]:
    matches = list(ARTICLE_RE.finditer(body))
    if not matches:
        return []

    chunks: List[Dict[str, object]] = []
    for i, m in enumerate(matches):
        num = m.group(1).strip()
        title = m.group(2).strip()
        article_id = f"Điều {num}"
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        article_body = body[start:end].strip()
        chunk = _build_chunk(
            source=source,
            source_url=source_url,
            chapter=chapter,
            article_id=article_id,
            article_title=title,
            body=article_body,
        )
        if chunk:
            chunks.append(chunk)
    return chunks


# --------------------------------------------------------------------------- #
# Dự phòng: phân đoạn theo đoạn văn
# --------------------------------------------------------------------------- #
def _chunk_paragraph_fallback(
    text: str,
    *,
    source: str,
    source_url: Optional[str],
    chapter: str = "",
) -> List[Dict[str, object]]:
    """Dùng khi không phát hiện cấu trúc Điều (trang HTML, văn xuôi thuần)."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[Dict[str, object]] = []
    buffer: List[str] = []
    buffer_chars = 0
    pb_index = 1

    def _flush() -> None:
        nonlocal buffer, buffer_chars, pb_index
        if not buffer:
            return
        body = "\n\n".join(buffer)
        article_id = f"PB-{pb_index}"
        chunk = _build_chunk(
            source=source,
            source_url=source_url,
            chapter=chapter,
            article_id=article_id,
            article_title="",
            body=body,
        )
        if chunk:
            chunks.append(chunk)
            pb_index += 1
        buffer = []
        buffer_chars = 0

    for para in paragraphs:
        if buffer_chars + len(para) > MAX_FALLBACK_CHARS and buffer:
            _flush()
        buffer.append(para)
        buffer_chars += len(para) + 2
    _flush()
    return chunks


# --------------------------------------------------------------------------- #
# API công khai
# --------------------------------------------------------------------------- #
def chunk_document(
    text: str,
    *,
    source: str,
    source_url: Optional[str] = None,
) -> List[Dict[str, object]]:
    """Chia một tài liệu (chuỗi) thành các đoạn có cấu trúc.

    Trả về danh sách dict của các đoạn. Xem docstring mô-đun để biết schema.
    """
    text = _normalize_text(text)
    if not text:
        return []

    # Trước tiên thử phân đoạn theo Điều trong từng chương.
    all_chunks: List[Dict[str, object]] = []
    article_chunks_found = False
    for chapter_title, chapter_body in _split_by_chapter(text):
        article_chunks = _chunk_by_articles(
            chapter_body,
            source=source,
            source_url=source_url,
            chapter=chapter_title,
        )
        if article_chunks:
            article_chunks_found = True
            all_chunks.extend(article_chunks)
        else:
            # Chương không có mốc Điều — chuyển sang các đoạn văn trong chương.
            all_chunks.extend(
                _chunk_paragraph_fallback(
                    chapter_body,
                    source=source,
                    source_url=source_url,
                    chapter=chapter_title,
                )
            )

    if not article_chunks_found and not all_chunks:
        # Toàn bộ tài liệu không có Chương hay Điều — chỉ dùng chế độ đoạn văn.
        all_chunks = _chunk_paragraph_fallback(
            text, source=source, source_url=source_url, chapter=""
        )

    logger.debug("chunked %s -> %d chunks", source, len(all_chunks))
    return all_chunks


def chunk_corpus(
    manifest_path: str | Path,
    *,
    out_path: str | Path | None = None,
) -> List[Dict[str, object]]:
    """Đọc manifest của trình thu thập và phân đoạn mọi tài liệu được tham chiếu.

    ``manifest_path`` là tệp JSONL do ``crawler.run_crawl`` ghi. Trả về danh
    sách đầy đủ các đoạn và (tùy chọn) ghi chúng vào ``out_path`` dạng JSONL.
    """
    manifest_path = Path(manifest_path)
    if out_path is None:
        out_path = manifest_path.parent.parent / "processed" / "chunks.jsonl"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunks: List[Dict[str, object]] = []
    with open(manifest_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            text_path = Path(entry["local_path"])
            if not text_path.exists():
                logger.warning("Missing text file: %s", text_path)
                continue
            text = text_path.read_text(encoding="utf-8", errors="ignore")
            source = text_path.stem  # dùng tên gốc (không có đuôi) làm tên nguồn
            doc_chunks = chunk_document(
                text,
                source=source,
                source_url=entry.get("url"),
            )
            for c in doc_chunks:
                c["title"] = entry.get("title", "") or c.get("article_title", "")
                c["source_kind"] = entry.get("source_kind", "")
            chunks.extend(doc_chunks)

    with open(out_path, "w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    logger.info("Wrote %d chunks to %s", len(chunks), out_path)
    return chunks


# --------------------------------------------------------------------------- #
# Giao diện dòng lệnh
# --------------------------------------------------------------------------- #
def _cli() -> None:  # pragma: no cover
    import argparse

    p = argparse.ArgumentParser(description="Chunk raw documents into structured chunks.")
    p.add_argument("--manifest", required=True, help="Path to crawler manifest.jsonl")
    p.add_argument("--out", default=None, help="Output chunks.jsonl path")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    chunks = chunk_corpus(args.manifest, out_path=args.out)
    print(f"Produced {len(chunks)} chunks.")


if __name__ == "__main__":  # pragma: no cover
    _cli()
