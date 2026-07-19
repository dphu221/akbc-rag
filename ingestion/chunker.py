"""Structure-aware chunker for Vietnamese university regulations.

The handbook documents are legal-style text organised as:

    Chương 1: NHỮNG QUY ĐỊNH CHUNG
    Điều 1. Phạm vi điều chỉnh
    1. Quy chế này quy định ...
    2. Sinh viên ... phải tuân thủ ...

We want **one chunk per Điều (article)** — not a fixed-length sliding window —
because each Điều is a self-contained semantic unit.  The chunker therefore:

1. Normalises whitespace and unicode.
2. Splits the document into "Chương" blocks (best-effort regex).
3. Within each Chương, finds every "Điều <n>." anchor and groups all text up
   to the next "Điều" anchor into a single chunk.
4. If no "Điều" anchors are found (e.g. an HTML info page), falls back to
   paragraph-based chunking with a maximum character budget.
5. Each chunk is stored as a dict with:
   - ``chunk_id``   : stable hash-derived id (source + chương + điều)
   - ``article_id`` : "Điều <n>" or "PB-<k>" for paragraph fallback
   - ``chapter``    : parent chương title (if any)
   - ``source``     : filename of the original document
   - ``source_url`` : url of the original document (if known)
   - ``text``       : the chunk body
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
# Regex patterns
# --------------------------------------------------------------------------- #
# Match "Chương 1", "Chương I", "Chương 12:", "CHƯƠNG 3.", "Chương 4 - Tên"
CHAPTER_RE = re.compile(
    r"""^\s*chương\s+          # prefix
        ([0-9IVXLCDM]+)         # number (arabic or roman)
        \s*[:\.\-–—]?\s*        # optional separator
        ([^\n]*)                # optional title
    $""",
    re.IGNORECASE | re.VERBOSE | re.MULTILINE,
)

# Match "Điều 1.", "Điều 12.", "ĐIỀU 3.", "Điều 4:" (optionally followed by title)
ARTICLE_RE = re.compile(
    r"""(?m)^\s*điều\s+        # prefix (must be at line start)
        (\d{1,3})               # article number (1-3 digits)
        \s*[:\.\-–—]?\s*        # optional separator
        ([^\n]*)                # optional title (the rest of the line)
    """,
    re.IGNORECASE | re.VERBOSE,
)

MAX_FALLBACK_CHARS = 1200
FALLBACK_OVERLAP = 150


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _normalize_text(text: str) -> str:
    """Collapse runs of whitespace, normalise line breaks."""
    # Replace Windows newlines.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove zero-width / soft hyphens that show up in PDF extractions.
    text = text.replace("\u200b", "").replace("\u00ad", "")
    # Trim trailing spaces on each line.
    lines = [ln.rstrip() for ln in text.splitlines()]
    # Collapse 3+ blank lines into 1.
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
    # Prepend the article title to the body so the embedding sees the topic.
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
# Article-based chunking
# --------------------------------------------------------------------------- #
def _split_by_chapter(text: str) -> List[Tuple[str, str]]:
    """Split ``text`` into (chapter_title, chapter_body) pairs.

    If no chapters are detected, returns a single pair with empty title.
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

    # Keep any leading text before the first chapter as its own block.
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
# Fallback: paragraph-based chunking
# --------------------------------------------------------------------------- #
def _chunk_paragraph_fallback(
    text: str,
    *,
    source: str,
    source_url: Optional[str],
    chapter: str = "",
) -> List[Dict[str, object]]:
    """Used when no Điều structure is detected (HTML info pages, plain prose)."""
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
# Public API
# --------------------------------------------------------------------------- #
def chunk_document(
    text: str,
    *,
    source: str,
    source_url: Optional[str] = None,
) -> List[Dict[str, object]]:
    """Chunk a single document (string) into structured chunks.

    Returns a list of chunk dicts.  See module docstring for the schema.
    """
    text = _normalize_text(text)
    if not text:
        return []

    # Try article-based chunking per chapter first.
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
            # Chapter had no Điều anchors — fall back to paragraphs within it.
            all_chunks.extend(
                _chunk_paragraph_fallback(
                    chapter_body,
                    source=source,
                    source_url=source_url,
                    chapter=chapter_title,
                )
            )

    if not article_chunks_found and not all_chunks:
        # Whole document had neither Chương nor Điều — pure paragraph mode.
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
    """Read the crawler manifest and chunk every referenced document.

    ``manifest_path`` is a JSONL file written by ``crawler.run_crawl``.
    Returns the full list of chunks and (optionally) writes them to
    ``out_path`` as JSONL.
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
            source = text_path.stem  # use stem (no extension) as source name
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
# CLI
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
