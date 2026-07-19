"""BFS crawler for https://handbook.uet.vnu.edu.vn/.

Strategy
--------
1. Start from the homepage.  Resolve every ``href`` that points **inside** the
   handbook site (same host, OR a relative link such as ``./Nội quy - quy chế/``).
2. For every internal page we download the HTML and also look for links to
   ``.pdf`` / ``.docx`` / ``.doc`` files.  Those files are downloaded **once**
   (content-addressed by URL) and saved under ``data/raw/files/``.
3. We extract plain text from:
     * each internal HTML page   -> ``data/raw/pages/<slug>.txt``
     * each downloaded document  -> ``data/raw/files/<name>.txt``
   The text files are what the ingestion module chunks.
4. We write a single ``data/raw/manifest.jsonl`` (one JSON object per source
   document) describing: url, local_path, title, source_kind, n_pages, n_chars.

External PDFs (e.g. on ``uet.vnu.edu.vn`` / ``vnu.edu.vn``) that are linked
from the handbook are also downloaded because they ARE the regulations
(Quy chế đào tạo, Quy chế sinh viên, ...).  Only the *pages* are restricted to
the handbook host itself.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .pdf_docx_extractor import extract_text_from_file

logger = logging.getLogger(__name__)

DEFAULT_BASE = "https://handbook.uet.vnu.edu.vn/"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "raw"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 UET-RAG-Crawler/1.0"
)
HTML_TIMEOUT = 20
FILE_TIMEOUT = 90
POLITENESS_SLEEP = 0.5  # seconds between requests
MAX_PAGES = 200  # safety cap
MAX_FILES = 80  # safety cap on documents

DOC_EXTS = (".pdf", ".docx", ".doc")


# --------------------------------------------------------------------------- #
# URL helpers
# --------------------------------------------------------------------------- #
def _normalize(url: str, base: str) -> Optional[str]:
    """Resolve ``url`` against ``base``; return absolute URL without fragment.

    Returns ``None`` for mailto / javascript / anchor-only links.
    """
    if not url:
        return None
    url = url.strip()
    if not url or url.startswith(("javascript:", "mailto:", "tel:", "#")):
        return None
    absolute = urljoin(base, url)
    absolute, _ = urldefrag(absolute)
    return absolute


def _is_internal(url: str, base_host: str) -> bool:
    """An URL is "internal" (a page to crawl) if it lives on the same host."""
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    return host == base_host


def _is_doc_link(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith(DOC_EXTS)


def _slugify(url: str) -> str:
    """Build a filesystem-safe filename from a URL."""
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    if not path:
        return "index"
    # Replace separators, keep unicode (Vietnamese) characters.
    slug = re.sub(r"[\s/]+", "_", path)
    slug = re.sub(r"[^A-Za-z0-9_\-.\u00C0-\u024F\u1E00-\u1EFF]", "", slug)
    return slug[:120] or "doc"


# --------------------------------------------------------------------------- #
# HTTP fetch helpers
# --------------------------------------------------------------------------- #
def _get(url: str, *, stream: bool = False, timeout: int = HTML_TIMEOUT) -> Optional[requests.Response]:
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept-Language": "vi,en;q=0.9"},
            timeout=timeout,
            stream=stream,
        )
        if resp.status_code == 200:
            # The handbook server returns UTF-8 content but no ``charset`` in
            # ``Content-Type``, so ``requests`` defaults to ISO-8859-1 which
            # turns Vietnamese hrefs into mojibake and they 404 downstream.
            # Force UTF-8 decoding for text responses.
            if not stream and resp.encoding is None or (resp.encoding or "").lower() not in (
                "utf-8",
                "utf8",
            ):
                # Only override when the body actually looks like UTF-8 text.
                ctype = (resp.headers.get("Content-Type") or "").lower()
                if "text" in ctype or "xml" in ctype or "html" in ctype or "json" in ctype:
                    resp.encoding = "utf-8"
            return resp
        logger.warning("HTTP %s for %s", resp.status_code, url)
    except requests.RequestException as exc:
        logger.warning("Request failed for %s: %s", url, exc)
    return None


def _save_response_content(resp: requests.Response, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                if chunk:
                    fh.write(chunk)
        return True
    except OSError as exc:
        logger.warning("Failed to write %s: %s", dest, exc)
        return False


# --------------------------------------------------------------------------- #
# Core crawl
# --------------------------------------------------------------------------- #
def _extract_links(html: str, base_url: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    urls: List[str] = []
    for a in soup.find_all("a", href=True):
        absolute = _normalize(a["href"], base_url)
        if absolute:
            urls.append(absolute)
    return urls


def _extract_title(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)
    return ""


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # Remove script / style noise.
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # Collapse blank lines but keep paragraph breaks.
    lines = [ln.strip() for ln in text.splitlines()]
    cleaned = "\n".join(ln for ln in lines if ln)
    return cleaned


def run_crawl(
    base_url: str = DEFAULT_BASE,
    out_dir: Path | str = DEFAULT_OUT,
    *,
    max_pages: int = MAX_PAGES,
    max_files: int = MAX_FILES,
    sleep: float = POLITENESS_SLEEP,
) -> Dict[str, object]:
    """Crawl the handbook site + linked regulations.

    Returns a small summary dict with counts and paths.
    """
    out = Path(out_dir)
    pages_dir = out / "pages"
    files_dir = out / "files"
    pages_dir.mkdir(parents=True, exist_ok=True)
    files_dir.mkdir(parents=True, exist_ok=True)

    base_host = urlparse(base_url).netloc.lower()
    visited: Set[str] = set()
    downloaded: Set[str] = set()
    queue: Deque[str] = deque([base_url])
    manifest: List[Dict[str, object]] = []

    while queue and len(visited) < max_pages:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        logger.info("[page %3d] %s", len(visited), url)
        resp = _get(url, timeout=HTML_TIMEOUT)
        time.sleep(sleep)
        if resp is None:
            continue

        # Save page HTML text.
        try:
            html = resp.text
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not decode %s: %s", url, exc)
            continue

        title = _extract_title(html)
        text = _html_to_text(html)
        slug = _slugify(url)
        text_path = pages_dir / f"{slug}.txt"
        text_path.write_text(text, encoding="utf-8")

        if text.strip():
            manifest.append(
                {
                    "url": url,
                    "local_path": str(text_path),
                    "title": title,
                    "source_kind": "html",
                    "n_pages": 1,
                    "n_chars": len(text),
                }
            )

        # Discover new links.
        for link in _extract_links(html, url):
            if _is_doc_link(link):
                if link in downloaded or len(downloaded) >= max_files:
                    continue
                downloaded.add(link)
                _download_document(link, files_dir, manifest)
                time.sleep(sleep)
            elif _is_internal(link, base_host) and link not in visited:
                queue.append(link)

    manifest_path = out / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        for entry in manifest:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    summary = {
        "pages_visited": len(visited),
        "documents_downloaded": len(downloaded),
        "manifest_path": str(manifest_path),
        "out_dir": str(out),
    }
    logger.info("Crawl finished: %s", summary)
    return summary


def _download_document(url: str, files_dir: Path, manifest: List[Dict[str, object]]) -> None:
    logger.info("  [doc] %s", url)
    resp = _get(url, stream=True, timeout=FILE_TIMEOUT)
    if resp is None:
        return

    # Build a unique filename: <slug>__<hash8>.<ext>
    parsed = urlparse(url)
    raw_name = Path(parsed.path).name or "document"
    raw_name = re.sub(r"[^A-Za-z0-9_\-.\u00C0-\u024F\u1E00-\u1EFF]", "_", raw_name)
    if not raw_name.lower().endswith(DOC_EXTS):
        raw_name += ".bin"
    dest = files_dir / raw_name[:160]
    if not _save_response_content(resp, dest):
        return

    # Extract text next to the binary.
    pages = extract_text_from_file(str(dest))
    full_text = "\n\n".join(pages)
    text_path = dest.with_suffix(dest.suffix + ".txt")
    text_path.write_text(full_text, encoding="utf-8")

    manifest.append(
        {
            "url": url,
            "local_path": str(text_path),
            "binary_path": str(dest),
            "title": dest.stem,
            "source_kind": dest.suffix.lower().lstrip("."),
            "n_pages": len(pages),
            "n_chars": len(full_text),
        }
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _cli() -> None:  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Crawl the UET handbook site.")
    parser.add_argument("--base", default=DEFAULT_BASE, help="Base URL to crawl.")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output directory for raw data (default: data/raw).",
    )
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES)
    parser.add_argument("--max-files", type=int, default=MAX_FILES)
    parser.add_argument(
        "--sleep", type=float, default=POLITENESS_SLEEP, help="Politeness delay (seconds)."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    summary = run_crawl(
        base_url=args.base,
        out_dir=args.out,
        max_pages=args.max_pages,
        max_files=args.max_files,
        sleep=args.sleep,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    _cli()
