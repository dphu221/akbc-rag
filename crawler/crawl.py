"""Trình thu thập BFS cho https://handbook.uet.vnu.edu.vn/.

Chiến lược
----------
1. Bắt đầu từ trang chủ. Phân giải mọi ``href`` trỏ **bên trong** trang sổ tay
   (cùng máy chủ HOẶC liên kết tương đối như ``./Nội quy - quy chế/``).
2. Với mỗi trang nội bộ, tải HTML và tìm liên kết đến các tệp ``.pdf`` /
   ``.docx`` / ``.doc``. Mỗi tệp chỉ được tải **một lần** (định danh nội dung
   bằng URL) và lưu trong ``data/raw/files/``.
3. Trích xuất văn bản thuần từ:
     * mỗi trang HTML nội bộ -> ``data/raw/pages/<slug>.txt``
     * mỗi tài liệu đã tải   -> ``data/raw/files/<name>.txt``
   Mô-đun nhập liệu sẽ phân đoạn các tệp văn bản này.
4. Ghi một tệp ``data/raw/manifest.jsonl`` (mỗi tài liệu nguồn là một đối tượng
   JSON) mô tả: url, local_path, title, source_kind, n_pages, n_chars.

Các tệp PDF bên ngoài (ví dụ trên ``uet.vnu.edu.vn`` / ``vnu.edu.vn``) được
liên kết từ sổ tay cũng sẽ được tải vì chúng chính là các quy định (Quy chế
đào tạo, Quy chế sinh viên, ...). Chỉ các *trang* mới bị giới hạn trong máy
chủ của sổ tay.
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
POLITENESS_SLEEP = 0.5  # số giây giữa các yêu cầu
MAX_PAGES = 200  # giới hạn an toàn
MAX_FILES = 80  # giới hạn an toàn cho số tài liệu

DOC_EXTS = (".pdf", ".docx", ".doc")


# --------------------------------------------------------------------------- #
# Tiện ích URL
# --------------------------------------------------------------------------- #
def _normalize(url: str, base: str) -> Optional[str]:
    """Phân giải ``url`` theo ``base``; trả về URL tuyệt đối không có fragment.

    Trả về ``None`` cho liên kết mailto / javascript / chỉ có anchor.
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
    """URL là "nội bộ" (trang cần thu thập) nếu nằm trên cùng máy chủ."""
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    return host == base_host


def _is_doc_link(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith(DOC_EXTS)


def _slugify(url: str) -> str:
    """Tạo tên tệp an toàn cho hệ thống tệp từ URL."""
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    if not path:
        return "index"
    # Thay dấu phân cách, giữ lại các ký tự Unicode (tiếng Việt).
    slug = re.sub(r"[\s/]+", "_", path)
    slug = re.sub(r"[^A-Za-z0-9_\-.\u00C0-\u024F\u1E00-\u1EFF]", "", slug)
    return slug[:120] or "doc"


# --------------------------------------------------------------------------- #
# Tiện ích tải qua HTTP
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
            # Máy chủ sổ tay trả về nội dung UTF-8 nhưng không có ``charset``
            # trong ``Content-Type``, nên ``requests`` mặc định dùng ISO-8859-1,
            # làm hỏng href tiếng Việt và gây lỗi 404 ở bước sau.
            # Buộc giải mã UTF-8 đối với phản hồi văn bản.
            if not stream and resp.encoding is None or (resp.encoding or "").lower() not in (
                "utf-8",
                "utf8",
            ):
                # Chỉ ghi đè khi phần thân thực sự có dạng văn bản UTF-8.
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
# Quy trình thu thập chính
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
    # Loại bỏ nội dung nhiễu từ script / style.
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # Thu gọn dòng trống nhưng vẫn giữ ngắt đoạn.
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
    """Thu thập trang sổ tay và các quy định được liên kết.

    Trả về dict tóm tắt nhỏ gồm số lượng và các đường dẫn.
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

        # Lưu văn bản của trang HTML.
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

        # Tìm các liên kết mới.
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

    # Tạo tên tệp duy nhất: <slug>__<hash8>.<ext>
    parsed = urlparse(url)
    raw_name = Path(parsed.path).name or "document"
    raw_name = re.sub(r"[^A-Za-z0-9_\-.\u00C0-\u024F\u1E00-\u1EFF]", "_", raw_name)
    if not raw_name.lower().endswith(DOC_EXTS):
        raw_name += ".bin"
    dest = files_dir / raw_name[:160]
    if not _save_response_content(resp, dest):
        return

    # Trích xuất văn bản cạnh tệp nhị phân.
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
# Giao diện dòng lệnh
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
