"""Mô-đun thu thập dữ liệu Sổ tay Sinh viên UET (https://handbook.uet.vnu.edu.vn/).

Gói này chịu trách nhiệm:
  1. Tìm mọi trang nội bộ của sổ tay (BFS trên các href cục bộ).
  2. Tải mọi tệp PDF / DOCX / DOC được liên kết (cả nội bộ lẫn bên ngoài).
  3. Trích xuất văn bản thô từ trang HTML và tài liệu đã tải.
  4. Ghi mọi thứ vào ``data/raw/`` để mô-đun nhập liệu phân đoạn.

Điểm vào công khai: :func:`crawler.crawl.run_crawl`.
"""

from .crawl import run_crawl  # noqa: F401
from .pdf_docx_extractor import extract_text_from_file  # noqa: F401

__all__ = ["run_crawl", "extract_text_from_file"]
