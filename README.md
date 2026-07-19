# UET RAG Chatbot — Sổ tay Sinh viên

End-to-end RAG application that lets students of **Trường Đại học Công nghệ (UET) — ĐHQG Hà Nội** ask natural-language questions about the student handbook, regulations, scholarships, dormitories, tuition fees, etc.

The system crawls [https://handbook.uet.vnu.edu.vn/](https://handbook.uet.vnu.edu.vn/), chunks the text by **Chương – Điều – Khoản** structure, embeds each chunk with **BAAI/bge-m3**, indexes them in **Chroma**, retrieves with **Hybrid Search (BM25 + Dense) + Reciprocal Rank Fusion + FlashRank rerank**, and answers with **Qwen2.5-3B-Instruct** through a minimalist Streamlit chat UI.

---

## 1. Kiến trúc tổng thể

```
rag-pipeline/
├── crawler/                  # Module cào và tải dữ liệu web/pdf/docx
│   ├── __init__.py
│   ├── crawl.py              # BFS crawler cho handbook.uet.vnu.edu.vn + PDF/DOCX tải về
│   └── pdf_docx_extractor.py # Trích text từ PDF/DOCX/DOC (3 back-end dự phòng)
├── ingestion/                # Module phân tách đoạn (chunking) và lập chỉ mục Vector
│   ├── __init__.py
│   ├── chunker.py            # Chunking theo Chương - Điều - Khoản (có fallback theo đoạn)
│   ├── embedder.py           # bge-m3 embedding + Chroma persistent store (cosine)
│   └── build_index.py        # Driver chạy end-to-end ingestion
├── retriever/                # Module Hybrid Search + RRF + FlashRank
│   ├── __init__.py
│   └── hybrid_retriever.py   # BM25 + Chroma Dense + RRF + FlashRank rerank
├── generation/               # Module LLM + Memory
│   ├── __init__.py
│   ├── llm.py                # Qwen2.5-3B-Instruct (4-bit NF4 trên CUDA)
│   ├── prompts.py            # System prompt tiếng Việt + anti-hallucination
│   └── memory.py             # SummaryMemory (rolling summary + recent window)
├── data/
│   ├── raw/                  # Crawler output
│   │   ├── pages/
│   │   ├── files/
│   │   └── manifest.jsonl
│   ├── processed/
│   │   └── chunks.jsonl
│   └── vector_db/
│       ├── chroma/           # Chroma SQLite + HNSW index
│       ├── chunks.jsonl
│       └── embedder_meta.json
├── app.py                    # Streamlit UI (Minimal Light style)
├── requirements.txt
└── README.md
```

---

## 2. Cài đặt

Yêu cầu **Python 3.10+** và (khuyến nghị) **GPU CUDA** để chạy mô hình LLM với tốc độ chấp nhận được.

> [!TIP]
> Để chạy dự án này trên nền tảng đám mây Kaggle hoàn toàn miễn phí (có hỗ trợ GPU T4), hãy tham khảo [Hướng dẫn chạy trên Kaggle](KAGGLE.md).


```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

pip install -r requirements.txt
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## 3. Pipeline chạy

### Bước 1 — Crawl dữ liệu

```bash
python -m crawler.crawl --out data/raw --max-pages 200 --max-files 80
```

Kết quả:
- `data/raw/pages/*.txt` — text trích từ các trang HTML nội bộ.
- `data/raw/files/*.{pdf,docx}` — file đính kèm tải về.
- `data/raw/files/*.txt` — text trích từ file đính kèm.
- `data/raw/manifest.jsonl` — manifest liệt kê mọi nguồn.

Crawler dùng BFS, polite delay 0.5 s/request, tự tải cả PDF/DOCX external (trên `uet.vnu.edu.vn`, `vnu.edu.vn`) vì đó chính là các văn bản quy chế được trích dẫn trong Sổ tay. Crawler cũng sửa lỗi encoding UTF-8: server không trả `charset` trong `Content-Type`, khiến `requests` mặc định decode ISO-8859-1 → Vietnamese URLs bị mojibake → 404.

### Bước 2 — Chunking + Embedding + Chroma

```bash
python -m ingestion.build_index \
    --manifest data/raw/manifest.jsonl \
    --db data/vector_db \
    --device auto \
    --batch-size 8
```

Thuật toán chunking (xem `ingestion/chunker.py`):

1. Tách văn bản theo **Chương** (regex: `^Chương\s+(arabic|roman)…`).
2. Trong mỗi Chương, tìm mọi anchor `^Điều\s+\d+…` và gom toàn bộ text tới Điều kế tiếp thành **một chunk**.
3. Nếu tài liệu không có cấu trúc Điều (HTML info page, văn xuông), fallback sang chunk theo đoạn với `MAX_FALLBACK_CHARS = 1200` ký tự.
4. Mỗi chunk có `chunk_id`, `article_id`, `article_title`, `chapter`, `source`, `source_url`, `text` — đúng schema yêu cầu trong đề bài.

Mỗi chunk được nhúng bằng **BAAI/bge-m3** (chiều 1024, L2-normalised), lưu vào **Chroma** collection với `hnsw:space=cosine`. Metadata gồm `chunk_id`, `article_id`, `chapter`, `source`, `source_url`, và `idx` (chỉ số dòng trong `chunks.jsonl`) để tra ngược khi retrieve.

### Bước 3 — Khởi động Chatbot

```bash
streamlit run app.py
```

Trình duyệt sẽ tự mở `http://localhost:8501`. Trong UI bạn có thể:

- Đặt câu hỏi, ví dụ: *"Điều kiện xét tốt nghiệp đại học hệ chính quy là gì?"*
- Hỏi nối tiếp (multi-turn): *"Mức học bổng loại xuất sắc là bao nhiêu?"*
- Tinh chỉnh `top_k`, trọng số BM25/Dense, hằng số RRF, bật/tắt FlashRank trong sidebar.
- Mở mục **Nguồn trích dẫn** để xem chính xác chunk nào được dùng, kèm `article_id`, `chapter`, `source`, `URL`, điểm RRF, điểm rerank, và snippet.
- Mở mục **Tóm tắt hội thoại (đang lưu)** để xem bản tóm tắt do LLM sinh ra cho các lượt cũ.
- Bấm **"Xoá lịch sử chat"** để bắt đầu phiên mới.

---

## 4. Thiết kế kỹ thuật trọng điểm

### 4.1 Hybrid Search + RRF + FlashRank

`retriever/hybrid_retriever.py` chạy hai nhánh song song:

| Nhánh | Mục đích | Cài đặt |
|---|---|---|
| **BM25** | Khớp chính xác từ chuyên ngành (mã môn, tên học kỳ, "Điều 16") | `rank_bm25.BM25Okapi` trên token không dấu + lowercase |
| **Dense (bge-m3)** | Hiểu ngữ nghĩa, paraphrase, từ đồng nghĩa | Chroma collection (cosine) |

Mỗi nhánh trả về `k_per_retriever = 20` kết quả, sau đó hợp nhất bằng **Reciprocal Rank Fusion**:

```
RRF_score(d) = w_bm25 / (k + rank_bm25(d)) + w_dense / (k + rank_dense(d))
```

Sau RRF, top `rerank_pool = 20` candidates được đưa vào **FlashRank** (cross-encoder `ranker_ms_marco_viT_5_1`, ~120 MB, chạy CPU) để rerank lấy `top_k = 5`. FlashRank cải thiện precision@5 đáng kể, đặc biệt khi BM25 và Dense bị conflict về thứ hạng.

Nếu FlashRank không được cài (hoặc lỗi runtime), retriever tự fallback sang RRF-only — hệ thống vẫn chạy được, chỉ mất lớp rerank cuối.

Tokenizer cho BM25 strip dấu tiếng Việt để truy vấn không dấu vẫn match — sinh viên hay gõ "quy che dao tao" thay vì "quy chế đào tạo".

### 4.2 Summary Memory

`generation/memory.py` thay sliding-window đơn giản bằng **summary memory**:

* **Recent window**: `recent_turns = 2` cặp (user + assistant) gần nhất được giữ **verbatim** để LLM thấy chính xác câu chữ của follow-up như *"loại xuất sắc thì sao?"*.
* **Rolling summary**: mỗi khi một cặp cũ bị đẩy ra khỏi recent window, LLM được gọi với prompt riêng (`_update_summary`) để gộp nội dung vào bản tóm tắt đang có. Tóm tắt này tích lũy thông tin cốt lõi: chủ đề đã hỏi, Điều luật được trích, con số cụ thể.
* **Auto-compress**: khi tóm tắt dài quá `max_summary_chars = 800`, LLM tự nén xuống dưới 400 chars.
* Prompt cuối nhận `[Tóm tắt các lượt trước]` + `[Các lượt gần đây]`, giữ token usage ≈ `O(recent_turns + summary_chars)` thay vì tăng tuyến tính theo độ dài hội thoại.

Lợi ích so với sliding window:
- Có thể nhớ thông tin từ 10+ lượt trước (không chỉ 6 lượt).
- Token usage ổn định khi hội thoại dài.
- LLM có "bộ nhớ dài" thực sự, không chỉ "bộ nhớ ngắn" của window cuối.

### 4.3 Anti-Hallucination

System prompt (`generation/prompts.py`) có quy tắc cứng:

> Nếu ngữ cảnh không chứa thông tin trả lời câu hỏi, hoặc không đủ chi tiết, bạn PHẢI trả lời đúng một câu: "Tôi không có đủ dữ liệu để trả lời câu hỏi này."

Cơ chế bảo vệ hai lớp:

1. Nếu retriever trả về 0 chunk → UI hiển thị ngay refusal, không gọi LLM.
2. Nếu retriever trả về chunk nhưng LLM vẫn trả `REFUSAL` → UI hiển thị nguyên văn.

Tham số sinh câu: `temperature=0.2, top_p=0.85, repetition_penalty=1.05, max_new_tokens=512` — thiên về ngắn gọn, ít sáng tạo, đúng văn phong quy chế.

### 4.4 Tối ưu tài nguyên GPU

- **bge-m3** (~2.3 GB): fp16 trên CUDA.
- **Qwen2.5-3B-Instruct** (~6 GB): 4-bit NF4 quantisation via `bitsandbytes.BitsAndBytesConfig`, `bnb_4bit_compute_dtype=float16`, double quantisation. Tổng footprint ~3.5–4 GB VRAM, thoải mái trên T4 16 GB.
- **FlashRank** (~120 MB): CPU-only, không tốn VRAM.
- Streamlit `@st.cache_resource` đảm bảo ba mô hình chỉ load **một lần** mỗi session.
- Embedding batch size mặc định 8; giảm xuống 4 nếu OOM.

### 4.5 UI — Minimal Light

* **Inter** font (Google Fonts), Vietnamese diacritics sắc nét.
* Bảng màu trắng + xám nhạt, **không gradient**, không logo.
* Một accent duy nhất: slate-600 (`#475569`) cho hover/active.
* Chat bubble viền mảnh 1px, không fill — focus vào nội dung.
* Sidebar typography uppercase + letter-spacing để phân cấp.
* Refusal bubble (yellow-100 + gold border) nổi bật mà không phá layout.

---

## 5. Đánh giá & kiểm thử nhanh

Sau khi chạy xong pipeline, kiểm tra nhanh trong Python:

```python
from retriever import HybridRetriever
from ingestion.embedder import load_embedder

r = HybridRetriever(embedder=load_embedder())
hits = r.search("Điều kiện xét học bổng", top_k=5)
for h in hits:
    print(h["score"], h["article_id"], h["source"], h["text"][:120])
```

Một số câu hỏi gợi ý để test end-to-end qua UI:

1. "Một học kỳ chính kéo dài bao nhiêu tuần?"
2. "Điều kiện xét tốt nghiệp đại học hệ chính quy?"
3. "Sinh viên bị kỷ luật cảnh cáo thì có bị xét học bổng không?"
4. "Mức học bổng loại xuất sắc là bao nhiêu?" (hỏi nối sau câu 3)
5. "Làm sao để đăng ký ký túc xá?"

---

## 6. Giới hạn & Mở rộng

- **Crawler không follow iframe / JS-rendered content**: trang Sổ tay là HTML tĩnh nên đủ, nhưng nếu sau này trang chuyển sang SPA thì cần đổi sang Playwright/Selenium.
- **BM25 tokenizer chưa tách từ ghép tiếng Việt** (vd. "đào tạo" thành 1 token). Có thể tích hợp `pyvi` hoặc `underthesea` để word-segment trước khi index nếu muốn tăng recall BM25.
- **Streaming output**: có thể chuyển `llm.chat()` sang `TextIteratorStreamer` để Streamlit hiển thị token theo thời gian thực.
- **Persistent chat history**: hiện lưu trong `st.session_state`. Để chạy đa phiên, có thể backend bằng SQLite hoặc Redis.
- **Swap Chroma → Qdrant** nếu cần horizontal scaling: chỉ cần sửa `ingestion/embedder.py` và phần Chroma client trong `retriever/hybrid_retriever.py`.