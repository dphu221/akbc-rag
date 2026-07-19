# Hướng dẫn chạy UET RAG Chatbot trên Kaggle

Tài liệu này hướng dẫn chi tiết từng bước để chạy dự án **UET RAG Chatbot** trên nền tảng Kaggle Notebook (hoàn toàn miễn phí, hỗ trợ GPU T4).

---

## 1. Chuẩn bị môi trường Kaggle

1. Truy cập [Kaggle](https://www.kaggle.com/) và đăng nhập tài khoản.
2. Tạo một Notebook mới bằng cách bấm vào **Create** -> **New Notebook**.
3. Cấu hình các thông số bên bảng điều khiển **Settings** ở cạnh phải màn hình (rất quan trọng):
   - **Accelerator**: Chọn **GPU T4 x2** hoặc **GPU T4** (Qwen 3B yêu cầu GPU CUDA để đạt tốc độ phản hồi tốt).
   - **Internet on**: Bật **ON** (gạt công tắc sang phải). Cài đặt này bắt buộc phải có để tải thư viện, nhân bản mã nguồn từ GitHub, tải mô hình từ HuggingFace và chạy Crawler.

---

## 2. Thiết lập dự án trên Notebook

Chạy các đoạn mã sau trong các ô (cells) của Kaggle Notebook:

### Bước 2.1 — Tải mã nguồn về Kaggle
Bạn có thể clone trực tiếp kho chứa từ GitHub:
```python
# Clone mã nguồn từ repo
!git clone https://github.com/dphu221/akbc-rag.git

# Di chuyển thư mục làm việc hiện tại vào thư mục dự án
%cd akbc-rag
```

### Bước 2.2 — Cài đặt các thư viện cần thiết
Kaggle đã cài sẵn PyTorch và một số thư viện cơ bản. Bạn cần cài các gói phụ thuộc đặc thù của dự án:
```python
!pip install -r requirements.txt
```

---

## 3. Chạy Pipeline dữ liệu (Crawl & Indexing)

Chạy các lệnh sau để thu thập dữ liệu và xây dựng cơ sở dữ liệu vector:

### Bước 3.1 — Cào dữ liệu từ Sổ tay UET
```python
!python -m crawler.crawl --out data/raw --max-pages 200 --max-files 80
```
*Lưu ý: Quá trình này sẽ tải các trang web và trích xuất nội dung từ các tệp PDF/DOCX có trong Sổ tay.*

### Bước 3.2 — Phân đoạn dữ liệu (Chunking) & Tạo chỉ mục Vector (Embedding)
Chạy script xây dựng index cho Chroma DB (quá trình này tự động tải mô hình embedding `BAAI/bge-m3`):
```python
!python -m ingestion.build_index \
    --manifest data/raw/manifest.jsonl \
    --db data/vector_db \
    --device auto \
    --batch-size 8
```

---

## 4. Chạy và Truy cập Streamlit Chatbot

Vì Kaggle chạy trên máy chủ ảo của Google và chặn các cổng kết nối thông thường (như `8501`), chúng ta cần chạy Streamlit trong nền và dùng các công cụ tạo đường hầm (tunnel) để truy cập giao diện web từ xa.

### Cách 1: Sử dụng Cloudflare Tunnel (Khuyên dùng - Ổn định nhất)
Cloudflare Tunnel không yêu cầu đăng ký tài khoản hay cấu hình mật khẩu IP.

1. **Cài đặt gói Cloudflare:**
   ```python
   !wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
   !dpkg -i cloudflared-linux-amd64.deb
   ```

2. **Khởi chạy Streamlit trong nền:**
   ```python
   import subprocess
   # Chạy Streamlit ở cổng 8501
   streamlit_proc = subprocess.Popen([
       "streamlit", "run", "app.py", 
       "--server.port", "8501", 
       "--server.address", "0.0.0.0"
   ])
   ```

3. **Mở đường hầm kết nối ngoại mạng:**
   ```python
   !cloudflared tunnel --url http://localhost:8501
   ```
   *Nhìn vào logs đầu ra, bạn sẽ thấy một đường dẫn dạng `https://xxx-xxx-xxx.trycloudflare.com`. Hãy click vào link đó để mở giao diện Chatbot.*

---

### Cách 2: Sử dụng Localtunnel
Localtunnel cũng là giải pháp thay thế hoàn toàn miễn phí.

1. **Cài đặt localtunnel:**
   ```python
   !npm install -g localtunnel
   ```

2. **Khởi chạy Streamlit trong nền:**
   ```python
   import subprocess
   streamlit_proc = subprocess.Popen([
       "streamlit", "run", "app.py", 
       "--server.port", "8501", 
       "--server.address", "0.0.0.0"
   ])
   ```

3. **Lấy địa chỉ IP Public của Kaggle (Dùng làm mật khẩu xác nhận của localtunnel):**
   ```python
   !curl ipv4.icanhazip.com
   ```

4. **Khởi tạo đường hầm:**
   ```python
   !lt --port 8501
   ```
   *Hãy nhấp vào link dạng `https://xxxx.loca.lt`, nhập địa chỉ IP vừa lấy được ở bước 3 vào ô **Tunnel Password** rồi nhấn **Submit** để vào Chatbot.*

---

## 5. Chạy thử nghiệm trực tiếp bằng Python (Không cần Streamlit)

Nếu bạn chỉ muốn kiểm tra nhanh kết quả mà không cần khởi động giao diện web, hãy chạy đoạn mã sau ngay trong ô Notebook:

```python
from retriever.hybrid_retriever import HybridRetriever
from ingestion.embedder import load_embedder
from generation.llm import load_generator
from generation.prompts import SYSTEM_PROMPT

# 1. Khởi tạo bộ truy xuất Hybrid (BM25 + Chroma Vector)
retriever = HybridRetriever(embedder=load_embedder())

# 2. Khởi tạo LLM Qwen (Tự động nạp cấu hình tối ưu 4-bit trên GPU)
generator = load_generator(device="auto")

# 3. Đặt câu hỏi và truy xuất ngữ cảnh liên quan
question = "Một học kỳ chính kéo dài bao nhiêu tuần?"
hits = retriever.search(question, top_k=5)
context = "\n\n".join([f"Nguồn: {h['source']} (Điều {h.get('article_id', 'N/A')})\nNội dung: {h['text']}" for h in hits])

# 4. Gửi câu hỏi và ngữ cảnh đến LLM để trả lời
user_prompt = f"Ngữ cảnh:\n{context}\n\nCâu hỏi: {question}"
answer = generator.chat(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)

print("\n=== CÂU HỎI MẪU ===")
print(question)
print("\n=== TRẢ LỜI TỪ RAG ===")
print(answer)
```

---

## 6. Các lỗi thường gặp & Cách khắc phục (Troubleshooting)

1. **Lỗi: `OutOfMemoryError` (OOM) khi Embed dữ liệu**
   - *Cách khắc phục:* Trong Bước 3.2, hãy giảm tham số `--batch-size` xuống còn `4` hoặc `2`.

2. **Mô hình sinh câu trả lời rất chậm**
   - *Cách khắc phục:* Kiểm tra xem bạn đã chọn Accelerator là **GPU T4** chưa. Nếu chưa chọn GPU, mô hình sẽ chạy trên CPU cực kỳ chậm. Bạn có thể kiểm tra bằng lệnh:
     ```python
     import torch
     print("CUDA Available:", torch.cuda.is_available())
     ```

3. **Lỗi không kết nối được Link Tunnel**
   - *Cách khắc phục:* Đôi khi dịch vụ Localtunnel hoặc Cloudflare Tunnel bị nghẽn. Bạn chỉ cần ngắt cell chạy lệnh tunnel (`Ctrl + C` hoặc nút Stop của Kaggle) và chạy lại.
