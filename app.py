"""UI chatbot Streamlit cho sổ tay RAG UET — phong cách sáng tối giản.

Chạy bằng:

    streamlit run app.py

Ngôn ngữ thiết kế
-----------------
* Bảng màu **sáng tối giản**: nền trắng tinh, viền xám nhẹ và một màu nhấn
  (xanh lam đá phiến dịu) cho các thao tác chính.
* Phông **Inter** (nạp từ Google Fonts) cho tiêu đề và nội dung. Dấu tiếng Việt
  được hiển thị sắc nét.
* Nhiều khoảng trắng, chiều cao dòng thoáng, không có gradient hay nền trang
  trí. Trọng tâm là chính nội dung hội thoại.
* Nguồn trích dẫn được hiển thị trong vùng mở rộng kín đáo, có chiều rộng khớp
  với bong bóng chat — xuất hiện khi cần và ẩn khi không cần.

Tính năng
---------
* Thanh bên: trạng thái mô hình, tham số truy hồi (top_k, trọng số, RRF k,
  tập tái xếp hạng), nút bật FlashRank, kích thước cửa sổ gần đây và nút
  "Xóa lịch sử chat".
* Nhiều lượt: ``SummaryMemory`` giữ bản tóm tắt các lượt cũ do LLM tạo và giữ
  nguyên văn N lượt gần nhất.
* Chống bịa thông tin: nếu bộ truy hồi trả về 0 đoạn HOẶC câu trả lời của LLM
  bằng ``REFUSAL``, UI hiển thị bong bóng "không có dữ liệu" riêng.
* Mỗi bong bóng trợ lý có bảng "Nguồn trích dẫn" có thể mở rộng.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retriever import HybridRetriever  # noqa: E402
from generation import (  # noqa: E402
    QwenGenerator,
    SummaryMemory,
    build_rag_prompt,
    SYSTEM_PROMPT,
    REFUSAL,
)
from ingestion.embedder import load_embedder  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag-app")


# --------------------------------------------------------------------------- #
# Các singleton được lưu đệm
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Đang nạp bộ truy hồi (Chroma + BM25) ...")
def get_retriever() -> HybridRetriever:
    embedder = load_embedder(device="auto")
    return HybridRetriever(embedder=embedder)


@st.cache_resource(show_spinner="Đang nạp Qwen2.5-3B-Instruct ...")
def get_llm() -> QwenGenerator:
    return QwenGenerator(device="auto", load_in_4bit=True)


# --------------------------------------------------------------------------- #
# Tiện ích UI
# --------------------------------------------------------------------------- #
def _render_sources(chunks: List[Dict[str, Any]]) -> None:
    if not chunks:
        st.markdown("*Không có nguồn trích dẫn.*")
        return
    with st.expander(f"Nguồn trích dẫn · {len(chunks)} đoạn", expanded=False):
        for i, c in enumerate(chunks, start=1):
            source = c.get("source", "?")
            article = c.get("article_id", "")
            chapter = c.get("chapter", "")
            url = c.get("source_url", "")
            score = c.get("score", 0.0)
            rrf = c.get("rrf_score")
            rerank_score = c.get("rerank_score")
            bm25_rank = c.get("bm25_rank")
            dense_rank = c.get("dense_rank")
            snippet = c.get("text", "")[:500].replace("\n", " ")

            header = f"**[{i}] {source}"
            if article:
                header += f" — {article}"
            if chapter:
                header += f" · {chapter}"
            header += "**"
            st.markdown(header)

            meta_bits = []
            if rerank_score is not None:
                meta_bits.append(f"rerank: `{rerank_score:.4f}`")
            if rrf is not None:
                meta_bits.append(f"RRF: `{rrf:.4f}`")
            if bm25_rank is not None:
                meta_bits.append(f"BM25 rank: {bm25_rank}")
            if dense_rank is not None:
                meta_bits.append(f"Dense rank: {dense_rank}")
            if meta_bits:
                st.caption(" · ".join(meta_bits))
            if url:
                st.caption(url)
            st.markdown(f"> {snippet}{'…' if len(c.get('text', '')) > 500 else ''}")
            if i < len(chunks):
                st.divider()


def _ensure_state() -> None:
    if "memory" not in st.session_state:
        st.session_state.memory = None  # nạp lười: tạo sau khi LLM được nạp
    if "display_turns" not in st.session_state:
        # display_turns: danh sách {role, content, chunks} dùng để hiển thị
        st.session_state.display_turns = []


def _get_memory(llm: QwenGenerator, recent_turns: int) -> SummaryMemory:
    if st.session_state.memory is None:
        st.session_state.memory = SummaryMemory(
            llm_chat=llm.chat,
            recent_turns=recent_turns,
        )
    else:
        st.session_state.memory.recent_turns = recent_turns
    return st.session_state.memory


# --------------------------------------------------------------------------- #
# Hàm chính
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(
        page_title="UET RAG Chatbot — Sổ tay Sinh viên",
        page_icon="🎓",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # ---- CSS: Thiết kế thích ứng ----------------------------------------
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* Hệ thống thiết kế thích ứng với chủ đề sáng và tối của Streamlit */
        :root {
            --user-bubble-bg: rgba(37, 99, 235, 0.06);
            --user-bubble-border: rgba(37, 99, 235, 0.22);
            --asst-bubble-bg: rgba(148, 163, 184, 0.08);
            --asst-bubble-border: rgba(148, 163, 184, 0.2);
            --text-muted: #64748b;
            --refusal-bg: #fef9c3;
            --refusal-text: #713f12;
            --refusal-border: #eab308;
            --card-border: rgba(148, 163, 184, 0.2);
        }

        @media (prefers-color-scheme: dark) {
            :root {
                --user-bubble-bg: rgba(59, 130, 246, 0.18);
                --user-bubble-border: rgba(59, 130, 246, 0.4);
                --asst-bubble-bg: rgba(30, 41, 59, 0.7);
                --asst-bubble-border: rgba(51, 65, 85, 0.8);
                --text-muted: #94a3b8;
                --refusal-bg: #422006;
                --refusal-text: #fef08a;
                --refusal-border: #ca8a04;
                --card-border: rgba(255, 255, 255, 0.12);
            }
        }

        [data-theme="dark"] {
            --user-bubble-bg: rgba(59, 130, 246, 0.18);
            --user-bubble-border: rgba(59, 130, 246, 0.4);
            --asst-bubble-bg: rgba(30, 41, 59, 0.7);
            --asst-bubble-border: rgba(51, 65, 85, 0.8);
            --text-muted: #94a3b8;
            --refusal-bg: #422006;
            --refusal-text: #fef08a;
            --refusal-border: #ca8a04;
            --card-border: rgba(255, 255, 255, 0.12);
        }

        html, body, .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }

        /* Giới hạn khung chứa */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 4rem;
            max-width: 780px;
        }

        /* Đường phân cách dưới tiêu đề */
        .title-rule {
            border: 0;
            border-top: 1px solid var(--card-border);
            margin: 0.8rem 0 1.5rem 0;
        }

        /* Bong bóng chat */
        .stChatMessage {
            border-radius: 10px !important;
            padding: 1rem 1.25rem !important;
            margin-bottom: 1rem !important;
            transition: background-color 0.2s ease, border-color 0.2s ease !important;
        }

        /* Bảo đảm nội dung tin nhắn kế thừa màu chữ gốc của Streamlit */
        .stChatMessage p, 
        .stChatMessage span, 
        .stChatMessage div, 
        .stChatMessage markdown {
            color: inherit !important;
        }

        /* Bong bóng tin nhắn người dùng */
        div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
        .stChatMessage:has([data-testid="user"]) {
            background-color: var(--user-bubble-bg) !important;
            border: 1px solid var(--user-bubble-border) !important;
        }

        /* Bong bóng tin nhắn trợ lý */
        div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]),
        .stChatMessage:has([data-testid="assistant"]) {
            background-color: var(--asst-bubble-bg) !important;
            border: 1px solid var(--asst-bubble-border) !important;
        }

        /* Kiểu dáng ảnh đại diện */
        .stChatMessageAvatar {
            border-radius: 6px !important;
        }

        /* Tiêu đề và văn bản thanh bên */
        section[data-testid="stSidebar"] .stMarkdown h1,
        section[data-testid="stSidebar"] .stMarkdown h2,
        section[data-testid="stSidebar"] .stMarkdown h3 {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--text-muted);
        }

        /* Các nút */
        .stButton > button {
            border: 1px solid var(--card-border);
            border-radius: 6px;
            font-weight: 500;
            transition: all 0.15s ease;
        }

        /* Bong bóng từ chối */
        .refusal {
            background: var(--refusal-bg);
            border: 1px solid var(--refusal-border);
            border-left: 4px solid var(--refusal-border);
            padding: 0.8rem 1.1rem;
            border-radius: 8px;
            color: var(--refusal-text) !important;
            font-size: 0.95rem;
            font-weight: 500;
        }
        .refusal * {
            color: var(--refusal-text) !important;
        }

        /* Chú thích và siêu dữ liệu */
        .stCaption, [data-testid="stCaptionContainer"] {
            color: var(--text-muted) !important;
            font-size: 0.8rem !important;
        }

        /* Vùng mở rộng */
        .stExpander {
            border: 1px solid var(--card-border) !important;
            border-radius: 8px !important;
            background-color: transparent !important;
        }

        /* Ẩn chân trang */
        footer { visibility: hidden; }

        /* Ô nhập hội thoại */
        .stChatInput textarea {
            font-family: 'Inter', sans-serif !important;
            border-radius: 8px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    _ensure_state()

    # ---- Đầu trang -----------------------------------------------------
    st.markdown(
        "<h3 style='margin-bottom:0.2rem'>UET RAG Chatbot</h3>"
        "<p style='color:var(--text-muted); margin-top:0; font-size:0.9rem'>"
        "Sổ tay Sinh viên · Trường Đại học Công nghệ (UET) — ĐHQG Hà Nội"
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='title-rule' />", unsafe_allow_html=True)

    # ---- Thanh bên -----------------------------------------------------
    with st.sidebar:
        st.markdown("### Cấu hình")
        top_k = st.slider("Số chunk trả về (top_k)", 1, 20, 5)
        k_per_retriever = st.slider("Số chunk mỗi retriever", 5, 50, 20)
        rerank_pool = st.slider("Kích thước pool rerank", 5, 40, 20)
        bm25_w = st.slider("Trọng số BM25", 0.0, 2.0, 1.0, 0.1)
        dense_w = st.slider("Trọng số Dense (bge-m3)", 0.0, 2.0, 1.0, 0.1)
        rrf_k = st.number_input("Hằng số RRF k", min_value=1, max_value=200, value=60)
        use_flashrank = st.checkbox("Bật FlashRank rerank", value=True)
        recent_turns = st.slider("Số lượt gần đây giữ nguyên", 1, 6, 2)

        st.markdown("---")
        if st.button("Xoá lịch sử chat", use_container_width=True):
            st.session_state.memory = None
            st.session_state.display_turns = []
            st.rerun()

        st.markdown("---")
        st.caption("Mô hình cố định")
        st.caption("• Embedding: `BAAI/bge-m3`")
        st.caption("• LLM: `Qwen/Qwen2.5-3B-Instruct`")
        st.caption("• Vector DB: Chroma (cosine)")
        st.caption("• Hybrid: BM25 + Dense + RRF + FlashRank")
        st.caption("• Memory: Summary + recent window")

    # ---- Nạp các mô hình (đã lưu đệm) ---------------------------------
    try:
        retriever = get_retriever()
    except Exception as exc:
        st.error(
            "Không thể nạp bộ truy hồi. Hãy chạy `python -m ingestion.build_index` trước. "
            f"Chi tiết: {exc}"
        )
        st.stop()
    try:
        llm = get_llm()
    except Exception as exc:
        st.error(f"Không thể nạp LLM Qwen2.5-3B-Instruct. Chi tiết: {exc}")
        st.stop()

    retriever.rrf_k = int(rrf_k)
    retriever.use_flashrank = bool(use_flashrank)
    if not use_flashrank:
        retriever._flashrank = None

    memory = _get_memory(llm, recent_turns)

    # ---- Tùy chọn: hiển thị bản tóm tắt đang tích lũy -----------------
    if memory.get_summary_text():
        with st.expander("Tóm tắt hội thoại (đang lưu)", expanded=False):
            st.caption("Được tạo và cập nhật tự động bởi LLM; dùng để giữ ngữ cảnh dài.")
            st.markdown(memory.get_summary_text())

    # ---- Hiển thị các lượt hội thoại ----------------------------------
    for turn in st.session_state.display_turns:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn["role"] == "assistant" and turn.get("chunks"):
                _render_sources(turn["chunks"])

    # ---- Đầu vào -------------------------------------------------------
    user_input = st.chat_input("Nhập câu hỏi về Sổ tay Sinh viên UET ...")
    if not user_input:
        return

    # Hiển thị bong bóng người dùng ngay lập tức.
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.display_turns.append(
        {"role": "user", "content": user_input, "chunks": None}
    )
    memory.add_user_turn(user_input)

    # ---- Truy hồi ------------------------------------------------------
    with st.chat_message("assistant"):
        with st.spinner("Đang truy hồi ngữ cảnh ..."):
            chunks = retriever.search(
                user_input,
                top_k=top_k,
                k_per_retriever=k_per_retriever,
                rerank_pool=rerank_pool,
                bm25_weight=bm25_w,
                dense_weight=dense_w,
            )
        if not chunks:
            st.markdown(
                f'<div class="refusal">{REFUSAL}</div>',
                unsafe_allow_html=True,
            )
            st.session_state.display_turns.append(
                {"role": "assistant", "content": REFUSAL, "chunks": []}
            )
            memory.add_assistant_turn(REFUSAL)
            return

        # Tạo prompt và gọi LLM.
        prompt = build_rag_prompt(
            question=user_input,
            retrieved_chunks=chunks,
            history=memory,
        )
        with st.spinner("Đang sinh câu trả lời ..."):
            try:
                answer = llm.chat(SYSTEM_PROMPT, prompt)
            except Exception as exc:
                answer = f"Đã xảy ra lỗi khi gọi LLM: {exc}"
                logger.exception("LLM call failed")

        st.markdown(answer)
        _render_sources(chunks)

        st.session_state.display_turns.append(
            {"role": "assistant", "content": answer, "chunks": chunks}
        )
        memory.add_assistant_turn(answer)


if __name__ == "__main__":
    main()
