"""Prompt templates for the UET handbook RAG chatbot.

All prompts are written in Vietnamese because the source documents and user
queries are Vietnamese.  The system prompt explicitly forbids hallucination
and instructs the model to refuse gracefully when retrieval is empty or
insufficient.
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

from .memory import SummaryMemory, build_history_from_memory

SYSTEM_PROMPT = """Bạn là trợ lý ảo của Sổ tay Sinh viên Trường Đại học Công nghệ (UET) - ĐHQG Hà Nội.
Nhiệm vụ của bạn là trả lời câu hỏi của sinh viên dựa trên các quy định, quy chế,
hướng dẫn chính thức được trích xuất từ Sổ tay Sinh viên UET.

NGUYÊN TẮC BẮT BUỘC:
1. CHỈ sử dụng thông tin trong phần "Ngữ cảnh" được cung cấp. KHÔNG được tự bịa
   ra quy định, con số, hoặc quy chế không có trong ngữ cảnh.
2. Nếu ngữ cảnh không chứa thông tin trả lời câu hỏi, hoặc không đủ chi tiết, bạn
   PHẢI trả lời đúng một câu: "Tôi không có đủ dữ liệu để trả lời câu hỏi này."
   và không giải thích thêm.
3. Khi trả lời, hãy trích dẫn rõ nguồn theo định dạng: [Nguồn: <tên file> - <Điều X>]
   ở cuối mỗi ý có sử dụng thông tin. Nếu ý dựa trên nhiều nguồn, liệt kê tất cả.
4. Trả lời ngắn gọn, rõ ràng, đúng trọng tâm. Ưu tiên gạch đầu dòng khi liệt kê
   nhiều điều kiện hoặc bước thủ tục.
5. Giữ nguyên thuật ngữ chuyên ngành (ví dụ: "tín chỉ", "GPA", "rèn luyện",
   "đào tạo chính quy", "chính sách sinh viên").
6. Nếu câu hỏi của người dùng tham chiếu đến một câu hỏi trước đó (ví dụ: "còn
   mức kia thì sao?", "loại xuất sắc thì sao?"), hãy dùng phần "Tóm tắt hội thoại
   trước đó" và "Các lượt gần đây" để giải tham chiếu đó trước khi trả lời.
7. KHÔNG đưa ra ý kiến cá nhân, lời khuyên, hoặc bình luận ngoài phạm vi Sổ tay.
"""

REFUSAL = "Tôi không có đủ dữ liệu để trả lời câu hỏi này."

# RAG prompt template (context + history + question).
RAG_TEMPLATE = """Ngữ cảnh (các đoạn trích từ Sổ tay Sinh viên UET):
{context}

---
Lịch sử hội thoại:
{history}

Câu hỏi của sinh viên:
{question}

Hãy trả lời câu hỏi dựa trên Ngữ cảnh ở trên. Nhớ tuân thủ nghiêm ngặt các nguyên tắc trong system prompt, đặc biệt là quy tắc chống ảo tưởng (mục 2) và quy tắc trích dẫn nguồn (mục 3).
"""


def _format_citation(source: str, article_id: str) -> str:
    """Build a citation string such as ``quy_che_dt_k67 - Điều 16``."""
    parts = []
    if source:
        parts.append(source)
    if article_id:
        parts.append(article_id)
    return " - ".join(parts) if parts else "không rõ"


def build_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Render the retrieved chunks as a numbered context block."""
    if not retrieved_chunks:
        return "(Không có đoạn ngữ cảnh nào được truy hồi.)"
    lines: List[str] = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        citation = _format_citation(
            chunk.get("source", ""),
            chunk.get("article_id", ""),
        )
        chapter = chunk.get("chapter", "")
        header = f"[{i}] Nguồn: {citation}"
        if chapter:
            header += f" | {chapter}"
        body = chunk.get("text", "").strip()
        lines.append(f"{header}\n{body}")
    return "\n\n".join(lines)


def build_history(
    history: Union[List[Dict[str, str]], SummaryMemory, None],
    *,
    max_turns: int = 6,
) -> str:
    """Render conversation history for the prompt.

    Accepts either:
      * a ``SummaryMemory`` instance (preferred — uses summary + recent window),
      * a list of ``{"role", "content"}`` dicts (legacy sliding window),
      * ``None``.
    """
    if isinstance(history, SummaryMemory):
        return build_history_from_memory(history)
    if not history:
        return "(chưa có lịch sử)"
    trimmed = history[-max_turns:]
    lines: List[str] = []
    for turn in trimmed:
        role = turn.get("role", "user")
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            lines.append(f"Sinh viên: {content}")
        else:
            lines.append(f"Trợ lý: {content}")
    return "\n".join(lines) if lines else "(chưa có lịch sử)"


def build_rag_prompt(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    history: Union[List[Dict[str, str]], SummaryMemory, None] = None,
    *,
    max_turns: int = 6,
) -> str:
    """Render the full user-side RAG prompt."""
    return RAG_TEMPLATE.format(
        context=build_context(retrieved_chunks),
        history=build_history(history, max_turns=max_turns),
        question=question.strip(),
    )