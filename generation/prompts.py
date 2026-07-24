"""Prompt templates for the UET handbook RAG chatbot.

All prompts are written in Vietnamese because the source documents and user
queries are Vietnamese. The system prompt explicitly forbids hallucination
and instructs the model to refuse gracefully when retrieval is empty or
insufficient.
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

from .memory import SummaryMemory, build_history_from_memory

SYSTEM_PROMPT = """Bạn là trợ lý ảo của Sổ tay Sinh viên Trường Đại học Công nghệ (UET) - ĐHQG Hà Nội.
Nhiệm vụ của bạn là trả lời câu hỏi của sinh viên một cách chính xác, ngắn gọn dựa trên các quy định được trích xuất trong phần Ngữ cảnh.

NGUYÊN TẮC BẮT BUỘC:
1. Trả lời dựa TRỰC TIẾP vào thông tin trong phần "Ngữ cảnh". Không suy đoán hoặc bổ sung thông tin không có trong tài liệu.
2. Nếu trong phần Ngữ cảnh KHÔNG CÓ thông tin để trả lời câu hỏi, hãy trả lời duy nhất: "Tôi không có đủ dữ liệu để trả lời câu hỏi này."
3. Trích dẫn nguồn ở cuối câu trả lời theo dạng: [Nguồn: <tên tệp> - <Điều X>].
4. Giữ câu trả lời đúng trọng tâm, ngắn gọn và giữ nguyên các thuật ngữ chuyên ngành.
"""

REFUSAL = "Tôi không có đủ dữ liệu để trả lời câu hỏi này."

# RAG prompt template (context + history + question).
RAG_TEMPLATE = """[Ngữ cảnh từ Sổ tay Sinh viên UET]
{context}

---
[Lịch sử hội thoại]
{history}

---
[Câu hỏi của sinh viên]
{question}

Hãy dựa vào Ngữ cảnh ở trên để trả lời câu hỏi của sinh viên một cách ngắn gọn, đúng trọng tâm và kèm trích dẫn nguồn thích hợp.
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
        header = f"Đoạn {i} (Nguồn: {citation}"
        if chapter:
            header += f" | {chapter}"
        header += "):"
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