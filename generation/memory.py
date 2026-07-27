"""Conversation memory with progressive summarisation.

Strategy
--------
* The last ``recent_turns`` turns (default 2 user + 2 assistant = 4 messages)
  are kept **verbatim** so the LLM sees exact wording for follow-up questions
  like "vậy loại xuất sắc thì sao?".
* Anything older is **condensed into a running summary** by the LLM itself.
  The summary is updated incrementally: each time a new turn exits the
  "recent window", the LLM is asked to fold it into the existing summary.
* The final prompt receives::

      [Tóm tắt hội thoại trước đó]
      <summary text or "(không có)">

      [Các lượt gần đây]
      Sinh viên: ...
      Trợ lý: ...
      Sinh viên: <current question>

This keeps prompt length roughly ``O(recent_turns + summary_tokens)`` instead
of growing linearly with conversation length.

Thread-safety
-------------
The class is not thread-safe; it is intended to be used from a single
Streamlit session (one per browser tab).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclass
class SummaryMemory:
    """Pluggable summary-memory manager.

    Parameters
    ----------
    llm_chat
        Callable ``(system_prompt: str, user_prompt: str) -> str`` used to
        produce the rolling summary.  Typically ``QwenGenerator.chat``.
    recent_turns
        Number of most-recent *message pairs* (user + assistant) to keep
        verbatim.  Default 2 → up to 4 raw messages in the prompt.
    max_summary_tokens
        Soft cap for the summary text (in characters, not tokens — close
        enough for Vietnamese where 1 token ≈ 2-4 chars).  When exceeded we
        ask the LLM to compress the summary further.
    """

    llm_chat: Callable[[str, str], str]
    recent_turns: int = 2
    max_summary_chars: int = 800

    summary: str = ""
    recent: List[Dict[str, str]] = field(default_factory=list)
    # Full raw history is kept for display in the UI, but only the summary +
    # recent window go into the prompt.
    full_history: List[Dict[str, str]] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def add_user_turn(self, content: str) -> None:
        """Record a user message.  Does NOT trigger summarisation."""
        self.full_history.append({"role": "user", "content": content})
        self.recent.append({"role": "user", "content": content})

    def add_assistant_turn(self, content: str) -> None:
        """Record an assistant reply and roll the window if needed."""
        self.full_history.append({"role": "assistant", "content": content})
        self.recent.append({"role": "assistant", "content": content})
        self._maybe_roll()

    def get_prompt_history(self) -> List[Dict[str, str]]:
        """Return the messages that should go into the next LLM prompt.

        The first message is a synthetic ``system``-style note carrying the
        summary; the rest are the verbatim recent turns.  Callers should
        prepend their own system prompt separately.
        """
        out: List[Dict[str, str]] = []
        if self.summary:
            out.append(
                {
                    "role": "system",
                    "content": f"Tóm tắt hội thoại trước đó:\n{self.summary}",
                }
            )
        out.extend(self.recent)
        return out

    def get_summary_text(self) -> str:
        """Return the running summary text (possibly empty)."""
        return self.summary

    def get_recent_text(self) -> str:
        """Render the recent window as plain text (for the RAG prompt template)."""
        if not self.recent:
            return "(chưa có lịch sử)"
        lines: List[str] = []
        for turn in self.recent:
            role = turn.get("role", "user")
            content = (turn.get("content") or "").strip()
            if not content:
                continue
            if role == "user":
                lines.append(f"Sinh viên: {content}")
            elif role == "assistant":
                lines.append(f"Trợ lý: {content}")
        return "\n".join(lines) if lines else "(chưa có lịch sử)"

    def clear(self) -> None:
        """Reset all state (used when the user clicks "Clear chat")."""
        self.summary = ""
        self.recent = []
        self.full_history = []

    # ------------------------------------------------------------------ #
    # Internal: rolling summarisation
    # ------------------------------------------------------------------ #
    def _maybe_roll(self) -> None:
        """If the recent window exceeds ``recent_turns`` pairs, fold the
        oldest pair into the summary."""
        # ``recent_turns`` is measured in *pairs* (user+assistant).
        max_messages = self.recent_turns * 2
        while len(self.recent) > max_messages:
            # Pop the oldest pair (user, assistant) — but be defensive: the
            # window might not be perfectly paired (e.g. user asked, no
            # assistant reply yet).
            oldest_user = self.recent.pop(0)
            oldest_assistant: Optional[Dict[str, str]] = None
            if self.recent and self.recent[0]["role"] == "assistant":
                oldest_assistant = self.recent.pop(0)
            self._update_summary(oldest_user, oldest_assistant)

        # If the summary has grown too large, compress it.
        if len(self.summary) > self.max_summary_chars:
            self._compress_summary()

    def _update_summary(
        self,
        user_turn: Dict[str, str],
        assistant_turn: Optional[Dict[str, str]],
    ) -> None:
        prev_summary = self.summary.strip()
        u = user_turn["content"].strip()
        a = (assistant_turn["content"].strip() if assistant_turn else "(không có phản hồi)")
        sys_prompt = (
            "Bạn là trình tóm tắt hội thoại cho chatbot Sổ tay Sinh viên UET. "
            "Nhiệm vụ: cập nhật bản tóm tắt hội thoại dựa trên tóm tắt cũ và một lượt trò chuyện mới. "
            "Quy tắc:\n"
            "1. Chỉ giữ lại thông tin CỐT LÕI: chủ đề sinh viên hỏi, quyết định/trả lời chính của trợ lý, "
            "các thực thể (Điều X, học bổng, học phí, ký túc xá, ...) đã được đề cập.\n"
            "2. Viết bằng tiếng Việt, ngắn gọn, dưới 150 từ.\n"
            "3. Nếu thông tin mới trùng lặp với tóm tắt cũ, không lặp lại.\n"
            "4. KHÔNG bịa ra thông tin; nếu không rõ, bỏ qua.\n"
            "5. Trả về duy nhất bản tóm tắt mới, không kèm giải thích."
        )
        user_prompt = (
            f"Tóm tắt hiện tại:\n{prev_summary or '(không có)'}\n\n"
            f"Lượt mới:\nSinh viên: {u}\nTrợ lý: {a}\n\n"
            f"Hãy trả về bản tóm tắt cập nhật."
        )
        try:
            new_summary = self.llm_chat(sys_prompt, user_prompt)
            self.summary = new_summary.strip()
        except Exception as exc:
            logger.warning("Summary update failed (%s); keeping previous summary.", exc)
            # Fallback: append a terse manual note.
            self.summary = (prev_summary + f"\n- SV hỏi: {u[:120]}").strip()

    def _compress_summary(self) -> None:
        sys_prompt = (
            "Bạn là trình nén tóm tắt.  Bản tóm tắt dưới đây đang quá dài.  Hãy nén nó "
            "xuống dưới 400 ký tự, giữ lại các thực thể quan trọng (Điều X, tên quy chế, "
            "số tiền, ngày tháng).  Trả về duy nhất bản tóm tắt đã nén."
        )
        try:
            self.summary = self.llm_chat(sys_prompt, self.summary).strip()
        except Exception as exc:
            logger.warning("Summary compression failed (%s).", exc)


# --------------------------------------------------------------------------- #
# Prompt helper that replaces the old sliding-window builder
# --------------------------------------------------------------------------- #
def build_history_from_memory(memory: SummaryMemory) -> str:
    """Render the memory's summary + recent window as a single text block.

    This is consumed by ``generation.prompts.build_rag_prompt``.
    """
    parts: List[str] = []
    summary = memory.get_summary_text().strip()
    if summary:
        parts.append(f"[Tóm tắt các lượt trước]\n{summary}")
    recent = memory.get_recent_text()
    if recent and recent != "(chưa có lịch sử)":
        parts.append(f"[Các lượt gần đây]\n{recent}")
    return "\n\n".join(parts) if parts else "(chưa có lịch sử)"