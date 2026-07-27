"""Bộ nhớ hội thoại với khả năng tóm tắt tăng dần.

Chiến lược
----------
* Các lượt ``recent_turns`` gần nhất (mặc định 2 người dùng + 2 trợ lý = 4
  tin nhắn) được giữ **nguyên văn** để LLM thấy chính xác câu chữ của câu hỏi
  nối tiếp như "vậy loại xuất sắc thì sao?".
* Nội dung cũ hơn được chính LLM **cô đọng vào bản tóm tắt đang tích lũy**.
  Bản tóm tắt được cập nhật dần: mỗi khi một lượt rời khỏi "cửa sổ gần đây",
  LLM được yêu cầu gộp lượt đó vào bản tóm tắt hiện có.
* Prompt cuối cùng nhận::

      [Tóm tắt hội thoại trước đó]
      <văn bản tóm tắt hoặc "(không có)">

      [Các lượt gần đây]
      Sinh viên: ...
      Trợ lý: ...
      Sinh viên: <current question>

Nhờ đó độ dài prompt xấp xỉ ``O(recent_turns + summary_tokens)`` thay vì tăng
tuyến tính theo độ dài hội thoại.

An toàn luồng
------------
Lớp này không an toàn luồng; nó được thiết kế để dùng trong một phiên
Streamlit duy nhất (mỗi tab trình duyệt một phiên).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Các lớp dữ liệu
# --------------------------------------------------------------------------- #
@dataclass
class SummaryMemory:
    """Trình quản lý bộ nhớ tóm tắt có thể thay thế.

    Tham số
    -------
    llm_chat
        Hàm gọi được ``(system_prompt: str, user_prompt: str) -> str`` dùng để
        tạo bản tóm tắt tích lũy. Thường là ``QwenGenerator.chat``.
    recent_turns
        Số *cặp tin nhắn* gần nhất (người dùng + trợ lý) cần giữ nguyên văn.
        Mặc định 2 → tối đa 4 tin nhắn thô trong prompt.
    max_summary_tokens
        Giới hạn mềm cho văn bản tóm tắt (tính bằng ký tự, không phải token —
        đủ gần với tiếng Việt, nơi 1 token ≈ 2-4 ký tự). Khi vượt giới hạn,
        chương trình yêu cầu LLM nén bản tóm tắt thêm.
    """

    llm_chat: Callable[[str, str], str]
    recent_turns: int = 2
    max_summary_chars: int = 800

    summary: str = ""
    recent: List[Dict[str, str]] = field(default_factory=list)
    # Lịch sử thô đầy đủ được giữ để hiển thị trên UI, nhưng chỉ bản tóm tắt
    # và cửa sổ gần đây được đưa vào prompt.
    full_history: List[Dict[str, str]] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # API công khai
    # ------------------------------------------------------------------ #
    def add_user_turn(self, content: str) -> None:
        """Ghi một tin nhắn người dùng. KHÔNG kích hoạt tóm tắt."""
        self.full_history.append({"role": "user", "content": content})
        self.recent.append({"role": "user", "content": content})

    def add_assistant_turn(self, content: str) -> None:
        """Ghi câu trả lời của trợ lý và cuộn cửa sổ nếu cần."""
        self.full_history.append({"role": "assistant", "content": content})
        self.recent.append({"role": "assistant", "content": content})
        self._maybe_roll()

    def get_prompt_history(self) -> List[Dict[str, str]]:
        """Trả về các tin nhắn cần đưa vào prompt LLM tiếp theo.

        Tin nhắn đầu tiên là ghi chú tổng hợp kiểu ``system`` chứa bản tóm tắt;
        phần còn lại là các lượt gần đây còn nguyên văn. Bên gọi nên tự thêm
        prompt hệ thống của mình vào trước.
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
        """Trả về văn bản tóm tắt đang tích lũy (có thể trống)."""
        return self.summary

    def get_recent_text(self) -> str:
        """Hiển thị cửa sổ gần đây dưới dạng văn bản thuần (cho mẫu prompt RAG)."""
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
        """Đặt lại toàn bộ trạng thái (dùng khi người dùng bấm "Xóa lịch sử chat")."""
        self.summary = ""
        self.recent = []
        self.full_history = []

    # ------------------------------------------------------------------ #
    # Nội bộ: tóm tắt tích lũy
    # ------------------------------------------------------------------ #
    def _maybe_roll(self) -> None:
        """Nếu cửa sổ gần đây vượt quá ``recent_turns`` cặp, gộp cặp cũ nhất
        vào bản tóm tắt."""
        # ``recent_turns`` được tính theo *cặp* (người dùng + trợ lý).
        max_messages = self.recent_turns * 2
        while len(self.recent) > max_messages:
            # Lấy cặp cũ nhất (người dùng, trợ lý) ra — nhưng vẫn phòng vệ vì
            # cửa sổ có thể chưa ghép cặp hoàn chỉnh (ví dụ người dùng đã hỏi
            # nhưng trợ lý chưa trả lời).
            oldest_user = self.recent.pop(0)
            oldest_assistant: Optional[Dict[str, str]] = None
            if self.recent and self.recent[0]["role"] == "assistant":
                oldest_assistant = self.recent.pop(0)
            self._update_summary(oldest_user, oldest_assistant)

        # Nếu bản tóm tắt quá dài, hãy nén lại.
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
            # Phương án dự phòng: thêm một ghi chú thủ công ngắn gọn.
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
# Tiện ích prompt thay thế trình tạo cửa sổ trượt cũ
# --------------------------------------------------------------------------- #
def build_history_from_memory(memory: SummaryMemory) -> str:
    """Hiển thị bản tóm tắt và cửa sổ gần đây thành một khối văn bản.

    Nội dung này được ``generation.prompts.build_rag_prompt`` sử dụng.
    """
    parts: List[str] = []
    summary = memory.get_summary_text().strip()
    if summary:
        parts.append(f"[Tóm tắt các lượt trước]\n{summary}")
    recent = memory.get_recent_text()
    if recent and recent != "(chưa có lịch sử)":
        parts.append(f"[Các lượt gần đây]\n{recent}")
    return "\n\n".join(parts) if parts else "(chưa có lịch sử)"
