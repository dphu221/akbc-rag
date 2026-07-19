"""Smoke test: SummaryMemory rolling summarisation with a stub LLM.

We simulate a multi-turn conversation, then verify:
  - recent window is truncated to ``recent_turns`` pairs
  - summary text grows as old turns roll out of the window
  - get_prompt_history() returns [summary system msg, recent turns...]
  - prompt text contains the summary block + recent block
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
logger = logging.getLogger("memory-test")

ROOT = Path(__file__).resolve().parent.parent / "rag-pipeline"
sys.path.insert(0, str(ROOT))

from generation.memory import SummaryMemory


# ---- Stub LLM: just echoes a condensed version ----------------------- #
def stub_llm_chat(system_prompt: str, user_prompt: str) -> str:
    """Pretend to summarise: return a short tag indicating what was folded."""
    # Extract the "Lượt mới" block to know what we're folding.
    if "Tóm tắt hiện tại" in user_prompt and "Lượt mới" in user_prompt:
        # Summarisation call
        # Find the new turn content
        lines = user_prompt.split("\n")
        user_line = next((ln for ln in lines if ln.startswith("Sinh viên:")), "")
        asst_line = next((ln for ln in lines if ln.startswith("Trợ lý:")), "")
        # Build a short summary
        prev = ""
        if "Tóm tắt hiện tại:" in user_prompt:
            idx = user_prompt.index("Tóm tắt hiện tại:")
            end = user_prompt.index("\n\n", idx)
            prev = user_prompt[idx + len("Tóm tắt hiện tại:"):end].strip()
        new_bits = [prev] if prev and prev != "(không có)" else []
        if user_line:
            new_bits.append(f"SV hỏi về: {user_line.replace('Sinh viên:', '').strip()[:80]}")
        if asst_line:
            new_bits.append(f"Trợ lý trả lời về: {asst_line.replace('Trợ lý:', '').strip()[:80]}")
        return " | ".join(new_bits)
    if "nén" in system_prompt.lower():
        # Compression call
        return user_prompt[:400] + " [compressed]"
    return "stub-response"


# ---- Run test -------------------------------------------------------- #
memory = SummaryMemory(llm_chat=stub_llm_chat, recent_turns=2)

logger.info("Turn 1: user")
memory.add_user_turn("Điều kiện nhận học bổng là gì?")
logger.info("  recent size: %d, summary: %r", len(memory.recent), memory.get_summary_text())

logger.info("Turn 1: assistant")
memory.add_assistant_turn("Theo Điều 4 quy định khen thưởng, sinh viên Xuất sắc được nhận học bổng.")
logger.info("  recent size: %d, summary: %r", len(memory.recent), memory.get_summary_text())

logger.info("Turn 2: user")
memory.add_user_turn("Mức học bổng loại xuất sắc là bao nhiêu?")
logger.info("  recent size: %d, summary: %r", len(memory.recent), memory.get_summary_text())

logger.info("Turn 2: assistant")
memory.add_assistant_turn("Mức học bổng loại xuất sắc là 5.000.000 VNĐ.")
logger.info("  recent size: %d, summary: %r", len(memory.recent), memory.get_summary_text())

logger.info("Turn 3: user (this should roll turn 1 into summary)")
memory.add_user_turn("Còn loại giỏi thì sao?")
logger.info("  recent size: %d, summary: %r", len(memory.recent), memory.get_summary_text())

logger.info("Turn 3: assistant")
memory.add_assistant_turn("Mức học bổng loại giỏi là 3.000.000 VNĐ.")
logger.info("  recent size: %d, summary: %r", len(memory.recent), memory.get_summary_text())

logger.info("Turn 4: user (this should roll turn 2 into summary)")
memory.add_user_turn("Vậy điều kiện duy trì học bổng?")
logger.info("  recent size: %d, summary: %r", len(memory.recent), memory.get_summary_text())

logger.info("--- Final prompt history ---")
for msg in memory.get_prompt_history():
    logger.info("  [%s] %s", msg["role"], msg["content"][:120])

logger.info("--- Rendered history for RAG prompt ---")
from generation.memory import build_history_from_memory
print(build_history_from_memory(memory))

logger.info("Done.")