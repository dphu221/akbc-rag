"""Gói sinh văn bản: LLM Qwen2.5-3B-Instruct + prompt RAG tiếng Việt + bộ nhớ tóm tắt."""

from .llm import QwenGenerator  # noqa: F401
from .memory import SummaryMemory, build_history_from_memory  # noqa: F401
from .prompts import build_rag_prompt, SYSTEM_PROMPT, REFUSAL  # noqa: F401

__all__ = [
    "QwenGenerator",
    "SummaryMemory",
    "build_history_from_memory",
    "build_rag_prompt",
    "SYSTEM_PROMPT",
    "REFUSAL",
]
