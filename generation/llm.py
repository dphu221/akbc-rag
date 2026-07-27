"""Lớp bao LLM Qwen2.5-3B-Instruct sử dụng HuggingFace transformers.

Trình nạp sẽ thử:
  * Dùng lượng tử hóa 4 bit (``bitsandbytes``) trên CUDA khi có để mô hình
    chạy thoải mái trên một GPU Kaggle T4 (16 GB).
  * Dự phòng bằng fp16 trên CUDA khi chưa cài bitsandbytes.
  * Dự phòng bằng fp32 trên CPU (rất chậm; chỉ để gỡ lỗi nhỏ).
  * Dùng ``device_map="auto"`` để phân mảnh mô hình trên nhiều GPU khi cần.

Mọi tham số sinh đều được tinh chỉnh cho câu trả lời tiếng Việt ngắn và thực tế:
  * temperature=0.2   -> phần lớn tất định, giảm bịa thông tin
  * top_p=0.85
  * repetition_penalty=1.05
  * max_new_tokens=512
"""

from __future__ import annotations

import logging
import os
from threading import Lock
from typing import List, Optional

logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


class QwenGenerator:
    """Lớp bao singleton nạp lười quanh mô hình Qwen2.5-3B-Instruct."""

    _instance: Optional["QwenGenerator"] = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):  # noqa: D401 - singleton
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialised = False
        return cls._instance

    def __init__(
        self,
        *,
        device: str = "auto",
        load_in_4bit: Optional[bool] = None,
        dtype: Optional[str] = None,
    ):
        if getattr(self, "_initialised", False):
            return
        self._initialised = True
        self.device = device
        self.model = None
        self.tokenizer = None
        self._load(load_in_4bit=load_in_4bit, dtype=dtype)

    # ------------------------------------------------------------------ #
    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _load(self, *, load_in_4bit: Optional[bool], dtype: Optional[str]) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        device = self._resolve_device()
        logger.info("Loading %s on %s ...", MODEL_NAME, device)

        # Xác định cấu hình lượng tử hóa.
        bnb_config = None
        if device == "cuda":
            want_4bit = load_in_4bit if load_in_4bit is not None else True
            if want_4bit:
                try:
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                    logger.info("Using 4-bit NF4 quantisation.")
                except Exception as exc:  # pragma: no cover
                    logger.warning("bitsandbytes unavailable (%s); using fp16.", exc)
                    bnb_config = None

        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        if dtype == "bf16" and device == "cuda":
            torch_dtype = torch.bfloat16

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        if device == "cpu":
            self.model.to("cpu")
        self.model.eval()
        logger.info("Model loaded. Memory footprint: %.2f MB", self._mem_mb())

    def _mem_mb(self) -> float:
        try:
            return self.model.get_memory_footprint() / 1024 / 1024  # type: ignore[union-attr]
        except Exception:
            return 0.0

    # ------------------------------------------------------------------ #
    # Sinh văn bản
    # ------------------------------------------------------------------ #
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.85,
        repetition_penalty: float = 1.05,
        do_sample: Optional[bool] = None,
    ) -> str:
        """Sinh một câu trả lời của trợ lý cho một lượt (hệ thống, người dùng)."""
        import torch

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        if do_sample is None:
            do_sample = temperature > 0.0

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                **gen_kwargs,
            )
        # Cắt bỏ token đầu vào, chỉ giải mã các token mới.
        in_len = inputs["input_ids"].shape[1]
        new_tokens = out[0][in_len:]
        reply = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return reply


def load_generator(
    *,
    device: str = "auto",
    load_in_4bit: Optional[bool] = None,
) -> QwenGenerator:
    """Hàm khởi tạo tiện dụng."""
    return QwenGenerator(device=device, load_in_4bit=load_in_4bit)
