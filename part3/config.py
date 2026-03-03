"""Part 3: LLM prompting config. Inherits BaseConfig (no gradient training)."""

from dataclasses import dataclass, field
from typing import Optional
from src.config import BaseConfig


@dataclass
class PromptingConfig(BaseConfig):
    name: str = "llm_v1"

    # ── Model ────────────────────────────────────────────────────────────
    model_name: str = "gemma"            # "gemma" or "codegemma"
    quantize: bool = False               # 4-bit quantization (for codegemma-7b)

    # ── Prompting ────────────────────────────────────────────────────────
    shot: int = 0                        # k-shot (0 = zero-shot)
    prompt_type: int = 0                 # prompt template variant
    max_new_tokens: int = 256

    # ── Evaluation ───────────────────────────────────────────────────────
    eval_splits: list = field(default_factory=lambda: ["dev", "test"])

    # ── Resume / time budget ─────────────────────────────────────────────
    max_wall_clock_hours: Optional[float] = 5
