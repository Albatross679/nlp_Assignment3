"""Part 3: LLM prompting config. Inherits BaseConfig (no training loop)."""

from dataclasses import dataclass, field
from src.config import BaseConfig


@dataclass
class PromptingConfig(BaseConfig):
    # Model
    model_name: str = "gemma"          # "gemma" or "codegemma"
    quantization: bool = False

    # Prompting
    shot: int = 0
    prompt_type: int = 0
    max_new_tokens: int = 256

    # Eval metrics
    eval_metrics: list = field(default_factory=lambda: [
        "record_f1", "record_em", "sql_em", "error_rate",
    ])

    def __post_init__(self):
        if self.name == "experiment":
            self.name = f"llm_{self.model_name}_{self.shot}shot"
        # No training loop — disable training-only infra
        self.checkpointing.enabled = False
        self.metricslog.enabled = False
