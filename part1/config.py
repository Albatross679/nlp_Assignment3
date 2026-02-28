"""Part 1: T5 fine-tune config v3 — improved settings. Inherits SLNeuralClsConfig."""

from dataclasses import dataclass
from typing import Optional
from src.config import SLNeuralClsConfig


@dataclass
class T5FineTuneConfig(SLNeuralClsConfig):
    name: str = "t5_ft_v3"
    model_checkpoint: str = "google-t5/t5-small"
    finetune: bool = True

    # ── Training hyperparameters ─────────────────────────────────────────
    num_epochs: int = 30
    batch_size: int = 16
    test_batch_size: int = 16
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    scheduler: str = "cosine"
    patience_epochs: int = 7
    num_warmup_epochs: int = 2
    grad_clip_norm: float = 1.0
    dropout: float = 0.1

    # ── Layer freezing (optional, all disabled by default) ────────────────
    freeze_encoder: bool = False
    freeze_embeddings: bool = False
    unfreeze_last_n_decoder: Optional[int] = None

    # ── Input formatting ─────────────────────────────────────────────────
    input_prefix: str = "translate English to SQL: "
    include_schema: bool = False

    # ── Resume / time budget ─────────────────────────────────────────────
    resume_run_dir: Optional[str] = None
    max_wall_clock_hours: Optional[float] = 5

    # ── Decoding ─────────────────────────────────────────────────────────
    max_new_tokens: int = 256
    num_beams: int = 1

    def __post_init__(self):
        super().__post_init__()

@dataclass
class T5FineTuneConfig_freeze_encoder(T5FineTuneConfig):
    name: str = "t5_ft_freeze_encoder"
    freeze_encoder: bool = True
    num_beams: int = 3
    input_prefix: str = ""
    
    