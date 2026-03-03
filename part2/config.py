"""Part 2: T5 from-scratch config — random initialization. Inherits SLNeuralClsConfig."""

import os
from dataclasses import dataclass, field
from typing import Optional
from src.config import SLNeuralClsConfig


@dataclass
class T5ScratchConfig(SLNeuralClsConfig):
    name: str = "t5_scr_v1"
    model_checkpoint: str = "google-t5/t5-small"
    finetune: bool = False

    # ── Training hyperparameters ─────────────────────────────────────────
    num_epochs: int = 50
    batch_size: int = 32
    test_batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    scheduler: str = "cosine"
    patience_epochs: int = 10
    num_warmup_epochs: int = 3
    grad_clip_norm: float = 1.0
    dropout: float = 0.1

    # ── Input formatting ─────────────────────────────────────────────────
    input_prefix: str = ""
    include_schema: bool = False

    # ── Resume / time budget ─────────────────────────────────────────────
    resume_run_dir: Optional[str] = None
    max_wall_clock_hours: Optional[float] = 8

    # ── Decoding ─────────────────────────────────────────────────────────
    max_new_tokens: int = 256
    num_beams: int = 1

    # ── Evaluation ───────────────────────────────────────────────────────
    sql_num_threads: int = field(default_factory=lambda: min(423, os.cpu_count() or 32))

    save_training_state: bool = False
