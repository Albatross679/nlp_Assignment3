"""
ML Configuration System for Assignment 3: NL-to-SQL.

Hierarchy:
  BaseConfig                    — output dir, console logging, checkpointing, metrics log
  └── SLNeuralConfig            — epoch-based, gradient-based training
      └── SLNeuralClsConfig     — classification (CrossEntropyLoss over vocab)
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch


def resolve_device(device: str = "auto") -> str:
    """Resolve a device string. 'auto' picks cuda > mps > cpu."""
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


# ── Built-in infrastructure groups ──────────────────────────────────────────

@dataclass
class OutputConfig:
    base_dir: str = "output"
    save_config: bool = True
    timestamp_format: str = "%Y%m%d_%H%M%S"
    subdirs: dict = field(default_factory=lambda: {
        "checkpoints": "checkpoints",
        "plots": "plots",
    })


@dataclass
class ConsoleConfig:
    enabled: bool = True
    filename: str = "console.log"
    tee_to_console: bool = True
    flush_frequency: int = 1


@dataclass
class CheckpointingConfig:
    enabled: bool = True
    save_best: bool = True
    save_last: bool = True
    save_every_n: int = 0
    save_training_state: bool = True   # saves optimizer/scheduler for resume; set False to save ~2 GB/run
    metric: str = "record_f1"
    mode: str = "max"
    best_filename: str = "model_best.pt"
    last_filename: str = "model_last.pt"


@dataclass
class MetricsLogConfig:
    enabled: bool = True
    filename: str = "metrics.jsonl"
    flush_every_epoch: bool = True


# ── Level 0: Base ───────────────────────────────────────────────────────────

@dataclass
class BaseConfig:
    name: str = "experiment"
    seed: int = 42
    device: str = "auto"

    output: OutputConfig = field(default_factory=OutputConfig)
    console: ConsoleConfig = field(default_factory=ConsoleConfig)
    checkpointing: CheckpointingConfig = field(default_factory=CheckpointingConfig)
    metricslog: MetricsLogConfig = field(default_factory=MetricsLogConfig)

    def __post_init__(self):
        self.device = resolve_device(self.device)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str):
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_dict(cls, d: dict):
        """Recursively reconstruct nested dataclasses from a plain dict."""
        field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
        kwargs = {}
        for k, v in d.items():
            if k in field_types and isinstance(v, dict):
                nested_cls = _resolve_type(field_types[k])
                if nested_cls is not None:
                    v = nested_cls(**v)
            kwargs[k] = v
        return cls(**kwargs)


# ── Level 1a: SL Neural ────────────────────────────────────────────────────

@dataclass
class SLNeuralConfig(BaseConfig):
    # Training
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0

    # Optimizer & scheduler
    optimizer: str = "AdamW"
    scheduler: Optional[str] = "cosine"   # None, "cosine", "linear"
    num_warmup_epochs: int = 0

    # Regularization
    grad_clip_norm: Optional[float] = None
    dropout: float = 0.0

    # Evaluation frequency
    eval_every_n_epochs: int = 1          # run eval every N epochs (1 = every epoch)

    # Early stopping
    patience_epochs: int = 0              # 0 disables

    # Logging flags
    log_system_metrics: bool = True


# ── Level 2b: SL Neural Classification ─────────────────────────────────────

@dataclass
class SLNeuralClsConfig(SLNeuralConfig):
    loss_fn: str = "cross_entropy"
    eval_metrics: list = field(default_factory=lambda: [
        "record_f1", "record_em", "sql_em", "error_rate",
    ])
    sql_num_threads: int = field(default_factory=lambda: min(423, os.cpu_count() or 32))


# ── Helpers ─────────────────────────────────────────────────────────────────

_NESTED_TYPES = {
    "OutputConfig": OutputConfig,
    "ConsoleConfig": ConsoleConfig,
    "CheckpointingConfig": CheckpointingConfig,
    "MetricsLogConfig": MetricsLogConfig,
}


def _resolve_type(type_hint: str) -> type | None:
    """Resolve a type hint string to the actual dataclass type."""
    for name, cls in _NESTED_TYPES.items():
        if name in str(type_hint):
            return cls
    return None
