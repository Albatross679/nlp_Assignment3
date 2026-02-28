"""Part 2: T5 from-scratch config. Inherits SLNeuralClsConfig."""

from dataclasses import dataclass
from src.config import SLNeuralClsConfig


@dataclass
class T5ScratchConfig(SLNeuralClsConfig):
    model_checkpoint: str = "google-t5/t5-small"
    finetune: bool = False

    # Override SL Neural defaults for training from scratch
    num_epochs: int = 50
    batch_size: int = 16
    test_batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    scheduler: str = "cosine"
    patience_epochs: int = 0          # no early stopping by default

    def __post_init__(self):
        if self.name == "experiment":
            self.name = "t5_scr"
