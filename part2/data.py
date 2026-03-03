"""Part 2 data: re-exports from part1 (identical T5 data pipeline)."""

from part1.data import (
    PAD_IDX,
    T5Dataset,
    _TOKENIZER,
    get_dataloader,
    load_t5_data,
    normal_collate_fn,
    test_collate_fn,
)

__all__ = [
    "PAD_IDX",
    "T5Dataset",
    "_TOKENIZER",
    "get_dataloader",
    "load_t5_data",
    "normal_collate_fn",
    "test_collate_fn",
]
