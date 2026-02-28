"""Part 1 data: T5Dataset, collate functions, and dataloader construction."""

import json
import os

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import T5TokenizerFast

PAD_IDX = 0
MODEL_CHECKPOINT = "google-t5/t5-small"

# Shared tokenizer — instantiated once, reused everywhere
_TOKENIZER = T5TokenizerFast.from_pretrained(MODEL_CHECKPOINT)
_BOS_ID = _TOKENIZER.convert_tokens_to_ids("<extra_id_0>")

# Cached schema string (loaded on first use)
_SCHEMA_CACHE = None


def _load_schema_string(schema_path="data/flight_database.schema"):
    """Load schema as a compact table-names-only listing (~94 tokens). Cached."""
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is not None:
        return _SCHEMA_CACHE
    with open(schema_path) as f:
        schema = json.load(f)
    tables = list(schema.get("ents", {}).keys())
    _SCHEMA_CACHE = "tables: " + ", ".join(tables) + " "
    return _SCHEMA_CACHE


class T5Dataset(Dataset):
    """
    Dataset for T5 encoder-decoder NL-to-SQL.

    Uses the shared T5-small tokenizer for both encoder (NL) and decoder (SQL).
    Test split has no SQL targets.
    """

    def __init__(self, data_folder, split, input_prefix="", include_schema=False):
        self.split = split
        self.tokenizer = _TOKENIZER
        self.input_prefix = input_prefix
        self.schema_string = _load_schema_string() if include_schema else ""
        self.encoder_inputs, self.decoder_targets = self._process_data(data_folder, split)

    def _process_data(self, data_folder, split):
        nl_path = os.path.join(data_folder, f"{split}.nl")
        lines = _load_lines(nl_path)
        prefix = self.schema_string + self.input_prefix
        if prefix:
            lines = [prefix + line for line in lines]
        encoder_inputs = self.tokenizer(
            lines, padding=False, truncation=True, return_attention_mask=False,
        )["input_ids"]

        if split == "test":
            return encoder_inputs, None

        sql_path = os.path.join(data_folder, f"{split}.sql")
        decoder_targets = self.tokenizer(
            _load_lines(sql_path), padding=False, truncation=True, return_attention_mask=False,
        )["input_ids"]
        return encoder_inputs, decoder_targets

    def __len__(self):
        return len(self.encoder_inputs)

    def __getitem__(self, idx):
        enc = torch.tensor(self.encoder_inputs[idx], dtype=torch.long)
        if self.decoder_targets is not None:
            dec = torch.tensor(self.decoder_targets[idx], dtype=torch.long)
            return enc, dec
        return (enc,)


def _pad_encoder(enc_list):
    """Shared encoder padding + BOS column construction."""
    encoder_ids = pad_sequence(enc_list, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    bos_column = torch.full((encoder_ids.size(0), 1), _BOS_ID, dtype=torch.long)
    return encoder_ids, encoder_mask, bos_column


def normal_collate_fn(batch):
    """
    Dynamic padding for train/dev. Each sample is (encoder_ids, decoder_target_ids).

    Returns:
        encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs
    """
    enc_list, dec_list = zip(*batch)
    encoder_ids, encoder_mask, bos_column = _pad_encoder(enc_list)

    decoder_targets = pad_sequence(dec_list, batch_first=True, padding_value=PAD_IDX)
    decoder_inputs = torch.cat([bos_column, decoder_targets[:, :-1]], dim=1)

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, bos_column


def test_collate_fn(batch):
    """
    Dynamic padding for test set (no targets).

    Returns:
        encoder_ids, encoder_mask, initial_decoder_inputs
    """
    enc_list = [sample[0] for sample in batch]
    return _pad_encoder(enc_list)


def get_dataloader(batch_size, split, input_prefix="", include_schema=False,
                   train_fraction=1.0, seed=42):
    dset = T5Dataset("data", split, input_prefix=input_prefix, include_schema=include_schema)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    if split == "train" and train_fraction < 1.0:
        n = len(dset)
        k = max(1, int(n * train_fraction))
        rng = torch.Generator().manual_seed(seed)
        indices = torch.randperm(n, generator=rng)[:k].tolist()
        dset = torch.utils.data.Subset(dset, indices)

    return DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def load_t5_data(batch_size, test_batch_size, input_prefix="", include_schema=False,
                 train_fraction=1.0, seed=42):
    train_loader = get_dataloader(batch_size, "train", input_prefix, include_schema,
                                  train_fraction=train_fraction, seed=seed)
    dev_loader = get_dataloader(test_batch_size, "dev", input_prefix, include_schema)
    test_loader = get_dataloader(test_batch_size, "test", input_prefix, include_schema)
    return train_loader, dev_loader, test_loader


def _load_lines(path):
    with open(path) as f:
        return [line.strip() for line in f.readlines()]
