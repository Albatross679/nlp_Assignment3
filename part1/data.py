"""Part 1 data: T5Dataset, collate functions, and dataloader construction."""

import os
import json
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

# Shared tokenizer instance — avoids re-instantiating on every collate call
_TOKENIZER = T5TokenizerFast.from_pretrained("google-t5/t5-small")
_BOS_ID = _TOKENIZER.convert_tokens_to_ids("<extra_id_0>")


def _load_schema_string(schema_path="data/flight_database.schema"):
    """Load schema as a compact table-names-only listing (~94 tokens)."""
    with open(schema_path) as f:
        schema = json.load(f)
    tables = list(schema.get("ents", {}).keys())
    return "tables: " + ", ".join(tables) + " "


class T5Dataset(Dataset):
    """
    Dataset for T5 encoder-decoder NL-to-SQL.

    Uses 'google-t5/t5-small' tokenizer for both encoder (NL) and decoder (SQL).
    Decoder gets an extra_id BOS token. Test split has no SQL targets.
    """

    def __init__(self, data_folder, split, input_prefix="", include_schema=False):
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
        self.bos_token_id = self.tokenizer.convert_tokens_to_ids("<extra_id_0>")
        self.input_prefix = input_prefix
        self.schema_string = _load_schema_string() if include_schema else ""
        self.encoder_inputs, self.decoder_targets = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl_path = os.path.join(data_folder, f"{split}.nl")
        lines = _load_lines(nl_path)
        prefix = self.schema_string + self.input_prefix
        if prefix:
            lines = [prefix + line for line in lines]
        encoder_inputs = tokenizer(
            lines, padding=False, truncation=True, return_attention_mask=False,
        )["input_ids"]

        if split == "test":
            return encoder_inputs, None

        sql_path = os.path.join(data_folder, f"{split}.sql")
        decoder_targets = tokenizer(
            _load_lines(sql_path), padding=False, truncation=True, return_attention_mask=False
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


def normal_collate_fn(batch):
    """
    Dynamic padding for train/dev. Each sample is (encoder_ids, decoder_target_ids).

    Returns:
        encoder_ids:            BxT   — input to T5 encoder
        encoder_mask:           BxT   — 1 for real tokens, 0 for padding
        decoder_inputs:         BxT'  — [BOS] + target[:-1] (teacher forcing)
        decoder_targets:        BxT'  — target tokens (supervision signal)
        initial_decoder_inputs: Bx1   — [BOS] only (for autoregressive eval)
    """
    enc_list, dec_list = zip(*batch)
    bos_id = _BOS_ID

    # Pad encoder
    encoder_ids = pad_sequence(enc_list, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()

    # Decoder: prepend BOS, build inputs (shifted right) and targets
    decoder_targets = pad_sequence(dec_list, batch_first=True, padding_value=PAD_IDX)
    bos_column = torch.full((decoder_targets.size(0), 1), bos_id, dtype=torch.long)
    decoder_inputs = torch.cat([bos_column, decoder_targets[:, :-1]], dim=1)

    initial_decoder_inputs = bos_column  # Bx1

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs


def test_collate_fn(batch):
    """
    Dynamic padding for test set (no targets).

    Returns:
        encoder_ids:            BxT
        encoder_mask:           BxT
        initial_decoder_inputs: Bx1
    """
    enc_list = [sample[0] for sample in batch]
    bos_id = _BOS_ID

    encoder_ids = pad_sequence(enc_list, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    initial_decoder_inputs = torch.full((encoder_ids.size(0), 1), bos_id, dtype=torch.long)

    return encoder_ids, encoder_mask, initial_decoder_inputs


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
