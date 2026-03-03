"""Part 3 data: load NL/SQL text files for prompting (no tokenization)."""

import os


def _load_lines(path):
    with open(path) as f:
        return [line.strip() for line in f.readlines()]


def load_prompting_data(data_folder="data"):
    """Load train/dev/test splits as raw text lists.

    Returns:
        train_x, train_y, dev_x, dev_y, test_x
    """
    train_x = _load_lines(os.path.join(data_folder, "train.nl"))
    train_y = _load_lines(os.path.join(data_folder, "train.sql"))
    dev_x = _load_lines(os.path.join(data_folder, "dev.nl"))
    dev_y = _load_lines(os.path.join(data_folder, "dev.sql"))
    test_x = _load_lines(os.path.join(data_folder, "test.nl"))
    return train_x, train_y, dev_x, dev_y, test_x
