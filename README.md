# Assignment 3 — Natural Language to SQL

CSE 5525, Spring 2026

## Overview

Supervised sequence prediction: translating natural language instructions into SQL queries using three approaches:
1. **Part 1** — Fine-tuning pretrained T5-small (encoder-decoder)
2. **Part 2** — Training T5 from scratch (randomly initialized)
3. **Part 3** — In-context learning with Gemma / CodeGemma LLMs

Evaluation uses Record F1, Record Exact Match, and SQL Query Exact Match on `flight_database.db`.

## Setup

```bash
pip install -e ".[dev]"
```

## Evaluation

```bash
python evaluate.py \
  --predicted_sql results/t5_ft_dev.sql \
  --predicted_records records/t5_ft_dev.pkl \
  --development_sql data/dev.sql \
  --development_records records/ground_truth_dev.pkl
```

## Submission

Upload a `.zip` to Gradescope containing all data, code, results, records, and the report PDF.
Exclude saved models and checkpoint files.

Required result files in `results/`: `{t5_ft, t5_scr, llm}_test.sql`
Required record files in `records/`: `{t5_ft, t5_scr, llm}_test.pkl`
