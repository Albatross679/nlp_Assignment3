# CLAUDE.md

## Repository

GitHub: `git@github.com:Albatross679/nlp_Assignment3.git`
URL: https://github.com/Albatross679/nlp_Assignment3

## Project Overview

CSE 5525 Assignment 3: Natural Language to SQL. Three approaches to translate natural
language flight-database queries into SQL:
1. Fine-tune pretrained T5-small encoder-decoder
2. Train T5-small from scratch (random init)
3. Prompt-based in-context learning with Gemma 2B / CodeGemma 7B

Database: `data/flight_database.db` (25 tables — airlines, flights, restrictions, etc.)
Evaluation: Record F1 (primary), Record EM, SQL EM via `evaluate.py`.

## Project Structure

```
assignment3/
├── .claude/              # Claude Code settings and memory
├── .gitignore
├── README.md
├── CLAUDE.md
├── pyproject.toml        # project metadata and dependencies
├── requirements.txt      # original pip requirements
├── temp.md               # scratch notes (not committed)
├── CSE_5525__Assignment_3__Spring_2026_.pdf  # assignment spec
├── a3-report-template.zip  # LaTeX report template
│
├── data/                 # datasets
│   ├── flight_database.db      # SQLite database
│   ├── flight_database.schema  # schema definition
│   ├── alignment.txt           # table-column alignment
│   ├── train.nl / train.sql    # training pairs
│   ├── dev.nl / dev.sql        # development pairs
│   └── test.nl                 # test queries (no ground-truth SQL)
│
├── records/              # pickled database records
│   └── ground_truth_dev.pkl
├── results/              # predicted SQL output files
├── model/                # saved model weights (not committed)
├── output/               # run outputs (not committed; checkpoints go here per-run)
│
├── train_t5.py           # T5 training loop (Parts 1 & 2)
├── t5_utils.py           # T5 model init, save/load, optimizer setup
├── load_data.py          # T5Dataset, DataLoaders, prompting data loading
├── prompting.py          # LLM prompting pipeline (Part 3)
├── prompting_utils.py    # schema reader, SQL extraction, log saving
├── utils.py              # metrics, record computation, seed setting
├── evaluate.py           # CLI evaluation script
│
├── src/                  # shared source code
│   ├── config.py         # BaseConfig → SLNeuralConfig → SLNeuralClsConfig
│   ├── infrastructure.py # setup_run, save_config, load_config, console logging
│   └── utils/            # generic reusable helpers
│
├── part1/                # T5 fine-tune experiment
│   └── config.py         # T5FineTuneConfig(SLNeuralClsConfig)
├── part2/                # T5 from-scratch experiment
│   └── config.py         # T5ScratchConfig(SLNeuralClsConfig)
├── part3/                # LLM prompting experiment
│   └── config.py         # PromptingConfig(BaseConfig)
│
├── doc/                  # documentation, report PDFs
├── media/                # images, plots
├── script/               # standalone utility scripts
```

## Do Not Rename or Move

All files from the original starter zip (`CSE_5525_HW3.zip`) must keep their original
names and paths. The submission is graded by file name/location. These are:

```
evaluate.py              # root
load_data.py             # root
prompting.py             # root
prompting_utils.py       # root
t5_utils.py              # root
train_t5.py              # root
utils.py                 # root
requirements.txt         # root
data/                    # entire directory as-is
records/ground_truth_dev.pkl
results/                 # output directory (graded file names below)
```

Required submission output files (name and path must match exactly):
- `results/{t5_ft,t5_scr,llm}_test.sql`
- `records/{t5_ft,t5_scr,llm}_test.pkl`

## Do Not Modify Contents

- `data/` — all dataset files (`.nl`, `.sql`, `.db`, `.schema`)
- `records/ground_truth_dev.pkl` — ground-truth dev records

## Files You Can Modify

- `train_t5.py` — implement `eval_epoch`, `test_inference`; tune hyperparameters
- `t5_utils.py` — implement `initialize_model`, `save_model`, `load_model_from_checkpoint`
- `load_data.py` — implement `T5Dataset`, collate functions, `load_prompting_data`
- `prompting.py` — implement `create_prompt`, `exp_kshot`, `eval_outputs`
- `prompting_utils.py` — implement `read_schema`, `extract_sql_query`

## Todoist Tracking

Project **nlp_as3** (ID: `6g5H7Vvvwhx95Pvv`) tracks progress for this assignment.
Tasks map to the three parts:
- `part1` — T5 fine-tune
- `part2` — T5 from scratch
- `part3` — LLM prompting

## MLflow

Always use **port 8080** when launching MLflow in this virtual machine:

```bash
mlflow ui --port 8080
```

## Key Commands

```bash
# Train T5 fine-tune
python train_t5.py --finetune --max_n_epochs 20 --learning_rate 1e-4 --patience_epochs 5

# Train T5 from scratch
python train_t5.py --max_n_epochs 50 --learning_rate 1e-3

# Run prompting
python prompting.py --shot 3 --model gemma

# Evaluate
python evaluate.py --predicted_sql results/t5_ft_dev.sql --predicted_records records/t5_ft_dev.pkl --development_sql data/dev.sql --development_records records/ground_truth_dev.pkl
```
