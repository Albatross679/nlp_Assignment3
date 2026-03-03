---
date_created: 2026-03-03
date_modified: 2026-03-03
tags: [training, t5-base, part1, monitoring]
---

# T5-Base Sequential Training Run

Sequential run of four T5-base fine-tune configs via `script/run_base_configs.sh`.
Configs: `T5FineTuneConfig_base`, `T5FineTuneConfig_base2`, `T5FineTuneConfig_base3`, `T5FineTuneConfig_base4`.

---

## Config Summary

| Config | Name | Changes vs base |
|---|---|---|
| `T5FineTuneConfig_base` | `t5_ft_base_v1` | Baseline T5-base, batch=16, beams=2 |
| `T5FineTuneConfig_base2` | `t5_ft_base_v2` | + `freeze_encoder=True` |
| `T5FineTuneConfig_base3` | `t5_ft_base_v3` | + `freeze_embeddings=True` |
| `T5FineTuneConfig_base4` | `t5_ft_base_v4` | `lr=1e-5`, `num_epochs=40` |

---

## Run Log

### Run 1 (configs 1–4, initial attempt)
- **Date**: 2026-03-03 04:22 UTC
- **Script**: `script/run_base_configs.sh`
- **Log file**: `/tmp/t5_base_train.log`
- **Result**: Config 1 completed. Process hung after cleanup (`gc.collect()` / `torch.cuda.empty_cache()` deadlock on futex). Manual kill caused script to abort (exit 143 treated as non-OOM failure).

### Run 2 (configs 2–4, resumed)
- **Date**: 2026-03-03 13:22 UTC
- **Script**: `nohup bash script/run_base_configs.sh --skip T5FineTuneConfig_base`
- **Log file**: `/tmp/t5_base_train_r2.log`
- **Improvements**: Added `--skip` flag, `PYTHONUNBUFFERED=1`, watchdog to kill hung processes after MLflow marks run FINISHED + 5 min grace.

---

## Issues & Fixes

### Issue 1: `python: command not found`
- **Error**: `script/run_base_configs.sh: line 26: python: command not found` (exit 127)
- **Cause**: System only has `python3`; no `python` symlink exists.
- **Fix**: Changed `python` → `python3` in the script.

### Issue 2: `PermissionError` on HuggingFace cache
- **Error**: `PermissionError: [Errno 13] Permission denied: '/workspace/.hf_home/hub'`
- **Cause**: System env has `HF_HOME=/workspace/.hf_home`, owned by root. Models already cached at `/home/coder/.cache/huggingface/hub/`.
- **Fix**: Added `export HF_HOME=/home/coder/.cache/huggingface` to the script.

### Issue 3: Process hung after training completion
- **Symptom**: PID 110973 (`t5_ft_base_v1`) completed all work (MLflow FINISHED, test outputs saved) but blocked indefinitely on `futex_wait_queue_me` during `gc.collect()` / `torch.cuda.empty_cache()`.
- **Cause**: Likely PyTorch/CUDA threading deadlock during GPU memory cleanup.
- **Impact**: Sequential script could not proceed to config 2.
- **Fix**: Added watchdog co-process to `run_base_configs.sh` that polls MLflow status every 60s and kills the training process if it's still alive 5 min after the run is marked FINISHED. Also treats exit codes 137 (SIGKILL) and 143 (SIGTERM) as success when coming from the watchdog.

### Issue 4: Buffered stdout with nohup/pipe
- **Symptom**: Log file showed no output for long periods; `tee` did not receive lines until Python's stdout buffer flushed.
- **Cause**: Python buffers stdout when not connected to a TTY (e.g., piped through `tee` or redirected by `nohup`).
- **Fix**: Added `export PYTHONUNBUFFERED=1` to the script.

---

## Results

| Config | Best record_f1 | Best record_em | Best sql_em | Epochs | Wall clock |
|---|---|---|---|---|---|
| `t5_ft_base_v1` | **0.5885** | 0.5601 | 0.0150 | 21 (best@17) | 5.1h |
| `t5_ft_base_v2` | _running_ | | | | |
| `t5_ft_base_v3` | _pending_ | | | | |
| `t5_ft_base_v4` | _pending_ | | | | |

### Config 1 epoch-by-epoch (selected)

| Epoch | record_f1 | error_rate | Notes |
|---|---|---|---|
| 0 | 0.1180 | 1.00 | Warmup, all SQL broken |
| 5 | 0.4931 | 0.32 | |
| 8 | 0.5507 | 0.36 | |
| 11 | 0.5702 | 0.37 | |
| 14 | 0.5804 | 0.36 | New best after 2-epoch dip |
| 17 | **0.5885** | 0.38 | Final best |
| 21 | 0.5748 | 0.37 | Early stop (patience=7) |
