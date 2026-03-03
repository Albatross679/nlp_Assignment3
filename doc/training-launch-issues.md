---
date_created: 2026-03-03
date_modified: 2026-03-03
tags: [debugging, training, environment, part1]
---

# Training Launch Issues & Resolutions

## Overview

When attempting to launch the Part 1 T5 fine-tune training (`part1/train.py`), two
environment issues were identified and resolved before training could run successfully.

---

## Issue 1 — `python` command not found

### Symptom

```
/bin/bash: line 1: python: command not found
```

Exit code `127` when invoking `python -m part1.train`.

### Root Cause

The system only has `python3` on `PATH`; no `python` symlink exists.

### Fix

Use `python3` explicitly for all invocations:

```bash
python3 -m part1.train --config T5FineTuneConfig
```

---

## Issue 2 — HuggingFace cache permission denied

### Symptom

```
PermissionError: [Errno 13] Permission denied: '/workspace/.hf_home/hub'
OSError: PermissionError at /workspace/.hf_home/hub when downloading google-t5/t5-small.
```

Raised at import time inside `part1/data.py` when `T5TokenizerFast.from_pretrained()`
tried to write to the default HF cache.

### Root Cause

The environment variable `HF_HOME` was pointing to `/workspace/.hf_home`, a directory
owned by `root` that the `coder` user cannot write into.

### Fix

Override `HF_HOME` at launch to a user-writable directory (`~/.cache/huggingface`
already existed and had correct permissions):

```bash
HF_HOME=~/.cache/huggingface python3 -m part1.train --config T5FineTuneConfig
```

### Permanent Fix (optional)

Add to `~/.bashrc` or the project's `.env`:

```bash
export HF_HOME=~/.cache/huggingface
```

---

## Final Working Launch Command

```bash
HF_HOME=~/.cache/huggingface python3 -m part1.train --config T5FineTuneConfig
```

To queue a follow-up run automatically after the first completes:

```bash
bash -c '
  while kill -0 <PID> 2>/dev/null; do sleep 30; done
  HF_HOME=~/.cache/huggingface python3 -m part1.train --config T5FineTuneConfig_freeze_encoder
'
```
