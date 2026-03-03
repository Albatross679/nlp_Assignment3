---
date_created: 2026-03-03
date_modified: 2026-03-03
tags: [performance, evaluation, optimization, training]
---

# Evaluation Performance Optimization

## Problem Identification

During Part 1 T5 fine-tuning, we observed that each training epoch took ~230s total, broken down as:

| Phase | Time | Resource |
|-------|------|----------|
| Training (teacher forcing) | ~51s | GPU |
| `model.generate()` on dev set | ~50s | GPU (autoregressive) |
| SQL execution on database | ~120s | CPU |
| **Total** | **~230s** | |

Training itself is only **22% of wall-clock time**. The other 78% is evaluation overhead.

### Root Cause: `eval_epoch` Does Three Serial Steps

**Step 1 — Dev loss** ([part1/train.py:240-258](../part1/train.py#L240-L258))
Teacher-forcing forward pass over the dev set. Fast (~5s) because it's a single forward pass per sample.

**Step 2 — Autoregressive generation** ([part1/train.py:262](../part1/train.py#L262))
`model.generate()` with `max_new_tokens=256`. For each dev sample, the model runs up to 256 sequential forward passes (one per output token). 423 dev samples × up to 256 steps = ~100k GPU operations. This is inherently slow and cannot be parallelized.

**Step 3 — SQL execution** ([utils.py:85-125](../utils.py#L85-L125))
Each generated SQL string is executed against `data/flight_database.db` to retrieve records for F1 scoring. Originally used 10 threads for 423 queries, giving ~42 queries per thread serially.

---

## Why Step 3 Was the Biggest Target

- Steps 1 and 2 use the GPU, which is already at 80% VRAM utilization — no room to overlap with training.
- Step 3 is purely CPU/I/O: SQLite reads that release Python's GIL and can run in true parallel.
- In early training (epochs 0–7), error rate is 90–96%: the model outputs the same malformed SQL repeatedly. Running identical queries against the DB every epoch is pure waste.

---

## Solutions Implemented

### Fix 1 — Increase Thread Count (`utils.py:94`)

```python
# Before
num_threads = 10

# After
num_threads = 32
```

SQLite supports concurrent reads. With 32 threads, 423 queries fan out to ~14 per thread instead of ~42. Estimated speedup: ~3× on SQL execution time.

### Fix 2 — Prediction Caching (`part1/train.py:264-276`)

After GPU generation, hash the full list of predicted SQL strings. If the hash matches the previous epoch's predictions, skip SQL execution entirely and return the cached metrics.

```python
preds_key = hash(tuple(all_preds))
if pred_cache is not None and pred_cache.get("key") == preds_key:
    # Reuse cached result — no DB calls
    sql_em, record_em, record_f1, error_msgs = pred_cache["metrics"]
else:
    # Execute SQL and update cache
    save_queries_and_records(all_preds, model_sql_path, model_record_path)
    sql_em, record_em, record_f1, error_msgs = compute_metrics(...)
    pred_cache["key"] = preds_key
    pred_cache["metrics"] = (sql_em, record_em, record_f1, error_msgs)
```

The `_pred_cache` dict is initialized once in `train()` and passed to `eval_epoch` each epoch.

**When this fires:** During early training when the model outputs the same invalid SQL for consecutive epochs. Saves the full ~120s SQL execution cost for those epochs.

---

## Options Not Implemented

**Async SQL eval (Option A):** Overlap SQL execution with the next training epoch using a background thread. Would save ~50s per epoch (the training-SQL overlap window). Not implemented because it requires restructuring the checkpoint/early-stopping logic — decisions for epoch N would need to be deferred to the start of epoch N+2. Added complexity not worth it given the caching already covers the worst case.

**Evaluate every N epochs (Option C):** Skip full eval on non-milestone epochs. Simple but delays observing F1 improvement, which matters for early stopping decisions.

---

## Expected Impact

| Scenario | Before | After |
|----------|--------|-------|
| Predictions unchanged (cache hit) | ~230s | ~105s (train + GPU gen only) |
| Predictions changed, 32 threads | ~230s | ~145s (SQL ~40s instead of ~120s) |

In early training (epochs 0–7), the cache fires frequently, roughly halving total training time for those epochs.
