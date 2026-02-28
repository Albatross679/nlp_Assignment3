#!/usr/bin/env python3
"""Load a T5 fine-tune checkpoint from an MLflow run (or output dir) and evaluate on dev examples.

Usage examples:
    # Auto-find the best MLflow run for part1
    python part1/eval_checkpoint.py

    # Specify an output directory directly
    python part1/eval_checkpoint.py --run_dir output/t5_ft_20260228_045307

    # Use a specific MLflow run ID
    python part1/eval_checkpoint.py --mlflow_run_id b31e35a7ddc0495e8b993c5d852e9fa2

    # Show more examples
    python part1/eval_checkpoint.py --num_examples 20

    # Use last checkpoint instead of best
    python part1/eval_checkpoint.py --use_last
"""

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

import torch

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from part1.model import load_model_from_checkpoint
from part1.data import get_dataloader, _TOKENIZER
from utils import compute_metrics, save_queries_and_records

DB_PATH = "data/flight_database.db"


def find_best_mlflow_run(experiment_name="part1_t5_finetune"):
    """Find the MLflow run with the best record_f1."""
    try:
        import mlflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        runs = mlflow.search_runs(
            experiment_names=[experiment_name],
            order_by=["metrics.record_f1 DESC"],
            max_results=1,
        )
        if runs.empty:
            print(f"No MLflow runs found for experiment '{experiment_name}'")
            return None
        row = runs.iloc[0]
        run_id = row["run_id"]
        best_f1 = row.get("metrics.record_f1", "N/A")
        print(f"Best MLflow run: {run_id}  (record_f1 = {best_f1})")
        return run_id
    except Exception as e:
        print(f"MLflow lookup failed: {e}")
        return None


def find_run_dir_from_mlflow(run_id):
    """Given an MLflow run_id, find the corresponding output directory."""
    import mlflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    run = mlflow.get_run(run_id)
    # Check if there's a run_dir param
    run_dir = run.data.params.get("run_dir", None)
    if run_dir and os.path.isdir(run_dir):
        return run_dir
    # Try to match by run_name tag
    run_name = run.data.tags.get("mlflow.runName", "")
    if run_name:
        for d in sorted(os.listdir("output"), reverse=True):
            if d.startswith(run_name) or run_name in d:
                candidate = os.path.join("output", d)
                if os.path.isdir(candidate):
                    return candidate
    return None


def find_latest_output_dir(prefix="t5_ft"):
    """Find the most recent output directory matching the prefix."""
    output_base = "output"
    if not os.path.isdir(output_base):
        return None
    dirs = sorted(
        [d for d in os.listdir(output_base) if d.startswith(prefix)],
        reverse=True,
    )
    if not dirs:
        return None
    return os.path.join(output_base, dirs[0])


def load_config(run_dir):
    """Load config.json from a run directory."""
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_path):
        return {}
    with open(config_path) as f:
        return json.load(f)


def load_metrics(run_dir):
    """Load the metrics.jsonl from a run directory and return the best epoch info."""
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(metrics_path):
        return None
    entries = []
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    if not entries:
        return None
    best = max(entries, key=lambda e: e.get("record_f1", 0))
    return {"all_epochs": entries, "best": best}


def execute_sql(query):
    """Execute a SQL query against the flight database and return results."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        results = cursor.fetchall()
        error = None
    except Exception as e:
        results = []
        error = f"{type(e).__name__}: {e}"
    conn.close()
    return results, error


def main():
    parser = argparse.ArgumentParser(description="Evaluate a T5 fine-tune checkpoint")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Path to output run directory (e.g. output/t5_ft_20260228_045307)")
    parser.add_argument("--mlflow_run_id", type=str, default=None,
                        help="MLflow run ID to look up")
    parser.add_argument("--use_last", action="store_true",
                        help="Use model_last.pt instead of model_best.pt")
    parser.add_argument("--num_examples", type=int, default=10,
                        help="Number of examples to display in detail")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    # ── 1. Locate the run directory ──────────────────────────────────────────
    run_dir = args.run_dir
    if run_dir is None and args.mlflow_run_id:
        run_dir = find_run_dir_from_mlflow(args.mlflow_run_id)
        if run_dir is None:
            print(f"Could not find output dir for MLflow run {args.mlflow_run_id}")

    if run_dir is None:
        # Try MLflow best run
        best_run_id = find_best_mlflow_run()
        if best_run_id:
            run_dir = find_run_dir_from_mlflow(best_run_id)

    if run_dir is None:
        # Fallback: latest output dir
        run_dir = find_latest_output_dir("t5_ft")

    if run_dir is None or not os.path.isdir(run_dir):
        print("ERROR: No run directory found. Use --run_dir to specify one.")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"Run directory: {run_dir}")
    print(f"{'='*70}")

    # ── 2. Load config ───────────────────────────────────────────────────────
    cfg = load_config(run_dir)
    print(f"\nConfig:")
    for key in ["finetune", "model_checkpoint", "num_epochs", "batch_size",
                 "learning_rate", "scheduler", "dropout", "patience_epochs",
                 "max_new_tokens", "num_beams", "input_prefix"]:
        if key in cfg:
            print(f"  {key}: {cfg[key]}")

    # ── 3. Load training metrics summary ─────────────────────────────────────
    metrics_info = load_metrics(run_dir)
    if metrics_info:
        best = metrics_info["best"]
        num_epochs = len(metrics_info["all_epochs"])
        print(f"\nTraining summary: {num_epochs} epochs completed")
        print(f"  Best epoch: {best['epoch']}")
        print(f"  Best record_f1:  {best.get('record_f1', 'N/A'):.4f}")
        print(f"  Best record_em:  {best.get('record_em', 'N/A'):.4f}")
        print(f"  Best sql_em:     {best.get('sql_em', 'N/A'):.4f}")
        print(f"  Best error_rate: {best.get('error_rate', 'N/A'):.4f}")
        print(f"  Best dev_loss:   {best.get('dev_loss', 'N/A'):.4f}")

    # ── 4. Locate checkpoint ─────────────────────────────────────────────────
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        print(f"ERROR: No checkpoints directory at {ckpt_dir}")
        sys.exit(1)

    use_best = not args.use_last
    ckpt_file = "model_best.pt" if use_best else "model_last.pt"
    ckpt_path = os.path.join(ckpt_dir, ckpt_file)
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"\nLoading checkpoint: {ckpt_path}")

    # ── 5. Load model ────────────────────────────────────────────────────────
    finetune = cfg.get("finetune", True)
    model_checkpoint = cfg.get("model_checkpoint", "google-t5/t5-small")
    dropout = cfg.get("dropout", 0.0)

    model = load_model_from_checkpoint(
        ckpt_dir, finetune=finetune, model_checkpoint=model_checkpoint,
        dropout=dropout, best=use_best, device=args.device,
    )
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {num_params:,} parameters")

    # ── 6. Load tokenizer and dev data ───────────────────────────────────────
    tokenizer = _TOKENIZER

    input_prefix = cfg.get("input_prefix", "")
    include_schema = cfg.get("include_schema", False)
    dev_loader = get_dataloader(args.batch_size, "dev", input_prefix=input_prefix,
                                include_schema=include_schema)

    # Load raw NL and SQL for display
    with open("data/dev.nl") as f:
        dev_nl = [line.strip() for line in f.readlines()]
    with open("data/dev.sql") as f:
        dev_sql = [line.strip() for line in f.readlines()]

    print(f"\nDev set: {len(dev_nl)} examples")

    # ── 7. Run inference on full dev set ─────────────────────────────────────
    print(f"\nRunning inference on dev set (beam={cfg.get('num_beams', 1)}, "
          f"max_tokens={cfg.get('max_new_tokens', 256)})...")

    max_new_tokens = cfg.get("max_new_tokens", 256)
    num_beams = cfg.get("num_beams", 1)
    all_preds = []

    with torch.no_grad():
        for batch in dev_loader:
            encoder_input = batch[0].to(args.device)
            encoder_mask = batch[1].to(args.device)

            outputs = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_preds.extend(preds)

    # ── 8. Compute metrics ───────────────────────────────────────────────────
    tmp_pred_sql = os.path.join(run_dir, "_eval_pred.sql")
    tmp_pred_pkl = os.path.join(run_dir, "_eval_pred.pkl")
    save_queries_and_records(all_preds, tmp_pred_sql, tmp_pred_pkl)

    sql_em, record_em, record_f1, error_msgs = compute_metrics(
        "data/dev.sql", tmp_pred_sql,
        "records/ground_truth_dev.pkl", tmp_pred_pkl,
    )

    print(f"\n{'='*70}")
    print(f"  DEV SET EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"  Record F1:    {record_f1:.4f}")
    print(f"  Record EM:    {record_em:.4f}")
    print(f"  SQL EM:       {sql_em:.4f}")
    num_errors = sum(1 for m in error_msgs if m)
    print(f"  Error rate:   {num_errors}/{len(error_msgs)} ({num_errors/len(error_msgs)*100:.1f}%)")
    print(f"{'='*70}")

    # ── 9. Show example predictions ──────────────────────────────────────────
    n = min(args.num_examples, len(dev_nl))
    print(f"\n{'='*70}")
    print(f"  EXAMPLE PREDICTIONS ({n} examples)")
    print(f"{'='*70}")

    for i in range(n):
        nl = dev_nl[i]
        gt = dev_sql[i]
        pred = all_preds[i]
        err = error_msgs[i] if i < len(error_msgs) else ""

        # Execute both queries to compare records
        gt_records, gt_err = execute_sql(gt)
        pred_records, pred_err = execute_sql(pred)

        match_sql = "EXACT MATCH" if gt == pred else "mismatch"
        match_rec = "MATCH" if set(gt_records) == set(pred_records) else "MISMATCH"

        print(f"\n--- Example {i+1} ---")
        print(f"  NL:   {nl}")
        print(f"  GT:   {gt[:120]}{'...' if len(gt) > 120 else ''}")
        print(f"  PRED: {pred[:120]}{'...' if len(pred) > 120 else ''}")
        print(f"  SQL:  [{match_sql}]")
        print(f"  Records: GT={len(gt_records)} rows, Pred={len(pred_records)} rows [{match_rec}]")
        if pred_err:
            print(f"  Pred ERROR: {pred_err}")
        if err:
            print(f"  Eval ERROR: {err}")

    # ── 10. Error analysis summary ───────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  ERROR ANALYSIS")
    print(f"{'='*70}")

    exact_sql_matches = sum(1 for gt, pred in zip(dev_sql, all_preds) if gt == pred)
    print(f"  Exact SQL matches: {exact_sql_matches}/{len(dev_sql)}")

    error_types = {}
    for msg in error_msgs:
        if msg:
            etype = msg.split(":")[0] if ":" in msg else msg
            error_types[etype] = error_types.get(etype, 0) + 1

    if error_types:
        print(f"\n  SQL error breakdown:")
        for etype, count in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"    {etype}: {count}")

    # Cleanup temp files
    for f in [tmp_pred_sql, tmp_pred_pkl]:
        Path(f).unlink(missing_ok=True)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
