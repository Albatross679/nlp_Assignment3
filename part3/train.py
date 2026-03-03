"""Part 3 training: LLM prompting pipeline for NL-to-SQL."""

import argparse
import os
import time

from tqdm import tqdm

from part3.data import load_prompting_data
from part3.model import initialize_model_and_tokenizer
from prompting import create_prompt, exp_kshot, eval_outputs
from prompting_utils import read_schema, extract_sql_query, save_logs
from src.mlflow_utils import (
    end_mlflow_run,
    log_epoch_metrics,
    log_extra_params,
    setup_run,
)
from utils import set_random_seeds, save_queries_and_records


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Part 3: LLM prompting for NL-to-SQL")

    # ── Config variant ──
    parser.add_argument("--config", type=str, default="PromptingConfig",
                        help="Config class name in part3.config")

    # ── Prompting settings ──
    parser.add_argument("--shot", "-s", type=int, default=None)
    parser.add_argument("--prompt_type", "-p", type=int, default=None)
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Model name: gemma or codegemma")
    parser.add_argument("--quantize", "-q", action="store_true", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)

    # ── Misc ──
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--name", type=str, default=None)

    return parser.parse_args()


_CLI_TO_CFG = {
    "shot": "shot",
    "prompt_type": "prompt_type",
    "model": "model_name",
    "quantize": "quantize",
    "max_new_tokens": "max_new_tokens",
    "seed": "seed",
    "name": "name",
}


def apply_cli_overrides(cfg, cli):
    """Apply non-None CLI arguments to the config object."""
    for cli_name, cfg_name in _CLI_TO_CFG.items():
        val = getattr(cli, cli_name)
        if val is not None:
            setattr(cfg, cfg_name, val)


def load_config(class_name):
    """Look up a config class by name in part3.config and return an instance."""
    import part3.config as cfg_mod
    cls = getattr(cfg_mod, class_name, None)
    if cls is None:
        available = [n for n in dir(cfg_mod) if not n.startswith("_") and isinstance(getattr(cfg_mod, n), type)]
        raise ValueError(f"Unknown config class '{class_name}'. Available: {available}")
    return cls()


# ── Entry point ─────────────────────────────────────────────────────────

def main():
    cli = parse_args()
    cfg = load_config(cli.config)
    apply_cli_overrides(cfg, cli)
    set_random_seeds(cfg.seed)
    device = cfg.device

    # Data
    train_x, train_y, dev_x, dev_y, test_x = load_prompting_data("data")

    # Model & tokenizer
    tokenizer, model = initialize_model_and_tokenizer(
        model_name=cfg.model_name,
        quantize=cfg.quantize,
        device=device,
    )

    # MLflow setup
    run_dir, mlflow_run_id = setup_run(cfg, experiment_name="part3_llm_prompting")
    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")
    print(f"Model: {cfg.model_name}, shot: {cfg.shot}, prompt_type: {cfg.prompt_type}")

    log_extra_params({
        "model_name": cfg.model_name,
        "shot": cfg.shot,
        "prompt_type": cfg.prompt_type,
    })

    # Evaluate each split
    for eval_split in cfg.eval_splits:
        eval_x = dev_x if eval_split == "dev" else test_x
        eval_y = dev_y if eval_split == "dev" else None

        print(f"\n{'='*70}")
        print(f"  {eval_split.upper()} split — {len(eval_x)} examples")
        print(f"{'='*70}")

        t0 = time.time()
        raw_outputs, extracted_queries = exp_kshot(tokenizer, model, eval_x, cfg.shot)
        elapsed = time.time() - t0
        print(f"Inference time: {elapsed:.1f}s ({elapsed/len(eval_x):.2f}s/example)")

        gt_sql_path = f"data/{eval_split}.sql"
        gt_record_path = f"records/ground_truth_{eval_split}.pkl"
        model_sql_path = f"results/llm_{eval_split}.sql"
        model_record_path = f"records/llm_{eval_split}.pkl"

        if eval_y is not None:
            sql_em, record_em, record_f1, model_error_msgs, error_rate = eval_outputs(
                eval_x, eval_y,
                gt_sql_path, model_sql_path,
                gt_record_path, model_record_path,
            )
            print(f"Record F1: {record_f1:.4f}, Record EM: {record_em:.4f}, SQL EM: {sql_em:.4f}")
            print(f"Error rate: {error_rate*100:.1f}%")

            log_epoch_metrics({
                f"{eval_split}/record_f1": record_f1,
                f"{eval_split}/record_em": record_em,
                f"{eval_split}/sql_em": sql_em,
                f"{eval_split}/error_rate": error_rate,
                f"{eval_split}/inference_seconds": elapsed,
            }, step=0)

            save_logs(
                str(run_dir / f"{eval_split}_log.txt"),
                sql_em, record_em, record_f1, model_error_msgs,
            )
        else:
            # Test split — save predictions only
            save_queries_and_records(extracted_queries, model_sql_path, model_record_path)
            print(f"Test predictions saved to {model_sql_path}")

    end_mlflow_run()
    print("\nDone.")


if __name__ == "__main__":
    main()
