"""Part 1 training: train loop, eval, test inference for T5 fine-tune."""

import argparse
import gc
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from part1.data import PAD_IDX, _TOKENIZER, load_t5_data
from part1.model import (
    initialize_model,
    load_model_from_checkpoint,
    load_training_state,
    save_model,
    save_training_state,
)
from src.mlflow_utils import (
    end_mlflow_run,
    log_epoch_metrics,
    log_extra_params,
    setup_run,
)
from src.utils.system_metrics import collect_system_metrics
from t5_utils import initialize_optimizer_and_scheduler
from utils import compute_metrics, save_queries_and_records, set_random_seeds

LOSS_FNS = {
    "cross_entropy": nn.CrossEntropyLoss,
}


# ── Shared generation helper ────────────────────────────────────────────

def _generate_predictions(model, loader, max_new_tokens, num_beams, device):
    """Run model.generate on every batch; return list of decoded strings."""
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(loader):
            encoder_input = batch[0].to(device)
            encoder_mask = batch[1].to(device)
            outputs = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            preds = _TOKENIZER.batch_decode(outputs, skip_special_tokens=True)
            all_preds.extend(preds)
    return all_preds


# ── Training ────────────────────────────────────────────────────────────────

def train(cfg, model, train_loader, dev_loader, optimizer, scheduler, run_dir,
          start_epoch=0, best_val=None, epochs_since_improvement=0, mlflow_run_id=None):
    if best_val is None:
        best_val = -1 if cfg.checkpointing.mode == "max" else float("inf")
    best_metrics = {}
    ckpt_dir = str(run_dir / "checkpoints")
    train_start = time.time()
    epoch_times = []
    device = cfg.device

    gt_sql_path = "data/dev.sql"
    gt_record_path = "records/ground_truth_dev.pkl"
    model_sql_path = str(run_dir / "dev_pred.sql")
    model_record_path = str(run_dir / "dev_pred.pkl")

    global_step = start_epoch * len(train_loader)
    _interrupted = False
    _pred_cache = {}
    try:
      for epoch in range(start_epoch, cfg.num_epochs):
        epoch_start = time.time()
        train_t0 = time.time()
        tr_loss, avg_grad_norm, train_tokens, global_step = train_epoch(
            cfg, model, train_loader, optimizer, scheduler, device, global_step
        )
        train_epoch_seconds = time.time() - train_t0
        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(
            cfg, model, dev_loader, gt_sql_path, model_sql_path, gt_record_path, model_record_path, device,
            pred_cache=_pred_cache,
        )
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        wall_clock = time.time() - train_start

        print(f"Epoch {epoch}: train loss = {tr_loss:.4f}")
        print(f"Epoch {epoch}: dev loss = {eval_loss:.4f}, F1 = {record_f1:.4f}, "
              f"EM = {record_em:.4f}, SQL_EM = {sql_em:.4f}, err = {error_rate*100:.1f}%")

        # ── MLflow epoch metrics ──
        log_epoch_metrics({
            "epoch": epoch,
            "train_loss": tr_loss,
            "dev_loss": eval_loss,
            "record_f1": record_f1,
            "record_em": record_em,
            "sql_em": sql_em,
            "error_rate": error_rate,
            "gradient_norm": avg_grad_norm,
            "lr": optimizer.param_groups[0]["lr"],
        }, step=epoch)

        log_epoch_metrics({
            "timing/epoch_seconds": epoch_time,
            "timing/wall_clock_seconds": wall_clock,
            "timing/train_epoch_seconds": train_epoch_seconds,
            "timing/train_tokens_per_sec": train_tokens / train_epoch_seconds if train_epoch_seconds > 0 else 0,
        }, step=epoch)

        # ── Check improvement (single computation) ──
        improved = (record_f1 > best_val) if cfg.checkpointing.mode == "max" else (record_f1 < best_val)

        log_epoch_metrics({
            "tracking/best_record_f1": record_f1 if improved else best_val,
            "tracking/epochs_since_improvement": 0 if improved else epochs_since_improvement + 1,
        }, step=epoch)

        if cfg.log_system_metrics:
            system = collect_system_metrics(device)
            log_epoch_metrics({f"system/{k}": v for k, v in system.items()}, step=epoch)

        # ── Checkpointing ──
        if improved:
            best_val = record_f1
            best_metrics = {
                "record_f1": record_f1, "record_em": record_em,
                "sql_em": sql_em, "error_rate": error_rate,
            }
            epochs_since_improvement = 0
            if cfg.checkpointing.enabled and cfg.checkpointing.save_best:
                save_model(ckpt_dir, model, best=True,
                           best_filename=cfg.checkpointing.best_filename,
                           last_filename=cfg.checkpointing.last_filename)
        else:
            epochs_since_improvement += 1

        if cfg.checkpointing.enabled and cfg.checkpointing.save_every_n > 0 and (epoch + 1) % cfg.checkpointing.save_every_n == 0:
            save_model(ckpt_dir, model, best=False,
                       last_filename=f"model_epoch_{epoch}.pt")

        # Save full training state for resume
        save_training_state(ckpt_dir, model, optimizer, scheduler,
                            epoch + 1, best_val, epochs_since_improvement,
                            mlflow_run_id=mlflow_run_id)

        if cfg.patience_epochs > 0 and epochs_since_improvement >= cfg.patience_epochs:
            print(f"Early stopping at epoch {epoch} (patience={cfg.patience_epochs})")
            break

        if cfg.max_wall_clock_hours and wall_clock >= cfg.max_wall_clock_hours * 3600:
            print(f"Time budget reached ({wall_clock/3600:.2f}h / {cfg.max_wall_clock_hours:.2f}h). Stopping after epoch {epoch}.")
            break

    except KeyboardInterrupt:
        _interrupted = True
        print(f"\nInterrupted at epoch {epoch}. Saving training state...")
        save_training_state(ckpt_dir, model, optimizer, scheduler,
                            epoch, best_val, epochs_since_improvement,
                            mlflow_run_id=mlflow_run_id)
        print(f"State saved to {ckpt_dir}. Resume with --resume {run_dir}")

    if epoch_times:
        log_extra_params({"avg_epoch_seconds": round(sum(epoch_times) / len(epoch_times), 2)})

    return best_val, _interrupted


def train_epoch(cfg, model, train_loader, optimizer, scheduler, device, global_step=0):
    model.train()
    total_loss = 0
    total_tokens = 0
    total_grad_norm = 0.0
    num_batches = 0
    criterion = LOSS_FNS[cfg.loss_fn]()

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(device)
        encoder_mask = encoder_mask.to(device)
        decoder_input = decoder_input.to(device)
        decoder_targets = decoder_targets.to(device)

        logits = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
        )["logits"]

        non_pad = decoder_targets != PAD_IDX
        loss = criterion(logits[non_pad], decoder_targets[non_pad])
        loss.backward()

        # clip_grad_norm_ returns the total (unclipped) gradient norm
        clip_val = cfg.grad_clip_norm if cfg.grad_clip_norm is not None else float("inf")
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), clip_val).item()
        total_grad_norm += grad_norm
        num_batches += 1

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

        log_epoch_metrics({
            "batch/loss": loss.item(),
            "batch/gradient_norm": grad_norm,
            "batch/lr": optimizer.param_groups[0]["lr"],
        }, step=global_step)
        global_step += 1

    avg_loss = total_loss / total_tokens
    avg_grad_norm = total_grad_norm / num_batches
    return avg_loss, avg_grad_norm, total_tokens, global_step


# ── Evaluation ──────────────────────────────────────────────────────────

def eval_epoch(cfg, model, dev_loader, gt_sql_path, model_sql_path, gt_record_path, model_record_path, device, pred_cache=None):
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = LOSS_FNS[cfg.loss_fn]()

    # Compute loss over dev set
    with torch.no_grad():
        for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(dev_loader):
            encoder_input = encoder_input.to(device)
            encoder_mask = encoder_mask.to(device)
            decoder_input = decoder_input.to(device)
            decoder_targets = decoder_targets.to(device)

            logits = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            )["logits"]

            non_pad = decoder_targets != PAD_IDX
            loss = criterion(logits[non_pad], decoder_targets[non_pad])
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    eval_loss = total_loss / total_tokens

    # Generate predictions using shared helper
    all_preds = _generate_predictions(model, dev_loader, cfg.max_new_tokens, cfg.num_beams, device)

    # Cache check: skip SQL execution if predictions are identical to last epoch
    preds_key = hash(tuple(all_preds))
    if pred_cache is not None and pred_cache.get("key") == preds_key:
        print("Predictions unchanged — reusing cached SQL metrics")
        sql_em, record_em, record_f1, error_msgs = pred_cache["metrics"]
    else:
        save_queries_and_records(all_preds, model_sql_path, model_record_path,
                                 num_threads=cfg.sql_num_threads)
        sql_em, record_em, record_f1, error_msgs = compute_metrics(
            gt_sql_path, model_sql_path, gt_record_path, model_record_path
        )
        if pred_cache is not None:
            pred_cache["key"] = preds_key
            pred_cache["metrics"] = (sql_em, record_em, record_f1, error_msgs)

    error_rate = sum(1 for m in error_msgs if m) / len(error_msgs) if error_msgs else 0

    return eval_loss, record_f1, record_em, sql_em, error_rate


# ── Test inference ──────────────────────────────────────────────────────

def test_inference(cfg, model, test_loader, model_sql_path, model_record_path, device):
    model.eval()
    all_preds = _generate_predictions(model, test_loader, cfg.max_new_tokens, cfg.num_beams, device)
    save_queries_and_records(all_preds, model_sql_path, model_record_path)
    print(f"Test predictions saved to {model_sql_path}")


# ── Entry point ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Part 1: T5 fine-tune training")

    # ── Config variant (class name in part1.config) ──
    parser.add_argument("--config", type=str, default="T5FineTuneConfig",
                        help="Config class name in part1.config (e.g. 'T5FineTuneConfig_freeze_encoder')")

    # ── Training hyperparameters ──
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--test_batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--scheduler", type=str, default=None, choices=["cosine", "linear", "none"])
    parser.add_argument("--patience_epochs", type=int, default=None)
    parser.add_argument("--optimizer", type=str, default=None, choices=["AdamW"])
    parser.add_argument("--num_warmup_epochs", type=int, default=None)
    parser.add_argument("--grad_clip_norm", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)

    # ── Layer freezing ──
    parser.add_argument("--freeze_encoder", action="store_true", default=None)
    parser.add_argument("--freeze_embeddings", action="store_true", default=None)
    parser.add_argument("--unfreeze_last_n_decoder", type=int, default=None)

    # ── Input formatting ──
    parser.add_argument("--input_prefix", type=str, default=None)
    parser.add_argument("--include_schema", action="store_true", default=None)

    # ── Resume / time budget ──
    parser.add_argument("--resume", type=str, default=None, help="Run dir to resume from")
    parser.add_argument("--max_time", type=float, default=None, help="Max wall clock hours")

    # ── Decoding ──
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--num_beams", type=int, default=None)

    # ── Misc ──
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--name", type=str, default=None)

    return parser.parse_args()


_CLI_TO_CFG = {
    "num_epochs": "num_epochs",
    "batch_size": "batch_size",
    "test_batch_size": "test_batch_size",
    "learning_rate": "learning_rate",
    "weight_decay": "weight_decay",
    "scheduler": "scheduler",
    "patience_epochs": "patience_epochs",
    "optimizer": "optimizer",
    "num_warmup_epochs": "num_warmup_epochs",
    "grad_clip_norm": "grad_clip_norm",
    "dropout": "dropout",
    "freeze_encoder": "freeze_encoder",
    "freeze_embeddings": "freeze_embeddings",
    "unfreeze_last_n_decoder": "unfreeze_last_n_decoder",
    "input_prefix": "input_prefix",
    "include_schema": "include_schema",
    "resume": "resume_run_dir",
    "max_time": "max_wall_clock_hours",
    "max_new_tokens": "max_new_tokens",
    "num_beams": "num_beams",
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
    """Look up a config class by name in part1.config and return an instance."""
    import part1.config as cfg_mod
    cls = getattr(cfg_mod, class_name, None)
    if cls is None:
        available = [n for n in dir(cfg_mod) if not n.startswith("_") and isinstance(getattr(cfg_mod, n), type)]
        raise ValueError(f"Unknown config class '{class_name}'. Available: {available}")
    return cls()


def main():
    cli = parse_args()
    cfg = load_config(cli.config)
    apply_cli_overrides(cfg, cli)
    set_random_seeds(cfg.seed)
    device = cfg.device

    # Data
    train_loader, dev_loader, test_loader = load_t5_data(
        cfg.batch_size, cfg.test_batch_size,
        input_prefix=cfg.input_prefix, include_schema=cfg.include_schema,
    )

    # Model
    model = initialize_model(
        finetune=cfg.finetune,
        model_checkpoint=cfg.model_checkpoint,
        dropout=cfg.dropout,
        freeze_encoder=cfg.freeze_encoder,
        freeze_embeddings=cfg.freeze_embeddings,
        unfreeze_last_n_decoder=cfg.unfreeze_last_n_decoder,
        device=device,
    )

    # Optimizer & scheduler
    args = argparse.Namespace(
        optimizer_type=cfg.optimizer,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        scheduler_type=cfg.scheduler or "none",
        num_warmup_epochs=cfg.num_warmup_epochs,
        max_n_epochs=cfg.num_epochs,
    )
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    # Resume: load training state first to recover MLflow run ID
    start_epoch, best_val, epochs_since_imp = 0, None, 0
    resume_mlflow_run_id = None
    if cfg.resume_run_dir:
        resume_ckpt = str(Path(cfg.resume_run_dir) / "checkpoints")
        start_epoch, best_val, epochs_since_imp, resume_mlflow_run_id = load_training_state(
            resume_ckpt, model, optimizer, scheduler, device
        )
        print(f"Resumed from epoch {start_epoch}, best_val={best_val:.4f}")
        if resume_mlflow_run_id:
            print(f"Resuming MLflow run {resume_mlflow_run_id}")

    # Single setup: creates run directory + starts MLflow run
    run_dir, mlflow_run_id = setup_run(
        cfg, experiment_name="part1_t5_finetune", resume_run_id=resume_mlflow_run_id,
    )
    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")

    # Log one-time model & data params (skip on resume — already logged)
    if not cfg.resume_run_dir:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        extra_params = {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "num_train_samples": len(train_loader.dataset),
            "num_dev_samples": len(dev_loader.dataset),
        }
        if torch.cuda.is_available():
            extra_params["gpu_name"] = torch.cuda.get_device_name()
        log_extra_params(extra_params)

    # Train
    _, interrupted = train(cfg, model, train_loader, dev_loader, optimizer, scheduler, run_dir,
                           start_epoch=start_epoch, best_val=best_val, epochs_since_improvement=epochs_since_imp,
                           mlflow_run_id=mlflow_run_id)

    if interrupted:
        end_mlflow_run()
        print("Training was interrupted. Skipping final eval and test inference.")
        return

    # Free training-only objects to reclaim VRAM before loading checkpoint
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()

    # Load best and run final eval + test
    ckpt_dir = str(run_dir / "checkpoints")
    model = load_model_from_checkpoint(
        ckpt_dir, finetune=cfg.finetune, model_checkpoint=cfg.model_checkpoint,
        dropout=cfg.dropout, best=True, device=device,
        best_filename=cfg.checkpointing.best_filename,
        last_filename=cfg.checkpointing.last_filename,
    )
    model.eval()

    # Final dev eval
    _, f1, em, sql_em, err = eval_epoch(
        cfg, model, dev_loader, "data/dev.sql", "results/t5_ft_dev.sql",
        "records/ground_truth_dev.pkl", "records/t5_ft_dev.pkl", device,
    )
    print(f"Final dev: F1={f1:.4f}, EM={em:.4f}, SQL_EM={sql_em:.4f}, err={err*100:.1f}%")

    # Test
    test_inference(cfg, model, test_loader, "results/t5_ft_test.sql", "records/t5_ft_test.pkl", device)
    end_mlflow_run()

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
