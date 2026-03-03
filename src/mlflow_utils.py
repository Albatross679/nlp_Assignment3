"""Single tracking system: run directory + MLflow metrics/params in one setup call."""

from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.tracking


def setup_run(cfg, experiment_name: str,
              tracking_uri: str = "sqlite:///mlflow.db",
              resume_run_id: str | None = None) -> tuple[Path, str]:
    """Create output directory and start MLflow run in one step.

    Fresh run: creates output/{name}_{timestamp}/, saves config.json, logs params.
    Resume:    reuses cfg.resume_run_dir and the same MLflow run.
    Returns (run_dir, mlflow_run_id).
    """
    if cfg.resume_run_dir:
        run_dir = Path(cfg.resume_run_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(cfg.output.base_dir) / f"{cfg.name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        with open(run_dir / "config.json", "w") as f:
            json.dump(cfg.to_dict(), f, indent=2)

    _clear_active_runs()
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    if resume_run_id:
        mlflow.start_run(run_id=resume_run_id)
    else:
        experiment = client.get_experiment_by_name(experiment_name)
        exp_id = (experiment.experiment_id if experiment is not None
                  else client.create_experiment(experiment_name))
        mlflow.start_run(experiment_id=exp_id, run_name=cfg.name)
        for k, v in cfg.to_dict().items():
            s = str(v)
            mlflow.log_param(k, s[:500] if len(s) > 500 else s)

    return run_dir, mlflow.active_run().info.run_id


def _clear_active_runs():
    """End any stale active MLflow run before starting a new one."""
    while mlflow.active_run() is not None:
        try:
            mlflow.end_run()
        except Exception:
            try:
                mlflow.tracking.fluent._active_run_stack.set(None)
            except Exception:
                break


def log_epoch_metrics(metrics_dict: dict, step: int):
    """Log numeric metrics for one epoch (or any step-indexed point)."""
    for k, v in metrics_dict.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(k, v, step=step)


def log_extra_params(params: dict):
    """Log additional params after initial setup (e.g. model size, data counts)."""
    for k, v in params.items():
        s = str(v)
        mlflow.log_param(k, s[:500] if len(s) > 500 else s)


def end_mlflow_run():
    """End the active MLflow run."""
    mlflow.end_run()
