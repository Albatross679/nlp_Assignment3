"""Centralized MLflow helpers for experiment tracking."""

import mlflow


def setup_mlflow(experiment_name: str, run_name: str, params_dict: dict,
                 tracking_uri: str = "sqlite:///mlflow.db",
                 run_id: str | None = None):
    """Set tracking URI, create/get experiment, start run, log all params.

    If *run_id* is given, resume that run instead of creating a new one.
    Returns the active run ID (useful for persisting across resume).
    """
    # Clear stale active runs before changing tracking URI
    while mlflow.active_run() is not None:
        try:
            mlflow.end_run()
        except Exception:
            # Run from a different/old store — force pop from internal stack
            try:
                mlflow.tracking.fluent._active_run_stack.set(None)
            except Exception:
                break
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.start_run(run_id=run_id, run_name=run_name)
    # Only log params on fresh runs (MLflow rejects duplicate param keys)
    if run_id is None:
        for k, v in params_dict.items():
            s = str(v)
            if len(s) > 500:
                s = s[:500]
            mlflow.log_param(k, s)
    return mlflow.active_run().info.run_id


def log_epoch_metrics(metrics_dict: dict, step: int):
    """Log metrics for one epoch (or any step-indexed metrics)."""
    for k, v in metrics_dict.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(k, v, step=step)


def log_extra_params(params: dict):
    """Log additional params after initial setup (e.g. model size, data counts)."""
    for k, v in params.items():
        s = str(v)
        if len(s) > 500:
            s = s[:500]
        mlflow.log_param(k, s)


def log_model_checkpoint(checkpoint_dir: str, artifact_subdir: str = "checkpoints"):
    """Log all files in checkpoint_dir as MLflow artifacts."""
    mlflow.log_artifacts(checkpoint_dir, artifact_path=artifact_subdir)


def end_mlflow_run():
    """End the active MLflow run (no-op if none active)."""
    mlflow.end_run()
