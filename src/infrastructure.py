"""
Infrastructure helpers: output directory, console logging, config save/load, metrics log.
"""

from __future__ import annotations
import sys
import json
from datetime import datetime
from pathlib import Path
from src.config import BaseConfig


def setup_run(cfg: BaseConfig) -> Path:
    """Create timestamped run directory with subdirs. Save config.json. Return run_dir."""
    timestamp = datetime.now().strftime(cfg.output.timestamp_format)
    run_dir = Path(cfg.output.base_dir) / f"{cfg.name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for subdir in cfg.output.subdirs.values():
        (run_dir / subdir).mkdir(exist_ok=True)

    if cfg.output.save_config:
        save_config(cfg, run_dir / "config.json")

    if cfg.console.enabled:
        _setup_console_logging(cfg, run_dir)

    metrics_logger = setup_metrics_log(cfg, run_dir)

    return run_dir, metrics_logger


class MetricsLogger:
    """Append-only JSON-lines logger. One JSON object per call to log()."""

    def __init__(self, path: Path, flush: bool = True):
        self._path = path
        self._flush = flush
        self._fh = open(path, "a")

    def log(self, metrics: dict):
        self._fh.write(json.dumps(metrics) + "\n")
        if self._flush:
            self._fh.flush()

    def close(self):
        self._fh.close()


def setup_metrics_log(cfg: BaseConfig, run_dir: Path) -> MetricsLogger | None:
    """Create a MetricsLogger if enabled, else return None."""
    if not cfg.metricslog.enabled:
        return None
    return MetricsLogger(
        path=run_dir / cfg.metricslog.filename,
        flush=cfg.metricslog.flush_every_epoch,
    )


def save_config(cfg: BaseConfig, path: str | Path):
    """Save config to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg.to_json(str(path))


def load_config(path: str | Path, config_cls: type = BaseConfig) -> BaseConfig:
    """Load config from JSON, reconstructing the correct type."""
    with open(path) as f:
        d = json.load(f)
    return config_cls.from_dict(d)


class _TeeStream:
    """Write to both a file and original stream."""

    def __init__(self, file_handle, original_stream, flush_freq=1):
        self._file = file_handle
        self._original = original_stream
        self._flush_count = 0
        self._flush_freq = flush_freq

    def write(self, data):
        self._original.write(data)
        self._file.write(data)
        self._flush_count += 1
        if self._flush_count >= self._flush_freq:
            self._file.flush()
            self._flush_count = 0

    def flush(self):
        self._original.flush()
        self._file.flush()

    def fileno(self):
        return self._original.fileno()

    def isatty(self):
        return self._original.isatty()


def _setup_console_logging(cfg: BaseConfig, run_dir: Path):
    """Redirect stdout/stderr to a log file, optionally teeing to console."""
    log_path = run_dir / cfg.console.filename
    fh = open(log_path, "w")

    if cfg.console.tee_to_console:
        sys.stdout = _TeeStream(fh, sys.stdout, cfg.console.flush_frequency)
        sys.stderr = _TeeStream(fh, sys.stderr, cfg.console.flush_frequency)
    else:
        sys.stdout = fh
        sys.stderr = fh
