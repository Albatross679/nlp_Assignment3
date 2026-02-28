"""System & GPU metrics collection for per-epoch logging."""

from __future__ import annotations

import os
from typing import Optional

import torch


def collect_system_metrics(device: Optional[str] = None) -> dict[str, float]:
    """Gather GPU and system metrics, returning a flat dict ready for logging.

    Gracefully degrades: missing ``pynvml`` or ``psutil`` simply omit those
    metrics rather than raising.
    """
    metrics: dict[str, float] = {}

    # ── GPU memory (torch.cuda) ──────────────────────────────────────────
    if device and "cuda" in str(device) and torch.cuda.is_available():
        dev_idx = torch.cuda.current_device()
        metrics["gpu_mem_allocated_mb"] = torch.cuda.memory_allocated(dev_idx) / 1e6
        metrics["gpu_mem_reserved_mb"] = torch.cuda.memory_reserved(dev_idx) / 1e6
        metrics["gpu_mem_peak_mb"] = torch.cuda.max_memory_allocated(dev_idx) / 1e6

    # ── GPU utilization / temp / power (pynvml) ──────────────────────────
    try:
        import pynvml

        pynvml.nvmlInit()
        dev_idx = int(str(device).replace("cuda:", "").replace("cuda", "0"))
        handle = pynvml.nvmlDeviceGetHandleByIndex(dev_idx)

        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        metrics["gpu_util_pct"] = float(util.gpu)

        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        metrics["gpu_temp_c"] = float(temp)

        power = pynvml.nvmlDeviceGetPowerUsage(handle)  # milliwatts
        metrics["gpu_power_w"] = power / 1000.0

        pynvml.nvmlShutdown()
    except Exception:
        pass

    # ── CPU / RAM (psutil) ───────────────────────────────────────────────
    try:
        import psutil

        vm = psutil.virtual_memory()
        metrics["ram_used_gb"] = vm.used / 1e9
        metrics["ram_pct"] = vm.percent

        proc = psutil.Process(os.getpid())
        metrics["process_rss_mb"] = proc.memory_info().rss / 1e6

        metrics["cpu_pct"] = psutil.cpu_percent(interval=None)
    except Exception:
        pass

    return metrics
