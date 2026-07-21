"""
evaluation/engineering.py
-------------------------
Engineering evaluation: latency, FPS, peak GPU memory.

This is the section that separates this benchmark from leaderboard-only repos.

Usage
-----
    from evaluation.engineering import profile_model

    report = profile_model(
        model=model,
        input_size=(1, 3, 360, 640),
        device="cuda",
        warmup=200,
        runs=500,
    )
    print(report)
"""

import time
from dataclasses import dataclass, asdict
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class EngineeringReport:
    model_name: str
    device: str
    input_size: Tuple[int, ...]
    # Runtime
    fps: float
    latency_ms_mean: float
    latency_ms_median: float
    latency_ms_p99: float
    # Memory
    peak_gpu_mb: float
    num_parameters_M: float

    def __str__(self) -> str:
        lines = [
            f"=== Engineering Report: {self.model_name} ===",
            f"  Device        : {self.device}",
            f"  Input size    : {self.input_size}",
            f"  Parameters    : {self.num_parameters_M:.1f}M",
            "",
            "  Runtime",
            f"    FPS         : {self.fps:.1f}",
            f"    Latency     : {self.latency_ms_mean:.2f} ms (mean)  "
            f"{self.latency_ms_median:.2f} ms (median)  "
            f"{self.latency_ms_p99:.2f} ms (p99)",
            "",
            "  Memory",
            f"    Peak GPU    : {self.peak_gpu_mb:.1f} MB",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)


def profile_model(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 360, 640),
    device: str = "cuda",
    warmup: int = 200,
    runs: int = 500,
    model_name: str = "model",
) -> EngineeringReport:
    """
    Profile a model's runtime and memory on a single fixed-size input.

    Parameters
    ----------
    model      : the model to profile (will be set to eval mode)
    input_size : (B, C, H, W) — use B=1 for latency measurement
    device     : "cuda" or "cpu"
    warmup     : number of warmup runs (not timed)
    runs       : number of timed runs
    model_name : label used in the report

    Returns
    -------
    EngineeringReport dataclass
    """
    model = model.to(device).eval()
    x = torch.randn(*input_size, device=device)

    # Parameter count
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    with torch.no_grad():
        _ = model(x)

    if device == "cuda":
        torch.cuda.synchronize(device)
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        peak_mb = 0.0

    # ------------------------------------------------------------------
    # Latency
    # ------------------------------------------------------------------
    latencies = []

    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize(device)

        # Timed runs
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(x)
            if device == "cuda":
                torch.cuda.synchronize(device)
            latencies.append((time.perf_counter() - t0) * 1000)  # ms

    import statistics
    latencies_sorted = sorted(latencies)
    mean_ms = statistics.mean(latencies)
    median_ms = statistics.median(latencies)
    p99_ms = latencies_sorted[int(0.99 * len(latencies_sorted))]
    fps = 1000.0 / mean_ms  # batch_size=1 assumed

    return EngineeringReport(
        model_name=model_name,
        device=device,
        input_size=input_size,
        fps=round(fps, 1),
        latency_ms_mean=round(mean_ms, 2),
        latency_ms_median=round(median_ms, 2),
        latency_ms_p99=round(p99_ms, 2),
        peak_gpu_mb=round(peak_mb, 1),
        num_parameters_M=round(n_params, 2),
    )
