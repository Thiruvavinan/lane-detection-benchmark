"""
visualization/failure_cases.py
-------------------------------
Export worst-performing samples grouped by scenario.

Produces side-by-side grids saved to:
    outputs/failure_cases/<scenario>_worst_<n>.png

Each grid row: [input image | ground truth | prediction | overlay]

Usage
-----
    exporter = FailureCaseExporter(out_dir="outputs/failure_cases")
    for batch in test_loader:
        logits = model(batch["image"].to(device))
        exporter.update(batch["image"], batch["mask"], logits.cpu(), batch["meta"])
    exporter.save(n_worst=10)
"""

import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch

from .overlay import make_comparison_grid, logits_to_mask

# Scenarios we care about
SCENARIOS = ["straight", "curve", "exit", "merge", "night", "rain", "shadow", "construction"]


class FailureCaseExporter:

    def __init__(self, out_dir: str = "outputs/failure_cases", threshold: float = 0.5):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.threshold = threshold

        # scenario → list of (iou, image_np, gt_np, pred_np)
        self._buckets: Dict[str, list] = defaultdict(list)
        # catch-all for samples with no scenario label
        self._unlabeled: list = []

    def update(
        self,
        images: torch.Tensor,   # [B, 3, H, W]  float32
        targets: torch.Tensor,  # [B, H, W]     int64
        logits: torch.Tensor,   # [B, 1, H, W]  float32
        meta: List[dict],
    ):
        B = images.shape[0]
        for i in range(B):
            img_np = _tensor_to_uint8(images[i])
            gt_np = targets[i].numpy().astype(np.uint8)
            pred_np = logits_to_mask(logits[i], self.threshold)

            iou = _binary_iou(pred_np, gt_np)
            scenario = meta[i].get("scenario")

            entry = (iou, img_np, gt_np, pred_np)
            if scenario in SCENARIOS:
                self._buckets[scenario].append(entry)
            else:
                self._unlabeled.append(entry)

    def save(self, n_worst: int = 10):
        saved = []

        buckets_to_save = dict(self._buckets)
        if self._unlabeled:
            buckets_to_save["unlabeled"] = self._unlabeled

        for scenario, entries in buckets_to_save.items():
            # Sort ascending by IoU — lowest IoU = worst predictions
            entries.sort(key=lambda e: e[0])
            worst = entries[:n_worst]

            rows = []
            for iou, img, gt, pred in worst:
                row = make_comparison_grid(img, gt, pred)
                # Add IoU label
                cv2.putText(row, f"IoU={iou:.3f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
                rows.append(row)

            if rows:
                grid = np.vstack(rows)
                out_path = self.out_dir / f"{scenario}_worst_{n_worst}.png"
                cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
                saved.append(str(out_path))

        print(f"Saved {len(saved)} failure case grids to {self.out_dir}")
        return saved


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """[3, H, W] float32 → [H, W, 3] uint8"""
    return (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)


def _binary_iou(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    inter = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return float(inter / (union + eps))
