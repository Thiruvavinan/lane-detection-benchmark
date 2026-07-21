"""
evaluation/metrics.py
---------------------
Accuracy metrics for binary lane segmentation.

Computes:
  - IoU  (Intersection over Union)
  - F1   (Dice coefficient)
  - Precision
  - Recall

And optionally a per-scenario breakdown when meta["scenario"] is set.

Usage
-----
    evaluator = SegmentationEvaluator(threshold=0.5)
    for batch in test_loader:
        logits = model(batch["image"].to(device))
        evaluator.update(logits.cpu(), batch["mask"], batch["meta"])
    results = evaluator.compute()
    print(results)
"""

from collections import defaultdict
from typing import Dict, List, Optional

import torch


class SegmentationEvaluator:
    """
    Accumulates predictions and computes metrics at the end.

    Parameters
    ----------
    threshold : float
        Sigmoid threshold for converting logits → binary predictions.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._reset()

    def _reset(self):
        # Global accumulators
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        # Per-scenario accumulators
        self._scenario: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        )

    def update(
        self,
        logits: torch.Tensor,   # [B, 1, H, W] float32
        targets: torch.Tensor,  # [B, H, W]    int64
        meta: Optional[List[dict]] = None,
    ):
        preds = (torch.sigmoid(logits).squeeze(1) > self.threshold).long()
        targets = targets.long()

        tp = ((preds == 1) & (targets == 1)).sum().item()
        fp = ((preds == 1) & (targets == 0)).sum().item()
        fn = ((preds == 0) & (targets == 1)).sum().item()
        tn = ((preds == 0) & (targets == 0)).sum().item()

        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn

        # Per-scenario
        if meta is not None:
            for i, m in enumerate(meta):
                scenario = m.get("scenario")
                if scenario is None:
                    continue
                s = self._scenario[scenario]
                p = preds[i]
                t = targets[i]
                s["tp"] += ((p == 1) & (t == 1)).sum().item()
                s["fp"] += ((p == 1) & (t == 0)).sum().item()
                s["fn"] += ((p == 0) & (t == 1)).sum().item()
                s["tn"] += ((p == 0) & (t == 0)).sum().item()

    def compute(self) -> dict:
        global_metrics = _compute_metrics(self.tp, self.fp, self.fn, self.tn)

        per_scenario = {}
        for scenario, counts in self._scenario.items():
            per_scenario[scenario] = _compute_metrics(**counts)

        return {
            "global": global_metrics,
            "per_scenario": per_scenario,
        }

    def reset(self):
        self._reset()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _compute_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return {
        "iou": round(iou, 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


def format_results(results: dict) -> str:
    """Pretty-print evaluation results."""
    lines = []
    g = results["global"]
    lines.append("=== Global ===")
    lines.append(f"  IoU       : {g['iou']:.4f}")
    lines.append(f"  F1        : {g['f1']:.4f}")
    lines.append(f"  Precision : {g['precision']:.4f}")
    lines.append(f"  Recall    : {g['recall']:.4f}")

    if results["per_scenario"]:
        lines.append("\n=== Per-Scenario ===")
        lines.append(f"  {'Scenario':<15} {'IoU':>6}  {'F1':>6}")
        lines.append("  " + "-" * 30)
        for scenario, m in sorted(results["per_scenario"].items()):
            lines.append(f"  {scenario:<15} {m['iou']:>6.4f}  {m['f1']:>6.4f}")

    return "\n".join(lines)
