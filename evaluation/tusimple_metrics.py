"""
evaluation/tusimple_metrics.py
-------------------------------
Official TuSimple lane-detection evaluation metric, ported to Python 3.

TuSimple does not annotate lanes as pixel masks. Each label is a sparse
set of keypoints: one x position per fixed y row ("h_samples"), with -2
where a lane has no point at that row. `metrics.py` measures pixel-level
IoU/F1 on masks, which is strict and not comparable to published TuSimple
leaderboard numbers. This file ports the official scoring logic from
https://github.com/TuSimple/tusimple-benchmark/blob/master/evaluate/lane.py
(originally Python 2 + ujson) unchanged, so results are directly
comparable to published numbers for other architectures.

What the official algorithm does (kept as-is, not simplified):
  - Per ground-truth lane, the matching pixel tolerance is the 20px
    threshold divided by cos(angle), where `angle` is the slope of a
    linear fit through that lane's points — steeper (more vertical)
    lanes get a tighter effective tolerance in x.
  - A predicted lane matches a ground-truth lane if their point-wise
    accuracy under that tolerance is >= 0.85 (`pt_thresh`).
  - An image's accuracy is the mean of its ground-truth lanes' best
    match accuracy, capped to at most 4 lanes (with more than 4 lanes,
    the single worst-scoring lane is dropped before averaging).
  - FP is unmatched predicted lanes / number of predicted lanes.
  - FN is unmatched ground-truth lanes / min(len(gt), 4), with one
    "free" miss forgiven when there are more than 4 gt lanes.
  - A submission exceeding the 200ms runtime budget, or predicting more
    than (num gt lanes + 2) lanes, is scored as a zero for that image.

Final Accuracy/FP/FN are the mean of these per-image values, exactly as
the official `bench_one_submit` computes them.

Usage
-----
    evaluator = TuSimpleEvaluator()
    for sample in test_set:
        evaluator.update(pred_lanes, sample["lanes"], sample["h_samples"])
    results = evaluator.compute()
    print(format_results(results))
"""

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import ujson as json  # noqa: F401  (kept for parity with the official script)
except ImportError:  # pragma: no cover
    import json  # noqa: F401

from sklearn.linear_model import LinearRegression

INVALID = -2  # TuSimple's convention for "no lane point at this row"


class LaneEval(object):
    """
    Official TuSimple scoring logic (`lane.py`), adapted to Python 3.

    The only changes from the original are Python 2 -> 3 fixes (no
    behavioral changes): the angle-adjusted threshold, the 0.85 matching
    criterion, the 4-lane cap, and the runtime/lane-count zero-score
    guard are all unchanged.
    """

    lr = LinearRegression()
    pixel_thresh = 20
    pt_thresh = 0.85

    @staticmethod
    def get_angle(xs, y_samples):
        xs, ys = xs[xs >= 0], y_samples[xs >= 0]
        if len(xs) > 1:
            LaneEval.lr.fit(ys[:, None], xs)
            k = LaneEval.lr.coef_[0]
            theta = np.arctan(k)
        else:
            theta = 0
        return theta

    @staticmethod
    def line_accuracy(pred, gt, thresh):
        pred = np.array([p if p >= 0 else -100 for p in pred])
        gt = np.array([g if g >= 0 else -100 for g in gt])
        return np.sum(np.where(np.abs(pred - gt) < thresh, 1.0, 0.0)) / len(gt)

    @staticmethod
    def bench(pred, gt, y_samples, running_time):
        if any(len(p) != len(y_samples) for p in pred):
            raise Exception("Format of lanes error.")
        if running_time > 200 or len(gt) + 2 < len(pred):
            return 0.0, 0.0, 1.0
        angles = [
            LaneEval.get_angle(np.array(x_gts), np.array(y_samples)) for x_gts in gt
        ]
        threshs = [LaneEval.pixel_thresh / np.cos(angle) for angle in angles]
        line_accs = []
        fn = 0.0
        matched = 0.0
        for x_gts, thresh in zip(gt, threshs):
            accs = [
                LaneEval.line_accuracy(np.array(x_preds), np.array(x_gts), thresh)
                for x_preds in pred
            ]
            max_acc = np.max(accs) if len(accs) > 0 else 0.0
            if max_acc < LaneEval.pt_thresh:
                fn += 1
            else:
                matched += 1
            line_accs.append(max_acc)
        fp = len(pred) - matched
        if len(gt) > 4 and fn > 0:
            fn -= 1
        s = sum(line_accs)
        if len(gt) > 4:
            s -= min(line_accs)
        return (
            s / max(min(4.0, len(gt)), 1.0),
            fp / len(pred) if len(pred) > 0 else 0.0,
            fn / max(min(len(gt), 4.0), 1.0),
        )


class TuSimpleEvaluator:
    """
    Accumulates the official per-image (accuracy, FP, FN) scores and
    averages them at the end, exactly as TuSimple's `bench_one_submit`
    does.
    """

    def __init__(self):
        self._reset()

    def _reset(self):
        # Global accumulators
        self._sum_acc = 0.0
        self._sum_fp = 0.0
        self._sum_fn = 0.0
        self._count = 0
        # Per-scenario accumulators
        self._scenario: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"acc": 0.0, "fp": 0.0, "fn": 0.0, "count": 0}
        )

    def update(
        self,
        pred_lanes: Sequence[Sequence[float]],
        gt_lanes: Sequence[Sequence[float]],
        h_samples: Sequence[float],
        run_time: float = 0.0,
        meta: Optional[dict] = None,
    ):
        """
        Score one image with the official `LaneEval.bench`.

        Parameters
        ----------
        pred_lanes : list of lanes, each a list of x per row in h_samples
        gt_lanes   : same format, from the JSON annotation's "lanes" field
        h_samples  : the shared y rows, from the JSON annotation's
                     "h_samples" field
        run_time   : inference time in ms; predictions over the official
                     200ms budget are scored as a zero for this image
        meta       : optional dict; if meta["scenario"] is set, the score
                     is also accumulated per scenario
        """
        acc, fp, fn = LaneEval.bench(pred_lanes, gt_lanes, h_samples, run_time)

        self._sum_acc += acc
        self._sum_fp += fp
        self._sum_fn += fn
        self._count += 1

        if meta is not None:
            scenario = meta.get("scenario")
            if scenario is not None:
                s = self._scenario[scenario]
                s["acc"] += acc
                s["fp"] += fp
                s["fn"] += fn
                s["count"] += 1

    def compute(self) -> dict:
        global_metrics = _average(self._sum_acc, self._sum_fp, self._sum_fn, self._count)

        per_scenario = {}
        for scenario, s in self._scenario.items():
            per_scenario[scenario] = _average(s["acc"], s["fp"], s["fn"], s["count"])

        return {
            "global": global_metrics,
            "per_scenario": per_scenario,
        }

    def reset(self):
        self._reset()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _average(sum_acc: float, sum_fp: float, sum_fn: float, count: int) -> dict:
    eps = 1e-7
    n = count if count > 0 else eps
    return {
        "accuracy": round(sum_acc / n, 4),
        "fp": round(sum_fp / n, 4),
        "fn": round(sum_fn / n, 4),
    }


def mask_to_lanes(
    mask: np.ndarray,
    h_samples: Sequence[float],
    mask_size: Tuple[int, int],
    orig_size: Tuple[int, int],
    max_gap_px: float = 40.0,
) -> List[List[float]]:
    """
    Extract lane x-coordinates from a dense binary mask, in the same
    per-row format as TuSimple's keypoint annotations, so predictions
    from a dense-mask model (every architecture in this benchmark) can
    be scored with the official point-level metric above.

    Our models have no notion of separate lane instances -- just one
    binary mask -- so this is a heuristic, not a learned prediction:
      1. For each h_samples row, scale it into mask coordinates and find
         contiguous runs of foreground pixels; each run's center is one
         candidate point, scaled back to original-image x.
      2. Walk rows top to bottom, greedily extending each open lane
         track with whichever unused candidate is closest in x (within
         `max_gap_px`); unmatched candidates start new tracks. A row
         with no nearby candidate leaves that track's point as -2
         ("no lane here"), matching TuSimple's own convention.

    Good enough for comparing architectures that all share this dense
    mask output -- not a substitute for a real instance-aware decoder.

    Parameters
    ----------
    mask       : [H, W] binary array, at `mask_size` resolution
    h_samples  : y rows to sample, in ORIGINAL image coordinates
    mask_size  : (H, W) of `mask`
    orig_size  : (H, W) of the original image the annotation was made on
    """
    mask_h, mask_w = mask_size
    orig_h, orig_w = orig_size
    scale_y = mask_h / orig_h
    scale_x_inv = orig_w / mask_w

    # 1. Per-row candidate x positions, in original-image coordinates.
    row_candidates: List[List[float]] = []
    for y in h_samples:
        y_mask = min(max(int(round(y * scale_y)), 0), mask_h - 1)
        row = mask[y_mask]
        candidates = []
        x = 0
        while x < mask_w:
            if row[x]:
                start = x
                while x < mask_w and row[x]:
                    x += 1
                candidates.append((start + x - 1) / 2.0 * scale_x_inv)
            else:
                x += 1
        row_candidates.append(candidates)

    # 2. Greedily link candidates across rows into lane tracks.
    tracks: List[List[float]] = []
    last_x: List[float] = []

    for row_idx, candidates in enumerate(row_candidates):
        used = [False] * len(candidates)

        for t in range(len(tracks)):
            best_j, best_dist = -1, max_gap_px
            for j, cx in enumerate(candidates):
                if used[j]:
                    continue
                dist = abs(cx - last_x[t])
                if dist < best_dist:
                    best_j, best_dist = j, dist
            if best_j >= 0:
                tracks[t][row_idx] = candidates[best_j]
                last_x[t] = candidates[best_j]
                used[best_j] = True

        for j, cx in enumerate(candidates):
            if used[j]:
                continue
            new_track = [INVALID] * len(h_samples)
            new_track[row_idx] = cx
            tracks.append(new_track)
            last_x.append(cx)

    return tracks


def format_results(results: dict) -> str:
    """Pretty-print evaluation results."""
    lines = []
    g = results["global"]
    lines.append("=== Global (TuSimple official) ===")
    lines.append(f"  Accuracy  : {g['accuracy']:.4f}")
    lines.append(f"  FP        : {g['fp']:.4f}")
    lines.append(f"  FN        : {g['fn']:.4f}")

    if results["per_scenario"]:
        lines.append("\n=== Per-Scenario ===")
        lines.append(f"  {'Scenario':<15} {'Acc':>6}  {'FP':>6}  {'FN':>6}")
        lines.append("  " + "-" * 38)
        for scenario, m in sorted(results["per_scenario"].items()):
            lines.append(
                f"  {scenario:<15} {m['accuracy']:>6.4f}  {m['fp']:>6.4f}  {m['fn']:>6.4f}"
            )

    return "\n".join(lines)
