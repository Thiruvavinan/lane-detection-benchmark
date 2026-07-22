#!/usr/bin/env python
"""
scripts/evaluate.py
-------------------
Run the target dataset's own official metric on a trained checkpoint.

Usage
-----
    python scripts/evaluate.py --config configs/unet_tusimple.yaml
    python scripts/evaluate.py --config configs/deeplab_tusimple.yaml \\
                               --checkpoint runs/deeplab_tusimple/best.pth

The model's output is already in the target dataset's own representation
(see data/datasets/base.py, models/base.py) — there is no mask, and no
conversion step between prediction and metric. Output: the official
metric (global + per-scenario, when the dataset provides scenario
labels), printed to stdout.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml
from torch.utils.data import DataLoader

from data.datasets import DATASETS, build_dataset
from models import build_model
from evaluation.tusimple_metrics import TuSimpleEvaluator, format_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    exist_threshold = cfg.get("evaluation", {}).get("threshold", 0.5)
    output_dir = Path(cfg["experiment"]["output_dir"])

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    ds_cfg = cfg["dataset"]
    ds_cls = DATASETS[ds_cfg["name"]]
    test_ds = build_dataset(
        ds_cfg["name"],
        root=ds_cfg["root"],
        split="test",
        image_size=tuple(ds_cfg["image_size"]),
        augment=False,
    )
    dl_cfg = cfg["dataloader"]
    test_loader = DataLoader(
        test_ds,
        batch_size=dl_cfg["batch_size"],
        shuffle=False,
        num_workers=dl_cfg.get("num_workers", 4),
        collate_fn=test_ds.collate_fn,
    )

    # ------------------------------------------------------------------
    # Model — output shape is dataset-defined, injected here, not in the config
    # ------------------------------------------------------------------
    model_cfg = dict(cfg["model"])
    model_name = model_cfg.pop("name")
    model = build_model(
        model_name,
        max_lanes=ds_cls.MAX_LANES,
        num_rows=ds_cls.NUM_ROWS,
        orig_width=ds_cls.ORIG_WIDTH,
        **model_cfg,
    )

    ckpt_path = args.checkpoint or str(output_dir / "best.pth")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    evaluator = TuSimpleEvaluator()
    h_samples = list(ds_cls.H_SAMPLES)

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            targets = batch["target"]  # [B, MAX_LANES, NUM_ROWS], ground truth on canonical grid
            meta = batch["meta"]

            pred = model(images).cpu()          # [B, 2, MAX_LANES, NUM_ROWS]
            pred_coord = pred[:, 0]
            pred_exists = torch.sigmoid(pred[:, 1]) > exist_threshold

            for i in range(images.shape[0]):
                pred_lanes = _lanes_from_tensor(pred_coord[i], pred_exists[i])
                gt_lanes = _lanes_from_target(targets[i])
                evaluator.update(pred_lanes, gt_lanes, h_samples, meta=meta[i])

    results = evaluator.compute()
    print("\n" + format_results(results))


def _lanes_from_tensor(coords: torch.Tensor, exists: torch.Tensor):
    """[MAX_LANES, NUM_ROWS] coord + bool exist tensors -> TuSimple's list-of-lists format."""
    return [
        [float(coords[slot, r]) if exists[slot, r] else -2.0 for r in range(coords.shape[1])]
        for slot in range(coords.shape[0])
    ]


def _lanes_from_target(target: torch.Tensor):
    """Ground-truth [MAX_LANES, NUM_ROWS] tensor -> TuSimple's list-of-lists format."""
    return [[x if x >= 0 else -2.0 for x in row] for row in target.tolist()]


if __name__ == "__main__":
    main()
