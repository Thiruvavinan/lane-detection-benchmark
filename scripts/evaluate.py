#!/usr/bin/env python
"""
scripts/evaluate.py
-------------------
Run accuracy evaluation on a trained checkpoint.

Usage
-----
    python scripts/evaluate.py --config configs/unet_tusimple.yaml
    python scripts/evaluate.py --config configs/deeplab_tusimple.yaml \\
                               --checkpoint runs/deeplab_tusimple/best.pth

Outputs:
  - Accuracy metrics (global + per-scenario) printed to stdout
  - Failure case grids saved to <output_dir>/failure_cases/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml
from torch.utils.data import DataLoader

from data.datasets import build_dataset
from models import build_model
from evaluation.metrics import SegmentationEvaluator, format_results
from visualization.failure_cases import FailureCaseExporter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-worst", type=int, default=10, help="Failure cases to export per scenario")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    threshold = cfg.get("evaluation", {}).get("threshold", 0.5)
    output_dir = Path(cfg["experiment"]["output_dir"])

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    ds_cfg = cfg["dataset"]
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
    # Model
    # ------------------------------------------------------------------
    model_cfg = dict(cfg["model"])
    model_name = model_cfg.pop("name")
    model = build_model(model_name, **model_cfg)

    ckpt_path = args.checkpoint or str(output_dir / "best.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    evaluator = SegmentationEvaluator(threshold=threshold)
    exporter = FailureCaseExporter(out_dir=str(output_dir / "failure_cases"), threshold=threshold)

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            masks = batch["mask"]
            meta = batch["meta"]

            logits = model(images)
            evaluator.update(logits.cpu(), masks, meta)
            exporter.update(batch["image"], masks, logits.cpu(), meta)

    results = evaluator.compute()
    print("\n" + format_results(results))
    exporter.save(n_worst=args.n_worst)


if __name__ == "__main__":
    main()
