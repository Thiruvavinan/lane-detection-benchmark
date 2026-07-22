#!/usr/bin/env python
"""
scripts/benchmark.py
--------------------
Engineering evaluation: FPS, latency, peak GPU memory.

This is the script that produces the engineering report — the section
that most benchmark repos skip entirely.

Usage
-----
    python scripts/benchmark.py --config configs/unet_tusimple.yaml
    python scripts/benchmark.py --config configs/unet_tusimple.yaml \\
                                --checkpoint runs/unet_tusimple/best.pth \\
                                --warmup 200 --runs 500

Output is printed to stdout and also saved as JSON to
<output_dir>/engineering_report.json so it can be aggregated into a
comparison table across architectures.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from data.datasets import DATASETS
from models import build_model
from evaluation.engineering import profile_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--runs", type=int, default=500)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg["experiment"]["output_dir"])
    image_size = tuple(cfg["dataset"]["image_size"])  # (H, W)

    # ------------------------------------------------------------------
    # Model — output shape is dataset-defined, injected here, not in the config
    # ------------------------------------------------------------------
    ds_cls = DATASETS[cfg["dataset"]["name"]]
    model_cfg = dict(cfg["model"])
    model_name = model_cfg.pop("name")
    model = build_model(
        model_name,
        max_lanes=ds_cls.MAX_LANES,
        num_rows=ds_cls.NUM_ROWS,
        orig_width=ds_cls.ORIG_WIDTH,
        **model_cfg,
    )

    if args.checkpoint or (output_dir / "best.pth").exists():
        ckpt_path = args.checkpoint or str(output_dir / "best.pth")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("No checkpoint found — profiling random-weight model.")

    # ------------------------------------------------------------------
    # Profile
    # ------------------------------------------------------------------
    input_size = (1, 3, image_size[0], image_size[1])
    report = profile_model(
        model=model,
        input_size=input_size,
        device=device,
        warmup=args.warmup,
        runs=args.runs,
        model_name=model_name,
    )

    print("\n" + str(report))

    # Save JSON
    out_path = output_dir / "engineering_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
