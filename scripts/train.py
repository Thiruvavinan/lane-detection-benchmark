#!/usr/bin/env python
"""
scripts/train.py
----------------
Entry point for training any model in the benchmark.

Usage
-----
    python scripts/train.py --config configs/unet_tusimple.yaml
    python scripts/train.py --config configs/deeplab_tusimple.yaml
    python scripts/train.py --config configs/lanesegnet_tusimple.yaml

The only thing that changes between runs is the config file.
"""

import argparse
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml
from torch.utils.data import DataLoader

from data.datasets import DATASETS, build_dataset
from models import build_model
from training.losses import build_loss
from training.optim import build_optimizer, build_scheduler
from training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--device", default=None, help="Override device (cuda/cpu/mps)")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from <output_dir>/last.pth if it exists, instead of starting over",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    ds_cfg = cfg["dataset"]
    train_ds = build_dataset(
        ds_cfg["name"],
        root=ds_cfg["root"],
        split="train",
        image_size=tuple(ds_cfg["image_size"]),
        augment=ds_cfg.get("augment", False),
    )
    val_ds = build_dataset(
        ds_cfg["name"],
        root=ds_cfg["root"],
        split="val",
        image_size=tuple(ds_cfg["image_size"]),
        augment=False,
    )

    dl_cfg = cfg["dataloader"]
    train_loader = DataLoader(
        train_ds,
        batch_size=dl_cfg["batch_size"],
        shuffle=True,
        num_workers=dl_cfg.get("num_workers", 4),
        pin_memory=device == "cuda",
        collate_fn=train_ds.collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=dl_cfg["batch_size"],
        shuffle=False,
        num_workers=dl_cfg.get("num_workers", 4),
        pin_memory=device == "cuda",
        collate_fn=val_ds.collate_fn,
    )

    # ------------------------------------------------------------------
    # Model — output shape is dataset-defined, injected here, not in the config
    # ------------------------------------------------------------------
    ds_cls = DATASETS[ds_cfg["name"]]
    model_cfg = dict(cfg["model"])
    model_name = model_cfg.pop("name")
    model = build_model(
        model_name,
        max_lanes=ds_cls.MAX_LANES,
        num_rows=ds_cls.NUM_ROWS,
        orig_width=ds_cls.ORIG_WIDTH,
        **model_cfg,
    )
    print(f"Model: {model_name}  ({model.num_parameters()/ 1e6:.1f}M params)")

    # ------------------------------------------------------------------
    # Loss, optimizer, scheduler
    # ------------------------------------------------------------------
    loss_cfg = dict(cfg["loss"])
    loss_name = loss_cfg.pop("name")
    loss_fn = build_loss(loss_name, **loss_cfg)

    opt_cfg = dict(cfg["optimizer"])
    opt_name = opt_cfg.pop("name")
    optimizer = build_optimizer(opt_name, model.parameters(), **opt_cfg)

    sched_cfg = dict(cfg.get("scheduler", {}))
    sched_name = sched_cfg.pop("name", None)
    scheduler = build_scheduler(sched_name, optimizer, **sched_cfg) if sched_name else None

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    train_cfg = cfg["training"]
    output_dir = Path(cfg["experiment"]["output_dir"])
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=str(output_dir),
    )

    resume_from = None
    if args.resume:
        candidate = output_dir / "last.pth"
        if candidate.exists():
            resume_from = str(candidate)
        else:
            print(f"--resume passed but no checkpoint found at {candidate}; starting fresh")

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=train_cfg["epochs"],
        val_every=train_cfg.get("val_every", 1),
        early_stopping_patience=train_cfg.get("early_stopping_patience"),
        resume_from=resume_from,
    )


if __name__ == "__main__":
    main()
