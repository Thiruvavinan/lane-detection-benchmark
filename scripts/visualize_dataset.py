#!/usr/bin/env python
"""
scripts/visualize_dataset.py
-----------------------------
Sanity-check the dataset pipeline before training: sample a few images,
draw their ground-truth lane points (the dataset's own target
representation — see data/datasets/tusimple.py), and display them in a
matplotlib grid.

Usage
-----
    python scripts/visualize_dataset.py --root data/tusimple --split train --n 6
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from data.datasets import build_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/tusimple")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, default=6, help="Number of samples to show")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dataset = build_dataset("tusimple", root=args.root, split=args.split, augment=False)
    print(f"Loaded {len(dataset)} samples from {args.root} ({args.split} split)")

    # Target x/y are in original TuSimple pixel scale; scale to the
    # resized image shown here.
    scale_x = dataset.image_size[1] / dataset.ORIG_WIDTH
    scale_y = dataset.image_size[0] / dataset.ORIG_HEIGHT
    h_samples = dataset.H_SAMPLES

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(dataset), size=min(args.n, len(dataset)), replace=False)

    cols = min(3, len(indices))
    rows = (len(indices) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, idx in zip(axes, indices):
        sample = dataset[int(idx)]
        image = (sample["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        target = sample["target"].numpy()  # [MAX_LANES, NUM_ROWS]

        ax.imshow(image)
        for slot in range(target.shape[0]):
            xs, ys = [], []
            for r, y in enumerate(h_samples):
                x = target[slot, r]
                if x >= 0:
                    xs.append(x * scale_x)
                    ys.append(y * scale_y)
            if xs:
                ax.plot(xs, ys, marker="o", markersize=2, linewidth=1.5)

        ax.set_title(f"idx {idx}", fontsize=9)
        ax.axis("off")

    for ax in axes[len(indices):]:
        ax.axis("off")

    fig.suptitle(f"TuSimple — {args.split} split ({len(indices)} of {len(dataset)} samples)")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
