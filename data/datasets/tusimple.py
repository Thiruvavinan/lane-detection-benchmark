"""
data/datasets/tusimple.py
-------------------------
TuSimple lane detection dataset loader.

TuSimple provides RGB images (1280x720) and JSON keypoint annotations:
one x per fixed y row ("h_samples"), -2 where a lane has no point there.
Every model targeting this dataset outputs that same keypoint
representation directly (see MAX_LANES / H_SAMPLES below) — no dense
mask in between. See BaseDataset's docstring for why.

Directory layout expected under `root` (matches the Kaggle TuSimple
archive layout, unzipped as-is):
    root/
      train_set/
        clips/            # raw training images
        label_data_0313.json
        label_data_0531.json
        label_data_0601.json
      test_set/
        clips/            # raw test images
      test_label.json     # test annotations, at root level

Target representation
----------------------
Train and test label files annotate different y-ranges (train starts at
y=240, test at y=160, both step 10 up to y=710 — checked directly).
Every sample is re-expressed on the union: H_SAMPLES = 160..710 step 10
(56 rows). Rows outside a sample's own range are UNKNOWN, distinct from
"no lane" (INVALID) — one is a missing label, the other a real negative,
and only the latter is a valid training signal.

Lanes are sorted left-to-right (mean x over valid points) into MAX_LANES
fixed slots, so "slot 0" means the same thing across every image — a
fixed-slot regression head can't learn otherwise. MAX_LANES=5 is the
observed max across every label file (checked directly, not assumed).
"""

import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

from .base import BaseDataset


class TuSimpleDataset(BaseDataset):
    """
    Parameters
    ----------
    root : str | Path
        Root directory of the TuSimple dataset.
    split : "train" | "val" | "test"
        Which split to load.
    image_size : (H, W)
        Target image size after resizing. Default: (360, 640).
    augment : bool
        Whether to apply training-time augmentation.
    """

    TRAIN_JSONS = [
        "label_data_0313.json",
        "label_data_0531.json",
        "label_data_0601.json",
    ]
    TEST_JSON = "test_label.json"

    # --- Target representation constants (see module docstring) ---
    H_SAMPLES: Tuple[int, ...] = tuple(range(160, 711, 10))  # 56 rows
    NUM_ROWS = len(H_SAMPLES)
    MAX_LANES = 5
    ORIG_WIDTH = 1280.0   # TuSimple images are always 1280x720
    ORIG_HEIGHT = 720.0
    INVALID = -2.0   # row is annotated; this lane slot has no point here
    UNKNOWN = -3.0   # row falls outside this sample's own annotated range

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: Tuple[int, int] = (360, 640),
        augment: bool = False,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.image_size = image_size  # (H, W)
        self.augment = augment

        # Images are nested under train_set/ or test_set/ depending on split;
        # annotation "raw_file" paths are relative to that subfolder.
        self.image_root = self.root / ("train_set" if self.split in ("train", "val") else "test_set")

        self.samples = self._load_samples()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_samples(self) -> List[dict]:
        samples = []
        if self.split in ("train", "val"):
            for fname in self.TRAIN_JSONS:
                fpath = self.root / "train_set" / fname
                if not fpath.exists():
                    raise FileNotFoundError(
                        f"Missing annotation file: {fpath}\n"
                        f"Run: python scripts/download_tusimple.py --out {self.root}"
                    )
                with open(fpath) as f:
                    for line in f:
                        samples.append(json.loads(line.strip()))
            # 90 / 10 train-val split by index (deterministic)
            n = len(samples)
            if self.split == "train":
                samples = samples[: int(0.9 * n)]
            else:
                samples = samples[int(0.9 * n) :]
        else:
            fpath = self.root / self.TEST_JSON
            with open(fpath) as f:
                for line in f:
                    samples.append(json.loads(line.strip()))
        return samples

    def _row_index(self, y: int):
        """Map a sample's own h_samples row onto the canonical H_SAMPLES grid."""
        idx = (y - self.H_SAMPLES[0]) // 10
        return idx if 0 <= idx < self.NUM_ROWS else None

    def _lanes_to_target(self, lanes, h_samples) -> torch.Tensor:
        """
        Re-express this sample's lanes on the canonical H_SAMPLES grid,
        sorted left-to-right, in MAX_LANES fixed slots.

        Returns [MAX_LANES, NUM_ROWS] float32: x in original pixel scale
        where a lane has a real point, INVALID where this sample's own
        annotation says "no lane here", UNKNOWN where this row simply
        isn't part of this sample's own h_samples range at all.
        """
        target = torch.full((self.MAX_LANES, self.NUM_ROWS), self.UNKNOWN, dtype=torch.float32)

        covered_rows = [r for r in (self._row_index(y) for y in h_samples) if r is not None]
        for r in covered_rows:
            target[:, r] = self.INVALID

        def mean_x(lane):
            xs = [x for x in lane if x >= 0]
            return sum(xs) / len(xs) if xs else float("inf")

        sorted_lanes = sorted(lanes, key=mean_x)[: self.MAX_LANES]

        for slot, lane in enumerate(sorted_lanes):
            for x, y in zip(lane, h_samples):
                if x < 0:
                    continue
                r = self._row_index(y)
                if r is not None:
                    target[slot, r] = float(x)

        return target

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # --- image ---
        img_rel = sample["raw_file"]
        img_path = str(self.image_root / img_rel)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        orig_h, orig_w = img_bgr.shape[:2]
        img_bgr = cv2.resize(img_bgr, (self.image_size[1], self.image_size[0]))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # --- target ---
        lanes = sample.get("lanes", [])
        h_samples = sample.get("h_samples", [])
        target = self._lanes_to_target(lanes, h_samples)

        # --- augmentation (training only) ---
        if self.augment:
            img_rgb, target = self._augment(img_rgb, target)

        image = self.to_tensor_image(img_rgb)

        return {
            "image": image,
            "target": target,
            "meta": {
                "image_path": img_path,
                # TuSimple has no scenario labels; set to None.
                # The evaluation step will skip per-scenario metrics.
                "scenario": None,
                "orig_size": (orig_h, orig_w),
            },
        }

    def _augment(self, image: np.ndarray, target: torch.Tensor):
        """Horizontal flip: mirror the image, x-coordinates, and lane slot order."""
        if np.random.rand() > 0.5:
            image = np.fliplr(image).copy()
            real = target >= 0
            flipped = target.clone()
            flipped[real] = (self.ORIG_WIDTH - 1) - target[real]
            # Reverse slot order so slot 0 stays "leftmost" after mirroring.
            target = torch.flip(flipped, dims=[0])
        return image, target
