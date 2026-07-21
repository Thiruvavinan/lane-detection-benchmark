"""
data/datasets/tusimple.py
-------------------------
TuSimple lane detection dataset loader.

TuSimple provides:
  - RGB images at 1280×720
  - JSON annotation files with lane keypoints (not pixel masks)

We convert the keypoint annotations to binary pixel masks on the fly so that
every model in the benchmark receives the same [C, H, W] image + [H, W] mask
input regardless of what the original annotation format was.

Directory layout expected under `root`:
    root/
      clips/              # raw images
      label_data_0313.json
      label_data_0531.json
      label_data_0601.json
      test_label.json
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

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

        self.samples = self._load_samples()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_samples(self) -> List[dict]:
        samples = []
        if self.split in ("train", "val"):
            for fname in self.TRAIN_JSONS:
                fpath = self.root / fname
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

    def _keypoints_to_mask(self, lanes, h_samples, orig_h, orig_w) -> np.ndarray:
        """Draw lane polylines onto a binary mask."""
        H, W = self.image_size
        scale_x = W / orig_w
        scale_y = H / orig_h
        mask = np.zeros((H, W), dtype=np.uint8)
        for lane in lanes:
            pts = [
                (int(x * scale_x), int(y * scale_y))
                for x, y in zip(lane, h_samples)
                if x >= 0
            ]
            if len(pts) >= 2:
                for i in range(len(pts) - 1):
                    cv2.line(mask, pts[i], pts[i + 1], color=1, thickness=5)
        return mask

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # --- image ---
        img_rel = sample["raw_file"]
        img_path = str(self.root / img_rel)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        orig_h, orig_w = img_bgr.shape[:2]
        img_bgr = cv2.resize(img_bgr, (self.image_size[1], self.image_size[0]))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # --- mask ---
        lanes = sample.get("lanes", [])
        h_samples = sample.get("h_samples", [])
        mask_np = self._keypoints_to_mask(lanes, h_samples, orig_h, orig_w)

        # --- augmentation (training only) ---
        if self.augment:
            img_rgb, mask_np = self._augment(img_rgb, mask_np)

        image = self.to_tensor_image(img_rgb)
        mask = self.to_tensor_mask(mask_np)

        return {
            "image": image,
            "mask": mask,
            "meta": {
                "image_path": img_path,
                # TuSimple has no scenario labels; set to None.
                # The evaluation step will skip per-scenario metrics.
                "scenario": None,
            },
        }

    def _augment(self, image: np.ndarray, mask: np.ndarray):
        """Minimal training augmentation: horizontal flip."""
        if np.random.rand() > 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
        return image, mask


def __init__():
    pass
