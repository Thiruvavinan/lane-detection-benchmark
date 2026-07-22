"""
data/datasets/base.py
---------------------
Abstract base class for all datasets in this benchmark.

Every concrete dataset must implement __getitem__ returning the canonical
schema documented below. The schema is the contract between data/ and every
other part of the pipeline (training/, evaluation/, visualization/).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """
    All lane detection datasets in this benchmark extend this class.

    __getitem__ contract
    --------------------
    Returns a dict with exactly these keys:

        image  : torch.Tensor  [C, H, W], float32, values in [0, 1]
        target : the dataset's OWN native ground-truth representation.
                 Its type and shape are defined by the dataset, not by
                 this base class. Every model trained on a given dataset
                 must output that dataset's representation directly, and
                 evaluation always scores it with that dataset's own
                 official metric, with no conversion step in between.
                 (e.g. TuSimpleDataset's target is a
                 [MAX_LANES, NUM_ROWS] tensor of x per lane per row.)
        meta   : dict  arbitrary metadata — must include at minimum:
                    "image_path" : str
                    "scenario"   : str | None
                                   one of: straight, curve, exit, merge,
                                           night, rain, shadow, construction
                                   None when the label is unavailable.

    Why not one universal target format (e.g. a dense mask) for every
    dataset? Because converting a model's output back into a dataset's
    own official-metric format after the fact is often lossy or
    ill-posed — e.g. a single-channel binary mask cannot reliably
    separate lane instances that sit close together, no matter how the
    extraction is written (we tried; see the git history of
    evaluation/tusimple_metrics.py). Making the model's output format
    match the target dataset's native annotation format directly avoids
    that conversion, and its accompanying information loss, entirely.

    The `scenario` tag is what enables per-scenario metrics in evaluation/.
    If your dataset does not provide scenario labels, set it to None and the
    evaluation step will simply skip the per-scenario breakdown.
    """

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ...

    # ------------------------------------------------------------------
    # Optional helpers subclasses can call
    # ------------------------------------------------------------------

    @staticmethod
    def to_tensor_image(np_image) -> torch.Tensor:
        """Convert HWC uint8 numpy array to CHW float32 tensor in [0, 1]."""
        import numpy as np
        img = np.asarray(np_image, dtype=np.float32) / 255.0
        return torch.from_numpy(img.transpose(2, 0, 1))

    # ------------------------------------------------------------------
    # DataLoader collation
    # ------------------------------------------------------------------

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Stack "image" and "target" into batch tensors; keep "meta" as a
        plain list of per-sample dicts.

        Every consumer downstream (evaluation/, visualization/) indexes
        meta per-sample (`meta[i]`), and meta values like `scenario` are
        often `None` — PyTorch's default collate can't batch `None` and
        would otherwise merge the list of dicts into a dict of lists.
        """
        return {
            "image": torch.stack([sample["image"] for sample in batch]),
            "target": torch.stack([sample["target"] for sample in batch]),
            "meta": [sample["meta"] for sample in batch],
        }
