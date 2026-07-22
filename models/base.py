"""
models/base.py
--------------
Abstract base class for all lane detection models in this benchmark.

Contract
--------
forward(x) receives a batch of RGB images and returns raw predictions in
whatever representation the TARGET DATASET natively uses — see
data/datasets/base.py. For TuSimple, that's a [B, 2, MAX_LANES, NUM_ROWS]
tensor (x-coordinate + existence logit per lane slot per canonical row);
a different dataset with a different native annotation format would mean
a different output shape. The shape is defined by the dataset, not by
this base class: every model targeting a given dataset must produce that
dataset's representation directly, so evaluation can apply that
dataset's own official metric with zero conversion in between.

    input x : [B, 3, H, W]  float32  pixel values in [0, 1]

Models that target a dataset with a dense dictionary-of-pixels-style
representation would instead output a dense tensor here — this base
class does not assume points any more than it assumes masks. Every
architecture in models/ must satisfy whatever contract its target
dataset defines so it can be dropped into the shared training and
evaluation pipeline unchanged.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  [B, 3, H, W]

        Returns
        -------
        torch.Tensor  shape and meaning defined by the target dataset
        (see module docstring above)
        """
        ...

    # ------------------------------------------------------------------
    # Convenience: parameter count (logged at training start)
    # ------------------------------------------------------------------

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters() if not trainable_only else self.parameters()
        return sum(
            p.numel()
            for p in self.parameters()
            if (not trainable_only or p.requires_grad)
        )
