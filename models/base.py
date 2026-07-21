"""
models/base.py
--------------
Abstract base class for all lane detection models in this benchmark.

Contract
--------
forward(x) receives a batch of RGB images and returns raw logits (not
sigmoid-activated). The training pipeline applies the loss; the model
does not need to know what loss function is used.

    input  x      : [B, 3, H, W]  float32  pixel values in [0, 1]
    output logits : [B, 1, H, W]  float32  unbounded

Every architecture in models/ must satisfy this contract so it can be
dropped into the shared training and evaluation pipeline unchanged.
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
        torch.Tensor  [B, 1, H, W]  raw logits
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
