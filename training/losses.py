"""
training/losses.py
------------------
Loss functions used across all architectures.

All losses accept:
    logits : [B, 1, H, W]  float32  (raw model output, no activation)
    targets: [B, H, W]     int64    (binary mask, values in {0, 1})

Keeping losses here — not inside model files — means every architecture
is evaluated with identical loss functions. Fair comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.

    Useful when positive pixels (lanes) are a small fraction of the image.
    Dice is less sensitive to class imbalance than plain BCE.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).squeeze(1)          # [B, H, W]
        targets = targets.float()
        intersection = (probs * targets).sum(dim=(1, 2))
        union = probs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """
    BCE + Dice.

    Default loss for all milestones.

    Parameters
    ----------
    bce_weight  : weight on the BCE term
    dice_weight : weight on the Dice term
    pos_weight  : optional positive-class weight passed to BCE
                  (use ~ratio of negative to positive pixels)
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: float | None = None,
    ):
        super().__init__()
        pw = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.dice = DiceLoss()
        self.bce_w = bce_weight
        self.dice_w = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_f = targets.float().unsqueeze(1)          # [B, 1, H, W]
        bce_loss = self.bce(logits, targets_f)
        dice_loss = self.dice(logits, targets)
        return self.bce_w * bce_loss + self.dice_w * dice_loss


def build_loss(name: str, **kwargs) -> nn.Module:
    LOSSES = {
        "bce": nn.BCEWithLogitsLoss,
        "dice": DiceLoss,
        "combined": CombinedLoss,
    }
    if name not in LOSSES:
        raise ValueError(f"Unknown loss '{name}'. Available: {list(LOSSES)}")
    return LOSSES[name](**kwargs)
