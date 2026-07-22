"""
training/losses.py
------------------
Loss functions used across all architectures.

A loss here is paired to a target dataset's own representation (see
data/datasets/base.py) — DiceLoss/CombinedLoss below are for a dense
[B, H, W] mask target, if some future dataset's native format is one.
TuSimple's target is points (see data/datasets/tusimple.py), scored by
LanePointLoss. Keeping losses here — not inside model files — means
every architecture targeting the same dataset is trained with the same
loss. Fair comparison.
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


class LanePointLoss(nn.Module):
    """
    Loss for TuSimple-style point targets: [MAX_LANES, NUM_ROWS] x per
    lane per row, UNKNOWN (-3) where the row falls outside a sample's
    own annotated range, INVALID (-2) where the row is annotated but
    this lane slot has no point, and >= 0 for a real x-coordinate (see
    data/datasets/tusimple.py for exactly what these mean).

    Combines:
      - SmoothL1 on x, only at rows with a real point (UNKNOWN/INVALID
        rows carry no coordinate information to regress toward).
      - BCE on the existence logit, at every row *except* UNKNOWN ones
        — INVALID rows are a real "no lane here" signal and must count,
        just not toward the coordinate term.

    Parameters
    ----------
    coord_weight : weight on the SmoothL1 coordinate term
    exist_weight : weight on the BCE existence term
    """

    UNKNOWN = -3.0

    def __init__(self, coord_weight: float = 1.0, exist_weight: float = 1.0):
        super().__init__()
        self.coord_weight = coord_weight
        self.exist_weight = exist_weight
        self.smooth_l1 = nn.SmoothL1Loss(reduction="none")
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred   : [B, 2, MAX_LANES, NUM_ROWS] -- pred[:,0]=x, pred[:,1]=existence logit
        target : [B, MAX_LANES, NUM_ROWS]     -- x, or UNKNOWN/INVALID sentinel
        """
        pred_coord = pred[:, 0]
        pred_exist_logit = pred[:, 1]

        has_point = target >= 0
        known = target != self.UNKNOWN

        coord_mask = has_point.float()
        coord_loss = (self.smooth_l1(pred_coord, target) * coord_mask).sum() / coord_mask.sum().clamp(min=1.0)

        exist_loss = self.bce(pred_exist_logit[known], has_point.float()[known])

        return self.coord_weight * coord_loss + self.exist_weight * exist_loss


def build_loss(name: str, **kwargs) -> nn.Module:
    LOSSES = {
        "bce": nn.BCEWithLogitsLoss,
        "dice": DiceLoss,
        "combined": CombinedLoss,
        "lane_points": LanePointLoss,
    }
    if name not in LOSSES:
        raise ValueError(f"Unknown loss '{name}'. Available: {list(LOSSES)}")
    return LOSSES[name](**kwargs)
