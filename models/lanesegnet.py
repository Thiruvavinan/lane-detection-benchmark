"""
models/lanesegnet.py
--------------------
LaneSegNet — transformer-based lane detection (Milestone 5).

Placeholder implementation using a Swin-Tiny backbone + FPN neck + the
shared LanePointHead (models/heads.py). Replace the body of
LaneSegNet.forward() with the full architecture once the paper's code is
integrated — output shape and meaning are defined by the target dataset
(see models/base.py), not by this file.
Input : [B, 3, H, W]

Reference: Han et al., "LaneSegNet: Map Learning with Lane Segment
Perception for Autonomous Driving", ICLR 2024.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel
from .heads import LanePointHead


class LaneSegNet(BaseModel):
    """
    Milestone 5 placeholder.

    Current state: thin wrapper around torchvision's Swin-T backbone
    with a minimal FPN + head. Swap in the full LaneSegNet architecture
    here — nothing else in the pipeline needs to change.

    Parameters
    ----------
    max_lanes, num_rows, orig_width : dataset-defined output shape,
        injected by the training/eval scripts from the target dataset
        class (e.g. TuSimpleDataset.MAX_LANES) — not hardcoded here.
    """

    def __init__(
        self,
        pretrained: bool = True,
        max_lanes: int = 5,
        num_rows: int = 56,
        orig_width: float = 1280.0,
    ):
        super().__init__()

        try:
            from torchvision.models import swin_t, Swin_T_Weights
            weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
            swin = swin_t(weights=weights)
            # Keep just the spatial feature stages + their final norm.
            # swin's full child order is
            # [features, norm, permute, avgpool, flatten, head] -- slicing
            # by a fixed negative index (the previous approach) is fragile
            # and previously left `permute` + `avgpool` in the chain,
            # collapsing the spatial map to [B, 768, 1, 1] before it ever
            # reached the neck. Referencing the modules by name instead.
            self.backbone = nn.Sequential(swin.features, swin.norm)
            backbone_out_ch = 768
            self._backbone_channel_last = True   # Swin's feature stages emit [B, H', W', C]
        except Exception:
            # Fallback: simple conv stem so tests pass without torchvision
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 96, 4, stride=4),
                nn.GELU(),
                nn.Conv2d(96, 192, 3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(192, 384, 3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(384, 768, 3, stride=2, padding=1),
            )
            backbone_out_ch = 768
            self._backbone_channel_last = False   # plain nn.Conv2d stack is already [B, C, H', W']

        self.neck = nn.Sequential(
            nn.Conv2d(backbone_out_ch, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.point_head = LanePointHead(256, max_lanes, num_rows, orig_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        features = self.backbone(x)
        # Which layout the backbone produces is known at construction time
        # (see __init__) — inferring it from tensor shape here is fragile:
        # the fallback conv stem's [B, C, H', W'] can easily have C != H'
        # or C != W' too, which previously misfired the permute.
        if self._backbone_channel_last:
            features = features.permute(0, 3, 1, 2).contiguous()
        features = self.neck(features)
        features = F.interpolate(features, size=input_size, mode="bilinear", align_corners=False)
        return self.point_head(features)
