"""
models/deeplab.py
-----------------
DeepLabV3+ for lane segmentation (Milestone 4).

Uses a ResNet-50 backbone pretrained on ImageNet, followed by an ASPP
module and a lightweight decoder. Output is a single-channel logit map
at full input resolution.

Input  : [B, 3, H, W]
Output : [B, 1, H, W]  raw logits

Reference: Chen et al., "Encoder-Decoder with Atrous Separable
Convolution for Semantic Image Segmentation", ECCV 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models._utils import IntermediateLayerGetter

from .base import BaseModel


# ------------------------------------------------------------------
# ASPP
# ------------------------------------------------------------------

class ASPPConv(nn.Sequential):
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_ch: int = 2048, out_ch: int = 256, dilations=(6, 12, 18)):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )]
            + [ASPPConv(in_ch, out_ch, d) for d in dilations]
            + [ASPPPooling(in_ch, out_ch)]
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (len(dilations) + 2), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        return self.project(torch.cat([c(x) for c in self.convs], dim=1))


# ------------------------------------------------------------------
# DeepLabV3+
# ------------------------------------------------------------------

class DeepLabV3Plus(BaseModel):
    """
    Parameters
    ----------
    pretrained : bool
        Load ImageNet-pretrained ResNet-50 backbone. Default: True.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet50(weights=weights)

        # Dilate layer4 to preserve spatial resolution (output stride 16 → 8)
        for n, m in backbone.layer4.named_modules():
            if isinstance(m, nn.Conv2d) and m.stride == (2, 2):
                m.stride = (1, 1)
                m.dilation = (2, 2)
                m.padding = (2, 2)

        self.backbone = IntermediateLayerGetter(
            backbone, return_layers={"layer1": "low", "layer4": "high"}
        )

        # Low-level feature projection (from layer1, 256 ch)
        self.low_proj = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_ch=2048, out_ch=256)

        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Conv2d(256, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        features = self.backbone(x)

        low = self.low_proj(features["low"])
        high = self.aspp(features["high"])

        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = self.decoder(torch.cat([high, low], dim=1))
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return self.head(x)
