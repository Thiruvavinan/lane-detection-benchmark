"""
models/unet.py
--------------
U-Net baseline (Milestone 2).

A standard encoder-decoder with skip connections.
Input  : [B, 3, H, W]
Output : [B, 1, H, W]  raw logits

Reference: Ronneberger et al., "U-Net: Convolutional Networks for
Biomedical Image Segmentation", MICCAI 2015.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel


# ------------------------------------------------------------------
# Building blocks
# ------------------------------------------------------------------

class DoubleConv(nn.Module):
    """Conv → BN → ReLU → Conv → BN → ReLU"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """MaxPool2d(2) → DoubleConv"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """Bilinear upsample → cat(skip) → DoubleConv"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle odd spatial dims
        dh = skip.shape[2] - x.shape[2]
        dw = skip.shape[3] - x.shape[3]
        x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv(torch.cat([skip, x], dim=1))


# ------------------------------------------------------------------
# U-Net
# ------------------------------------------------------------------

class UNet(BaseModel):
    """
    Parameters
    ----------
    base_channels : int
        Number of feature channels after the first convolution.
        Doubled at each encoder stage. Default: 64.
    """

    def __init__(self, base_channels: int = 64):
        super().__init__()
        c = base_channels

        # Encoder
        self.enc1 = DoubleConv(3, c)
        self.enc2 = Down(c, c * 2)
        self.enc3 = Down(c * 2, c * 4)
        self.enc4 = Down(c * 4, c * 8)

        # Bottleneck
        self.bottleneck = Down(c * 8, c * 16)

        # Decoder
        self.dec4 = Up(c * 16 + c * 8, c * 8)
        self.dec3 = Up(c * 8 + c * 4, c * 4)
        self.dec2 = Up(c * 4 + c * 2, c * 2)
        self.dec1 = Up(c * 2 + c, c)

        # Output head — raw logit, no activation
        self.head = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)

        # Decoder
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        return self.head(d1)
