"""
models/heads.py
----------------
Shared output head(s) usable by any backbone in this benchmark.

LanePointHead turns a backbone's spatial feature map into TuSimple's own
lane representation directly: an x-coordinate and an existence logit per
lane slot per row of the dataset's canonical grid (see
data/datasets/tusimple.py). Any dense-feature backbone can attach this
as its final layer — it has no opinion on what produced the features, so
U-Net and DeepLab both use the exact same head rather than each growing
their own bespoke output layer. Architectures whose native output
already is point-based (e.g. PINet) don't need this at all.
"""

import torch
import torch.nn as nn


class LanePointHead(nn.Module):
    """
    Parameters
    ----------
    in_channels : channels of the incoming feature map
    max_lanes   : number of lane slots to predict (dataset-defined)
    num_rows    : number of canonical row positions (dataset-defined)
    orig_width  : original image width, to scale predicted x back to
                  the dataset's own pixel coordinate space

    Design
    ------
    Height is pooled to `num_rows`; width is left at native resolution
    and read via a per-row soft-argmax (weighted average of x-positions
    under a softmax over the row's logits) rather than pooled away,
    since pooling width first would discard the spatial signal needed to
    localize x. Existence is the peak (max) response of that same
    softmax input — a sharp, confident row looks like a real point; a
    lane-free row produces a flat distribution. Keeps the head to one
    conv layer; a reasonable start, not a claim of optimality.

    Output
    ------
    [B, 2, max_lanes, num_rows] — channel 0 is x in original pixel
    scale, channel 1 is the existence logit (pre-sigmoid).
    """

    def __init__(self, in_channels: int, max_lanes: int, num_rows: int, orig_width: float):
        super().__init__()
        self.orig_width = orig_width
        self.pool_h = nn.AdaptiveAvgPool2d((num_rows, None))
        self.row_logits = nn.Conv2d(in_channels, max_lanes, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = self.pool_h(features)              # [B, C, num_rows, W]
        logits = self.row_logits(pooled)             # [B, max_lanes, num_rows, W]

        probs = torch.softmax(logits, dim=-1)
        width = logits.shape[-1]
        positions = torch.linspace(0, self.orig_width, width, device=logits.device)
        coord = (probs * positions).sum(dim=-1)       # [B, max_lanes, num_rows]

        exist_logit = logits.amax(dim=-1)              # [B, max_lanes, num_rows]

        return torch.stack([coord, exist_logit], dim=1)   # [B, 2, max_lanes, num_rows]
