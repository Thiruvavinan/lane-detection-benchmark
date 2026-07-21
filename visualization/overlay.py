"""
visualization/overlay.py
------------------------
Draw predicted lane masks over input images.

All functions return numpy arrays (H, W, 3) uint8 ready for cv2.imwrite
or matplotlib display.
"""

import numpy as np
import torch


def mask_overlay(
    image: np.ndarray,       # [H, W, 3] uint8
    pred_mask: np.ndarray,   # [H, W]    binary
    gt_mask: np.ndarray | None = None,  # [H, W] binary, optional
    pred_color: tuple = (0, 255, 0),    # green
    gt_color: tuple = (255, 0, 0),      # red
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlay prediction (and optionally ground truth) on an RGB image.

    Colors:
        Green — true positive (pred=1, gt=1)
        Red   — false negative (pred=0, gt=1)  [only when gt provided]
        Blue  — false positive (pred=1, gt=0)  [only when gt provided]
    """
    out = image.copy().astype(np.float32)

    if gt_mask is None:
        # Simple: just draw the prediction
        for c, v in enumerate(pred_color):
            out[:, :, c] = np.where(pred_mask, out[:, :, c] * (1 - alpha) + v * alpha, out[:, :, c])
    else:
        # Three-way: TP green, FP blue, FN red
        tp = (pred_mask == 1) & (gt_mask == 1)
        fp = (pred_mask == 1) & (gt_mask == 0)
        fn = (pred_mask == 0) & (gt_mask == 1)

        colors = [(tp, (0, 255, 0)), (fp, (0, 0, 255)), (fn, (255, 0, 0))]
        for mask, color in colors:
            for c, v in enumerate(color):
                out[:, :, c] = np.where(mask, out[:, :, c] * (1 - alpha) + v * alpha, out[:, :, c])

    return out.clip(0, 255).astype(np.uint8)


def make_comparison_grid(
    image: np.ndarray,      # [H, W, 3]
    gt_mask: np.ndarray,    # [H, W]
    pred_mask: np.ndarray,  # [H, W]
) -> np.ndarray:
    """
    Produce a side-by-side grid: [input | ground truth | prediction | overlay]
    """
    H, W = image.shape[:2]

    def to_rgb_mask(m):
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        rgb[m == 1] = (255, 255, 255)
        return rgb

    gt_rgb = to_rgb_mask(gt_mask)
    pred_rgb = to_rgb_mask(pred_mask)
    ov = mask_overlay(image, pred_mask, gt_mask)

    return np.concatenate([image, gt_rgb, pred_rgb, ov], axis=1)


def logits_to_mask(logits: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """Convert [1, H, W] logit tensor to [H, W] binary numpy array."""
    probs = torch.sigmoid(logits.squeeze(0)).cpu().numpy()
    return (probs > threshold).astype(np.uint8)
