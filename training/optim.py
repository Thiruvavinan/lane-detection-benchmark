"""
training/optim.py
-----------------
Optimizer and scheduler factories.

Resolved from config YAML:
    optimizer:
      name: adamw
      lr: 1e-4
      weight_decay: 1e-4

    scheduler:
      name: cosine
      T_max: 50
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    OneCycleLR,
)


def build_optimizer(name: str, params, **kwargs) -> optim.Optimizer:
    OPTIMIZERS = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "sgd": optim.SGD,
    }
    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer '{name}'. Available: {list(OPTIMIZERS)}")
    return OPTIMIZERS[name](params, **kwargs)


def build_scheduler(name: str, optimizer: optim.Optimizer, **kwargs):
    SCHEDULERS = {
        "cosine": CosineAnnealingLR,
        "step": StepLR,
        "onecycle": OneCycleLR,
    }
    if name not in SCHEDULERS:
        raise ValueError(f"Unknown scheduler '{name}'. Available: {list(SCHEDULERS)}")
    return SCHEDULERS[name](optimizer, **kwargs)
