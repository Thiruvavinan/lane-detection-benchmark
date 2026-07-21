"""
models/__init__.py
------------------
Model registry. Maps config names to classes.

Usage (from config YAML):
    model:
      name: unet
      base_channels: 64
"""

from .base import BaseModel
from .unet import UNet
from .deeplab import DeepLabV3Plus
from .lanesegnet import LaneSegNet

MODELS = {
    "unet": UNet,
    "deeplab": DeepLabV3Plus,
    "lanesegnet": LaneSegNet,
}


def build_model(name: str, **kwargs) -> BaseModel:
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODELS)}")
    return MODELS[name](**kwargs)
