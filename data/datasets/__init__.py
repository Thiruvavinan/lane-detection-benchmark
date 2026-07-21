"""
data/datasets/__init__.py
-------------------------
Registry of all datasets.

To add a new dataset:
  1. Create data/datasets/<name>.py and subclass BaseDataset
  2. Import it here and add it to DATASETS

The training script resolves dataset class by name from the config YAML:
  dataset:
    name: tusimple
    root: data/tusimple
"""

from .base import BaseDataset
from .tusimple import TuSimpleDataset

DATASETS = {
    "tusimple": TuSimpleDataset,
}


def build_dataset(name: str, **kwargs) -> BaseDataset:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASETS)}")
    return DATASETS[name](**kwargs)
