# data/

**Why does this folder exist?**

This folder owns everything that touches raw data: download helpers, preprocessing, split definitions, and the dataset interface that the training pipeline consumes.

Nothing in here knows about models, losses, or metrics.

---

## Contents

```
data/
├── tusimple/           # Raw TuSimple files (after download)
├── datasets/
│   ├── __init__.py
│   ├── base.py         # Abstract Dataset — all datasets implement this
│   └── tusimple.py     # TuSimple-specific loader
└── README.md
```

---

## Dataset interface

Every dataset implements `BaseDataset`:

```python
class BaseDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx) -> dict:
        # Must return:
        # {
        #   "image":  torch.Tensor [C, H, W],
        #   "mask":   torch.Tensor [H, W],   # binary lane mask
        #   "meta":   dict                   # scenario tags, image path, etc.
        # }
```

The `meta` dict is what powers failure-case analysis. Every sample carries its scenario tag (`curve`, `night`, `rain`, …) so the evaluation step can compute per-scenario metrics automatically when labels are available.

---

## Adding a new dataset

1. Subclass `BaseDataset` in `datasets/<name>.py`
2. Implement `__getitem__` returning the schema above
3. Register it in `datasets/__init__.py`
4. Add a config file in `configs/`

No changes needed in training, evaluation, or visualization.

---

## Why TuSimple first?

- Widely used → easy to compare against published numbers
- Clean binary lane masks → simple baseline target
- Small enough to iterate on a single GPU
