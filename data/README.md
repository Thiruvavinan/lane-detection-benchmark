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
        #   "target": dataset's OWN native ground-truth representation —
        #             type/shape defined by the dataset, not by this
        #             class. TuSimple's is a [MAX_LANES, NUM_ROWS] tensor
        #             of x per lane per row (see datasets/tusimple.py).
        #   "meta":   dict                   # scenario tags, image path, etc.
        # }
```

Every model outputs its dataset's `target` format directly — not a
universal mask. A single-channel mask can't reliably separate lane
instances after the fact, so the model matches the dataset's own
annotation format instead, and evaluation scores it with zero conversion.

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
- Native sparse-keypoint annotations → matches the official evaluation
  format exactly, no lossy conversion needed
- Small enough to iterate on a single GPU
