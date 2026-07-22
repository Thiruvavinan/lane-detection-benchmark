# models/

**Why does this folder exist?**

This folder contains **architecture definitions only**.

It does not contain:
- Training loops
- Loss functions
- Optimizers or schedulers
- Metric computation
- Dataset loading

Those live in `training/` and `evaluation/` respectively.

---

## The model interface

Every architecture implements `BaseModel`:

```python
class BaseModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x      : [B, 3, H, W]  float32
        # return : shape and meaning defined by the TARGET DATASET, not
        #          by this class — e.g. for TuSimple, a
        #          [B, 2, MAX_LANES, NUM_ROWS] tensor (x-coord + existence
        #          logit per lane slot per row). See data/datasets/base.py.
```

That's the entire contract — the model never sees the loss or dataset,
but its output must match whatever dataset it targets, so evaluation can
apply that dataset's own official metric with no conversion. U-Net and
DeepLab both use the shared `LanePointHead` (`models/heads.py`) for
this — one head, reusable across backbones.

---

## Contents

```
models/
├── base.py         # BaseModel — all architectures implement this
├── heads.py         # LanePointHead — shared output head, any backbone can attach it
├── unet.py         # Milestone 2: U-Net baseline
├── deeplab.py      # Milestone 4: DeepLabV3+
├── lanesegnet.py   # Milestone 5: LaneSegNet (transformer)
└── __init__.py     # Registry — build_model("unet", ...)
```

---

## Adding a new architecture

1. Create `models/<name>.py` and subclass `BaseModel`
2. Implement `forward(x) -> logits`
3. Register it in `models/__init__.py`
4. Add a config in `configs/<name>_tusimple.yaml`

Zero changes needed in training/, evaluation/, or visualization/.
