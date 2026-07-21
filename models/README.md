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
        # x     : [B, 3, H, W]  float32
        # return : [B, 1, H, W]  float32  (raw logits, not sigmoid)
```

That's the entire contract. The training pipeline calls `forward`, applies the
loss, and runs the optimizer. The model never sees the loss or the dataset.

---

## Contents

```
models/
├── base.py         # BaseModel — all architectures implement this
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
