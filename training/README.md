# training/

**Why does this folder exist?**

This folder owns the training loop and everything the loop needs:
optimizer, scheduler, and loss functions.

It does not contain:
- Model architectures (those are in `models/`)
- Metric computation (that is in `evaluation/`)
- Dataset definitions (those are in `data/`)

---

## Contents

```
training/
├── trainer.py      # Training loop — model-agnostic
├── losses.py       # Loss functions (BCE, Dice, combined)
├── optim.py        # Optimizer and scheduler factories
└── README.md
```

---

## Design

The `Trainer` receives:
- A model that satisfies `BaseModel.forward(x) -> logits`
- A loss function
- Train and validation `DataLoader`s
- An optimizer and scheduler

It knows nothing about which architecture is being trained.
Swapping U-Net for DeepLab requires changing only the config file.

---

## Loss functions

| Name | When to use |
|------|-------------|
| `BCEWithLogitsLoss` | Simple baseline |
| `DiceLoss` | Handles class imbalance (lanes are sparse) |
| `CombinedLoss` | BCE + Dice — default for all milestones |
