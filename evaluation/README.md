# evaluation/

**Why does this folder exist?**

This folder computes **metrics only**.

It does not contain:
- Model architectures
- Training loops or losses
- Dataset loading logic
- Visualization code (that is in `visualization/`)

---

## Contents

```
evaluation/
├── metrics.py            # IoU, F1, per-scenario breakdown (pixel-level, strict)
├── tusimple_metrics.py   # Official TuSimple Accuracy/FP/FN (point-level, ±20px)
├── engineering.py        # FPS, latency, peak GPU memory
└── README.md
```

---

## Two kinds of evaluation

### 1. Accuracy metrics (`metrics.py`)

Standard segmentation metrics computed over the test set:

| Metric | Description |
|--------|-------------|
| IoU | Intersection over Union (binary) |
| F1 | Dice coefficient = 2·P·R / (P+R) |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |

**Per-scenario breakdown** — when `meta["scenario"]` is available,
metrics are also reported per scenario:

| Scenario | IoU | F1 |
|----------|-----|----|
| straight | | |
| curve | | |
| night | | |
| … | | |

### 2. Engineering metrics (`engineering.py`)

Profiled on a fixed GPU (A100 / RTX 3090 — record which one):

| Metric | Description |
|--------|-------------|
| FPS | Frames per second at batch size 1 |
| Latency (ms) | Median over 200 warmup + 500 timed runs |
| Peak GPU (MB) | `torch.cuda.max_memory_allocated()` |

---

## Why separate from training?

Metrics are computed post-training on a held-out test set. Keeping them
here means:
- The training loop is not cluttered with metric logic
- You can re-evaluate a checkpoint with different metrics without retraining
- Engineering evaluation can be run independently on any saved model
