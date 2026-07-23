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

### 1. Accuracy metrics (`tusimple_metrics.py`)

Models output their target dataset's own representation directly (see
data/datasets/base.py, models/base.py) — for TuSimple, sparse keypoints.
`scripts/evaluate.py` has no dataset-specific code: it resolves the
right evaluator via `evaluation.build_evaluator(dataset_name)` (see
`EVALUATORS` in `evaluation/__init__.py`) and lets the dataset class
convert its own output into that evaluator's input (`score_batch`, see
`data/datasets/base.py`). For TuSimple that's `TuSimpleEvaluator`,
scored with no conversion step in between:

| Metric | Description |
|--------|-------------|
| Accuracy | fraction of matched ground-truth points within ±20px |
| FP | unmatched predicted lanes / predicted lanes |
| FN | unmatched ground-truth lanes / ground-truth lanes |

**Per-scenario breakdown** — when `meta["scenario"]` is available,
metrics are also reported per scenario. TuSimple itself has no scenario
labels, so this is currently empty for it; a future dataset that does
label scenarios would populate it automatically.

`metrics.py` (pixel IoU/F1 on a mask) isn't wired into `evaluate.py`
yet — predictions are points now, not masks. Rendering points into a
mask is well-defined (unlike the reverse), so it's a reasonable
follow-up, just not done.

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
