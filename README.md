# Lane Detection Benchmark

A modular benchmark for evaluating modern lane detection architectures under a **common training and evaluation pipeline**.

---

## Objectives

| # | Objective |
|---|-----------|
| 1 | **Common pipeline** — swapping a model requires changes in one place only |
| 2 | **Fair comparison** — every architecture trains and evaluates identically |
| 3 | **Engineering over leaderboard** — accuracy is one column, not the whole story |

---

## Milestones

```
Milestone 1 — Dataset       TuSimple
Milestone 2 — Baseline      U-Net
Milestone 3 — Evaluation    Metrics · Visualizations · Failure cases
Milestone 4 — Architecture  DeepLab (same pipeline, zero changes elsewhere)
Milestone 5 — Transformer   LaneSegNet
```

---

## What makes this different

Most benchmarks stop at:

```
IoU: 95.2   F1: 0.94   ✓ Done
```

This one adds an **Engineering Evaluation** section for every architecture:

### Runtime
| Metric | Description |
|--------|-------------|
| FPS | Frames per second on a fixed GPU |
| Latency (ms) | End-to-end inference time per frame |

### Memory
| Metric | Description |
|--------|-------------|
| Peak GPU (MB) | Maximum allocation during inference |

### Failure Analysis
Qualitative examples across hard scenarios:

| Scenario | Tag |
|----------|-----|
| Curves | `curve` |
| Shadows | `shadow` |
| Night | `night` |
| Rain | `rain` |
| Construction zones | `construction` |
| Lane merge | `merge` |
| Exit ramps | `exit` |

### Robustness Table
Instead of a single IoU number, every architecture reports:

| Scenario | IoU | F1 | FPS |
|----------|-----|----|-----|
| Straight roads | | | |
| Curves | | | |
| Exit | | | |
| Merge | | | |
| Night | | | |
| Rain | | | |

---

## Repository layout

```
lane-detection-benchmark/
│
├── data/               # Dataset download, preprocessing, splits
├── models/             # Architectures only — no training code
├── training/           # Optimizer, scheduler, losses — no models
├── evaluation/         # Metrics only — no training, no models
├── visualization/      # Plots, overlays, failure-case exports
├── configs/            # One YAML per experiment
├── scripts/            # Entry-point scripts (train, eval, benchmark)
└── docs/               # Design decisions and pipeline diagrams
```

Every folder contains a `README.md` that answers: **why does this folder exist?**

---

## Pipeline

```
Dataset
  ↓
Model interface   (models/)
  ↓
Training          (training/)
  ↓
Evaluation        (evaluation/)
  ↓
Visualization     (visualization/)
```

Every model plugs into the same pipeline. Changing the architecture does **not** require touching training, evaluation, or visualization code.

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download TuSimple
python scripts/download_tusimple.py --out data/tusimple

# 3. Train baseline
python scripts/train.py --config configs/unet_tusimple.yaml

# 4. Evaluate + engineering report
python scripts/evaluate.py --config configs/unet_tusimple.yaml
python scripts/benchmark.py  --config configs/unet_tusimple.yaml
```

---

## Results

### U-Net (Milestone 2 baseline)

Trained on TuSimple, 360×640, early-stopped at epoch 10 (`val_loss` stopped
improving past that point — see `runs/unet_tusimple/`). Evaluated on the
full TuSimple test set (2,782 images).

| Metric set | Metric | Value |
|---|---|---|
| Pixel-level (ours, strict) | IoU | 0.6070 |
| | F1 | 0.7555 |
| | Precision | 0.7410 |
| | Recall | 0.7705 |
| Point-level (official TuSimple, ±20px) | Accuracy | 0.7734 |
| | FP | 0.1194 |
| | FN | 0.2450 |

Note: the official-metric numbers aren't directly comparable to published
TuSimple leaderboard entries (which cluster around 95–97% accuracy) — this
baseline stopped well short of convergence, and `mask_to_lanes()`
(`evaluation/tusimple_metrics.py`) is a heuristic for turning a dense mask
into per-row keypoints (connected components per row, linked into tracks
by extrapolating each track's trend), not a real instance-aware lane
decoder. Good enough for comparing architectures under this pipeline; not
a leaderboard submission. It also only handles dense-mask predictions —
a model with native point output (planned for PINet, Milestone 4) should
bypass this heuristic and score its real points directly, since a plain
binary mask can't reliably separate lane instances that are close together
(no per-lane identity in a single-channel mask).

Engineering evaluation (FPS, latency, peak GPU memory) and per-scenario
failure analysis are not yet populated — TuSimple provides no scenario
labels (`curve`/`night`/`rain`/etc.), so the failure-case export only has
a single "unlabeled" bucket for now.

Remaining milestones (PINet, LaneSegNet) will be added to this table as
they're trained.

---

## Contributing

See [`docs/contributing.md`](docs/contributing.md).
