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

*Populated after Milestone 3.*

---

## Contributing

See [`docs/contributing.md`](docs/contributing.md).
