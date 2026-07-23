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
Milestone 4 — Architecture  PINet
Milestone 5 — Transformer   LaneSegNet
```

---

## What makes this different

Most benchmarks stop at a single accuracy number. This one also tracks
engineering cost (FPS, latency, peak GPU memory — `evaluation/engineering.py`)
and exports failure cases per scenario (curve, night, rain, …) when a
dataset provides scenario labels. TuSimple doesn't, so that's currently
one "unlabeled" bucket rather than a real per-scenario breakdown.

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

*Pending — U-Net is retraining under a new output-format design (see
`data/README.md` / `models/README.md`). Pixel-level IoU/F1 and
per-scenario failure analysis are on hold — see `evaluation/README.md`.*

---

## Contributing

See [`docs/contributing.md`](docs/contributing.md).
