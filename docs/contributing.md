# Contributing

## Adding a new model (same dataset)

1. Create `models/<name>.py`, subclass `BaseModel`, implement `forward(x)` returning the
   target dataset's representation (for TuSimple: attach the shared `LanePointHead` —
   see `models/heads.py` — rather than inventing a new head)
2. Register in `models/__init__.py`
3. Add `configs/<name>_tusimple.yaml` — copy an existing config, change only the `model:` block
4. Run the full pipeline to confirm nothing else breaks:
   ```
   python scripts/train.py    --config configs/<name>_tusimple.yaml
   python scripts/evaluate.py --config configs/<name>_tusimple.yaml
   python scripts/benchmark.py --config configs/<name>_tusimple.yaml
   ```
5. Update the results table in `README.md`

## Adding a new dataset

The model's output format is defined by the dataset it targets, not fixed
across the whole repo (see `data/datasets/base.py`, `models/base.py`) —
so this is more than a loader. Nothing outside these files should need
to change; `scripts/train.py`/`evaluate.py`/`benchmark.py` are already
generic over any dataset in the registries below.

1. `data/datasets/<name>.py` — subclass `BaseDataset`. Implement
   `__getitem__` returning `{image, target, meta}` (`meta["scenario"]`
   at minimum), and `score_batch(pred, targets, meta, evaluator, **kwargs)`
   — the dataset owns converting its own target/output shape into
   whatever its official evaluator expects. Register in
   `data/datasets/__init__.py`'s `DATASETS`.
2. `evaluation/<name>_metrics.py` — the dataset's own official metric
   (not `evaluation/metrics.py` — that's the generic pixel IoU/F1,
   already dataset-agnostic). Register in `evaluation/__init__.py`'s
   `EVALUATORS`.
3. If the new dataset's native representation is a genuinely different
   *family* (not just a different size of the same point-grid shape —
   see `models/heads.py`), every model needs a matching head: either a
   new shared head module, or a per-model addition if it can't be shared.
4. `configs/<model>_<dataset>.yaml`

## Style

- No training code in `models/`
- No model imports in `training/` or `evaluation/`
- Every new folder gets a `README.md` answering "why does this exist?"
