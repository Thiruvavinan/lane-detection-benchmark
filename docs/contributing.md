# Contributing

## Adding a new model

1. Create `models/<name>.py`, subclass `BaseModel`, implement `forward(x) → logits`
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

1. Create `data/datasets/<name>.py`, subclass `BaseDataset`
2. Implement `__getitem__` returning `{image, mask, meta}` with `meta["scenario"]`
3. Register in `data/datasets/__init__.py`
4. Add a download helper in `scripts/download_<name>.py`
5. Add a config in `configs/<model>_<dataset>.yaml`

## Adding a new metric

Add a method to `evaluation/metrics.py`. Do not add metric logic to training/, models/, or visualization/.

## Style

- No training code in `models/`
- No model imports in `training/` or `evaluation/`
- Every new folder gets a `README.md` answering "why does this exist?"
