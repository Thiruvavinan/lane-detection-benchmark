# visualization/

**Why does this folder exist?**

This folder turns numbers and predictions into images and plots that a
human can look at. It is the only folder allowed to produce image files.

It does not contain:
- Model code
- Loss functions
- Metric computation

---

## Contents

```
visualization/
├── overlay.py          # Draw predicted mask over the input image
├── failure_cases.py    # Export worst-performing samples per scenario
├── plots.py            # Training curves, per-scenario bar charts
└── README.md
```

---

## Failure case export

`failure_cases.py` takes the test set predictions, sorts each scenario
bucket by IoU (ascending), and saves the N worst examples as side-by-side
image grids:

```
outputs/failure_cases/
  curve_worst_10.png
  night_worst_10.png
  rain_worst_10.png
  …
```

Each grid shows:  `[input image | ground truth | prediction]`

This is the visual companion to the per-scenario metrics table.
