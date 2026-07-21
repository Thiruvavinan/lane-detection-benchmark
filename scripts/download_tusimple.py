#!/usr/bin/env python
"""
scripts/download_tusimple.py
----------------------------
Instructions and helper for obtaining the TuSimple dataset.

TuSimple requires a Kaggle account to download. This script prints the
exact steps and, if kaggle-api is installed and configured, downloads
and extracts the dataset automatically.

Usage
-----
    python scripts/download_tusimple.py --out data/tusimple
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


KAGGLE_DATASET = "manideep1108/tusimple"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/tusimple", help="Output directory")
    parser.add_argument("--auto", action="store_true", help="Try kaggle-api auto-download")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TuSimple Dataset")
    print("=" * 60)

    if args.auto:
        _try_kaggle_download(KAGGLE_DATASET, out)
    else:
        _print_manual_instructions(out)


def _try_kaggle_download(dataset: str, out: Path):
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("[!] kaggle package not found. Install with:")
        print("      pip install kaggle")
        print("    Then set up ~/.kaggle/kaggle.json")
        sys.exit(1)

    print(f"Downloading {dataset} to {out} ...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset, "-p", str(out), "--unzip"],
        check=True,
    )
    print(f"Done. Dataset available at: {out}")


def _print_manual_instructions(out: Path):
    print(f"""
Manual download steps
---------------------
1. Create a Kaggle account at https://www.kaggle.com

2. Go to: https://www.kaggle.com/datasets/manideep1108/tusimple

3. Click "Download" and save the zip to {out}/

4. Unzip:
       unzip {out}/tusimple.zip -d {out}/

5. Expected layout:
       {out}/
         clips/                  ← image sequences
         label_data_0313.json
         label_data_0531.json
         label_data_0601.json
         test_label.json

Alternatively, install kaggle-api and run:
    python scripts/download_tusimple.py --out {out} --auto
""")


if __name__ == "__main__":
    main()
