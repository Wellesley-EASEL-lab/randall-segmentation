#!/usr/bin/env python3
"""Build explicit train/validation/test and unlabeled classifier splits from reviews."""

from __future__ import annotations

import argparse
from pathlib import Path

from randall_pipeline.classifier_build import build_classifier_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews-csv", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    dataset = build_classifier_dataset(args.reviews_csv, args.asset_root, args.output_dir)
    print({split: len(rows) for split, rows in dataset.items()})


if __name__ == "__main__":
    main()
