"""Build the include/exclude classifier dataset from the canonical review CSV."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, Value

from .io import normalize_review_row, read_csv, safe_join

LABELED_SPLITS = ("train", "validation", "test")
LABEL_NAMES = ("exclude", "include")


def validate_and_partition(review_csv: Path, asset_root: Path) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    raw_rows = read_csv(review_csv)
    if not raw_rows:
        raise ValueError(f"no review rows in {review_csv}")
    required = {"asset_id", "location", "manuscript", "checked", "include", "query", "split"}
    if not required.issubset(raw_rows[0]):
        raise ValueError(f"review CSV missing columns {sorted(required - set(raw_rows[0]))}")
    rows = [normalize_review_row(row) for row in raw_rows]
    identifiers = [row["asset_id"] for row in rows]
    if not all(identifiers) or len(identifiers) != len(set(identifiers)):
        raise ValueError("asset_id values must be nonblank and unique")
    labeled: dict[str, list[dict[str, Any]]] = {split: [] for split in LABELED_SPLITS}
    unlabeled: list[dict[str, Any]] = []
    manuscript_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        image_path = safe_join(asset_root, row["location"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        example = {
            "image": str(image_path),
            "asset_id": row["asset_id"],
            "label": int(row["include"]),
            "include": bool(row["include"]),
            "filename": row["filename"],
            "page": row["page"],
            "manuscript": row["manuscript"],
            "location": row["location"],
            "notes": row["notes"],
        }
        if row["checked"] and not row["query"]:
            split = str(row["split"]).strip().lower()
            if split not in LABELED_SPLITS:
                raise ValueError(f"labeled asset {row['asset_id']} requires an explicit train/validation/test split")
            if not str(row["manuscript"]).strip():
                raise ValueError(f"labeled asset {row['asset_id']} has no manuscript group")
            manuscript_splits[str(row["manuscript"])].add(split)
            labeled[split].append(example)
        else:
            example["label"] = -1
            unlabeled.append(example)
    overlaps = {manuscript: splits for manuscript, splits in manuscript_splits.items() if len(splits) > 1}
    if overlaps:
        raise ValueError(f"manuscripts cross classifier splits: {overlaps}")
    for split, examples in labeled.items():
        if not examples:
            raise ValueError(f"classifier split {split!r} is empty")
    if {example["label"] for example in labeled["train"]} != {0, 1}:
        raise ValueError("classifier train split must contain both include and exclude labels")
    return labeled, unlabeled


def _features(labeled: bool) -> Features:
    return Features({
        "image": Image(),
        "asset_id": Value("string"),
        "label": ClassLabel(names=list(LABEL_NAMES)) if labeled else Value("int64"),
        "include": Value("bool"),
        "filename": Value("string"),
        "page": Value("string"),
        "manuscript": Value("string"),
        "location": Value("string"),
        "notes": Value("string"),
    })


def build_classifier_dataset(review_csv: Path, asset_root: Path, output_dir: Path) -> DatasetDict:
    labeled, unlabeled = validate_and_partition(review_csv, asset_root)
    datasets = DatasetDict({
        **{split: Dataset.from_list(examples, features=_features(True)) for split, examples in labeled.items()},
        "unlabeled": Dataset.from_list(unlabeled, features=_features(False)),
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets.save_to_disk(str(output_dir))
    info = {
        "review_csv": str(review_csv),
        "label_names": list(LABEL_NAMES),
        "splits": {name: len(dataset) for name, dataset in datasets.items()},
        "queried_rows_are_unlabeled": True,
    }
    (output_dir / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return datasets
