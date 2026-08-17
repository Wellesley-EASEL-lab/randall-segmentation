#!/usr/bin/env python3
"""Evaluate the hosted Randall model on a supported Hugging Face image/mask dataset."""

from __future__ import annotations

import argparse
import csv
import io
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from tqdm.auto import tqdm

from randall_pipeline.constants import ID2LABEL, OPEN_SEGMENTATIONS, SEGMENTATION_MODEL
from randall_pipeline.datasets import iter_records, resolved_revision
from randall_pipeline.inference import load_segmentation_model, predict_image
from randall_pipeline.io import write_json_atomic
from randall_pipeline.metrics import confusion_matrix, segmentation_metrics
from randall_pipeline.tiling import TileInfo, retile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-repo", default=OPEN_SEGMENTATIONS)
    parser.add_argument("--prepared-root", type=Path, help="Prepared directory written by preprocess_segmentation_data.py; takes precedence over --dataset-repo")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--split", default="test")
    parser.add_argument("--image-manifest", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--model-repo", default=SEGMENTATION_MODEL)
    parser.add_argument("--model-revision")
    parser.add_argument("--adapter-checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--id2label", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--device")
    parser.add_argument("--write-parquet", action="store_true")
    return parser.parse_args()


def png_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array.astype(np.uint8), mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def prepared_pages(root: Path, split: str):
    manifest = root / "retiling.csv"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    if not rows:
        raise ValueError(f"prepared root contains no {split!r} rows")
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["asset_id"]].append(row)
    fields = tuple(TileInfo.__dataclass_fields__)
    for asset_id, page_rows in groups.items():
        items = []
        for row in sorted(page_rows, key=lambda value: int(value["tile_id"])):
            filename = row["tile_filename"]
            image_path = root / split / filename
            mask_path = root / "annotation" / split / filename
            if not image_path.is_file() or not mask_path.is_file():
                raise FileNotFoundError(f"missing prepared image/mask pair for {filename}")
            with Image.open(image_path) as image_handle, Image.open(mask_path) as mask_handle:
                image = image_handle.convert("RGB").copy()
                mask = np.asarray(mask_handle).astype(np.uint8)
            info = TileInfo(**{field: int(row[field]) for field in fields})
            items.append((image, mask, info))
        yield asset_id, page_rows[0]["name"], items


def main() -> None:
    args = parse_args()
    dataset_revision = None if args.prepared_root else resolved_revision(args.dataset_repo, args.dataset_revision)
    model, config, labels = load_segmentation_model(
        base_checkpoint=args.base_checkpoint, model_repo=args.model_repo,
        revision=args.model_revision, adapter_checkpoint=args.adapter_checkpoint,
        config_path=args.config, id2label_path=args.id2label, device=args.device,
    )
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    parquet_rows = []
    if args.prepared_root:
        page_source = prepared_pages(args.prepared_root, args.split)
    else:
        records = iter_records(
            args.dataset_repo, split=args.split, revision=dataset_revision,
            image_manifest=args.image_manifest, cache_dir=args.cache_dir,
        )
        page_source = ((record.asset_id, record.name, [(record.image, np.asarray(record.mask), None)]) for record in records)
    for asset_id, name, items in tqdm(page_source, desc="Evaluating pages", unit="page"):
        if items[0][2] is None:
            image, target, _ = items[0]
            if target.ndim != 2:
                raise ValueError("evaluation requires class-id ground-truth masks")
            prediction = predict_image(
                model, image, image_size=int(config["preprocessing"]["image_size"]),
                batch_size=args.batch_size,
            )
        else:
            predicted_tiles = []
            target_tiles = []
            for image, target_tile, info in items:
                if image.size != (int(config["preprocessing"]["image_size"]),) * 2:
                    raise ValueError(f"prepared tile {asset_id} has incompatible dimensions {image.size}")
                predicted_tile = predict_image(
                    model, image, image_size=image.width, tile_threshold=image.width,
                    batch_size=args.batch_size,
                )
                predicted_tiles.append((predicted_tile, info))
                target_tiles.append((target_tile, info))
            width, height = items[0][2].original_width, items[0][2].original_height
            prediction = retile(predicted_tiles, width, height)
            target = retile(target_tiles, width, height)
        targets.append(target.astype(np.uint8))
        predictions.append(prediction)
        if args.write_parquet:
            parquet_rows.append({
                "asset_id": asset_id, "name": name,
                "target_png": png_bytes(target), "prediction_png": png_bytes(prediction),
            })
    matrix = confusion_matrix(targets, predictions, int(config["num_classes"]))
    metrics = segmentation_metrics(matrix)
    metrics["dataset_repo"] = args.dataset_repo if not args.prepared_root else None
    metrics["prepared_root"] = str(args.prepared_root.resolve()) if args.prepared_root else None
    metrics["dataset_revision"] = dataset_revision
    metrics["model_repo"] = args.model_repo
    metrics["model_revision"] = args.model_revision
    metrics["pages"] = len(targets)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_dir / "metrics.json", metrics)
    if args.write_parquet:
        pd.DataFrame(parquet_rows).to_parquet(args.output_dir / "predictions.parquet", index=False)
    figure = plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=[labels.get(i, ID2LABEL.get(i, str(i))) for i in range(len(matrix))],
                yticklabels=[labels.get(i, ID2LABEL.get(i, str(i))) for i in range(len(matrix))])
    plt.xlabel("Predicted")
    plt.ylabel("Ground truth")
    plt.tight_layout()
    figure.savefig(args.output_dir / "confusion_matrix.png", dpi=150)
    plt.close(figure)
    print(f"Evaluated {len(targets)} pages; metrics are in {args.output_dir}")


if __name__ == "__main__":
    main()
