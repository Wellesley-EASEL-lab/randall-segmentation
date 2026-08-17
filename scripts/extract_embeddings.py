#!/usr/bin/env python3
"""Extract configured classifier representations for labeled and unlabeled marginals."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoImageProcessor

from randall_pipeline.classifier_data import IncludeCollator, load_datasets, make_processor
from randall_pipeline.classifier_model import load_classifier, load_model_config
from randall_pipeline.io import write_csv_atomic, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    datasets = load_datasets(args.dataset)
    config = load_model_config(args.checkpoint)
    try:
        processor = AutoImageProcessor.from_pretrained(args.checkpoint)
    except OSError:
        processor = make_processor(config["model_name"])
    collator = IncludeCollator(processor)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_classifier(args.checkpoint, map_location=device)
    model.to(device).eval()
    chunks: list[np.ndarray] = []
    rows = []
    for split, dataset in datasets.items():
        for start in tqdm(range(0, len(dataset), args.batch_size), desc=f"Embedding {split}", unit="batch"):
            examples = [dataset[index] for index in range(start, min(start + args.batch_size, len(dataset)))]
            batch = collator(examples)
            with torch.inference_mode():
                features = model.extract_features(batch["pixel_values"].to(device))
            chunks.append(features.cpu().numpy())
            for example in examples:
                label_value = int(example["label"])
                rows.append({
                    "asset_id": example["asset_id"],
                    "split": split,
                    "filename": example["filename"],
                    "page": example["page"],
                    "manuscript": example["manuscript"],
                    "location": example["location"],
                    "true_label": label_value if split != "unlabeled" else "",
                })
    if not chunks:
        raise ValueError("dataset contains no examples")
    embeddings = np.concatenate(chunks, axis=0)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_prefix.with_suffix(".npy"), embeddings)
    write_csv_atomic(
        args.output_prefix.with_suffix(".csv"), rows,
        ("asset_id", "split", "filename", "page", "manuscript", "location", "true_label"),
    )
    write_json_atomic(args.output_prefix.with_suffix(".json"), {
        "dataset": str(args.dataset),
        "checkpoint": str(args.checkpoint),
        "model_name": config["model_name"],
        "pooling": config.get("pooling", "cls"),
        "rows": len(rows),
        "embedding_dim": int(embeddings.shape[1]),
    })
    print(f"Wrote {embeddings.shape} embeddings to {args.output_prefix.with_suffix('.npy')}")


if __name__ == "__main__":
    main()
