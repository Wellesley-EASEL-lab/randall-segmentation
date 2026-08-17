#!/usr/bin/env python3
"""Evaluate a trained include classifier on a saved Hugging Face dataset split."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from transformers import AutoImageProcessor

from randall_pipeline.classifier_data import IncludeCollator, load_datasets, make_processor
from randall_pipeline.classifier_model import load_classifier, load_model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the saved DatasetDict from build_classifier_dataset.py.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint directory written by train_classifier.py.",
    )
    parser.add_argument("--split", default="test", help="Dataset split to evaluate.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Optional CSV path for per-image predictions.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Optional JSON path for metrics. Defaults to <checkpoint>/<split>_metrics.json.",
    )
    parser.add_argument(
        "--by-manuscript",
        action="store_true",
        help="Include per-manuscript metrics in the JSON output.",
    )
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda or cpu.")
    return parser.parse_args()


def prediction_metrics(labels: list[int], predictions: list[int], probs: list[float]) -> dict[str, Any]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }
    try:
        metrics["auroc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        metrics["auroc"] = None
    return metrics


def iter_batches(dataset: Any, batch_size: int):
    for start in range(0, len(dataset), batch_size):
        examples = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
        yield examples


def load_processor(checkpoint: Path, model_name: str):
    try:
        return AutoImageProcessor.from_pretrained(checkpoint)
    except OSError:
        return make_processor(model_name)


def main() -> None:
    args = parse_args()
    datasets = load_datasets(args.dataset)
    if args.split not in datasets:
        raise ValueError(f"Dataset has no split {args.split!r}; available splits: {list(datasets)}")

    config = load_model_config(args.checkpoint)
    processor = load_processor(args.checkpoint, config["model_name"])
    collator = IncludeCollator(processor)

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    model = load_classifier(args.checkpoint, map_location=device)
    model.to(device)
    model.eval()

    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[float] = []
    rows: list[dict[str, Any]] = []

    split_dataset = datasets[args.split]
    with torch.no_grad():
        for examples in iter_batches(split_dataset, args.batch_size):
            batch = collator(examples)
            pixel_values = batch["pixel_values"].to(device)
            logits = model(pixel_values=pixel_values).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            preds = np.argmax(logits.detach().cpu().numpy(), axis=-1)

            for example, prob_include, predicted_label in zip(examples, probs, preds):
                true_label = int(example["label"])
                labels.append(true_label)
                predictions.append(int(predicted_label))
                probabilities.append(float(prob_include))
                rows.append(
                    {
                        "filename": example["filename"],
                        "manuscript": example["manuscript"],
                        "location": example["location"],
                        "true_label": true_label,
                        "predicted_label": int(predicted_label),
                        "prob_include": float(prob_include),
                    }
                )

    metrics = prediction_metrics(labels, predictions, probabilities)
    if args.by_manuscript:
        grouped: dict[str, dict[str, list[Any]]] = defaultdict(
            lambda: {"labels": [], "predictions": [], "probabilities": []}
        )
        for row in rows:
            group = grouped[row["manuscript"]]
            group["labels"].append(row["true_label"])
            group["predictions"].append(row["predicted_label"])
            group["probabilities"].append(row["prob_include"])
        metrics["by_manuscript"] = {
            manuscript: prediction_metrics(
                values["labels"], values["predictions"], values["probabilities"]
            )
            for manuscript, values in sorted(grouped.items())
        }

    metrics_output = args.metrics_output or (args.checkpoint / f"{args.split}_metrics.json")
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    if args.predictions is not None:
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(json.dumps(metrics, indent=2))
    print(f"Wrote metrics to {metrics_output}")


if __name__ == "__main__":
    main()
