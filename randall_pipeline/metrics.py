"""Transparent semantic-segmentation metrics computed from a confusion matrix."""

from __future__ import annotations

import numpy as np

from .constants import ID2LABEL


def confusion_matrix(targets: list[np.ndarray], predictions: list[np.ndarray], num_classes: int = 8) -> np.ndarray:
    if len(targets) != len(predictions):
        raise ValueError("target and prediction counts differ")
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        if target.shape != prediction.shape:
            raise ValueError(f"shape mismatch: {target.shape} != {prediction.shape}")
        valid = (target >= 0) & (target < num_classes) & (prediction >= 0) & (prediction < num_classes)
        indices = num_classes * target[valid].astype(np.int64) + prediction[valid].astype(np.int64)
        matrix += np.bincount(indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return matrix


def segmentation_metrics(matrix: np.ndarray) -> dict[str, object]:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("confusion matrix must be square")
    tp = np.diag(matrix).astype(float)
    support = matrix.sum(axis=1).astype(float)
    predicted = matrix.sum(axis=0).astype(float)
    union = support + predicted - tp
    iou = np.divide(tp, union, out=np.zeros_like(tp), where=union != 0)
    dice_den = support + predicted
    dice = np.divide(2 * tp, dice_den, out=np.zeros_like(tp), where=dice_den != 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support != 0)
    total = matrix.sum()
    labels = {i: ID2LABEL.get(i, f"class_{i}") for i in range(len(tp))}
    per_class = {
        labels[i]: {"iou": float(iou[i]), "dice": float(dice[i]), "precision": float(precision[i]), "recall": float(recall[i]), "support": int(support[i])}
        for i in range(len(tp))
    }
    foreground_support = support[1:].sum()
    return {
        "overall_accuracy": float(tp.sum() / total) if total else 0.0,
        "foreground_accuracy": float(tp[1:].sum() / foreground_support) if foreground_support else 0.0,
        "foreground_mean_iou": float(iou[1:].mean()) if len(iou) > 1 else 0.0,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }
