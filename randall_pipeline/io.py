"""Portable identifiers, manifests, masks, and safe filesystem helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from .constants import PALETTE, REVIEW_FIELDS

TRUE_VALUES = {"1", "true", "t", "yes", "y"}


def parse_bool(value: Any) -> bool:
    return value if isinstance(value, bool) else str(value or "").strip().lower() in TRUE_VALUES


def stable_id(*parts: object) -> str:
    normalized = "\x1f".join(str(part).strip() for part in parts)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def safe_join(root: Path, relative: str | Path) -> Path:
    root = root.expanduser().resolve()
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"path escapes declared root: {relative}")
    return candidate


def relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_atomic(path: Path, rows: Iterable[dict[str, Any]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def class_mask(mask: Image.Image | np.ndarray, class_id: int) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 2:
        return array.astype(np.int64) == class_id
    if array.ndim == 3 and array.shape[2] in {3, 4}:
        target = np.asarray(PALETTE[class_id], dtype=array.dtype)
        return np.all(array[..., :3] == target, axis=-1)
    raise ValueError(f"unsupported segmentation mask shape {array.shape}")


def colorize(mask: np.ndarray) -> Image.Image:
    if mask.ndim != 2:
        raise ValueError("class-id mask must be two-dimensional")
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in PALETTE.items():
        rgb[mask == class_id] = color
    return Image.fromarray(rgb, mode="RGB")


def normalize_review_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {field: row.get(field, "") for field in REVIEW_FIELDS}
    for field in ("checked", "include", "query"):
        normalized[field] = parse_bool(normalized[field])
    normalized["notes"] = str(normalized["notes"] or "")
    return normalized

