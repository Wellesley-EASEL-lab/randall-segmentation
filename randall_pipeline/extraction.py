"""Connected-component marginal extraction and canonical review inventories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label

from .constants import REVIEW_FIELDS
from .datasets import SegmentationRecord
from .io import class_mask, relative_or_absolute, stable_id, write_csv_atomic


Box = tuple[int, int, int, int]


def _overlap(a: Box, b: Box, tolerance: int = 0) -> bool:
    return a[0] - tolerance <= b[2] and a[2] + tolerance >= b[0] and a[1] - tolerance <= b[3] and a[3] + tolerance >= b[1]


def merge_boxes(boxes: list[Box], tolerance: int = 0) -> list[Box]:
    pending = list(boxes)
    changed = True
    while changed:
        changed = False
        result: list[Box] = []
        for box in pending:
            for index, existing in enumerate(result):
                if _overlap(existing, box, tolerance):
                    result[index] = (
                        min(existing[0], box[0]), min(existing[1], box[1]),
                        max(existing[2], box[2]), max(existing[3], box[3]),
                    )
                    changed = True
                    break
            else:
                result.append(box)
        pending = result
    return pending


def component_boxes(mask: np.ndarray, *, min_pixels: int = 0) -> list[Box]:
    if mask.ndim != 2:
        raise ValueError("component mask must be two-dimensional")
    components, count = label(mask)
    boxes: list[Box] = []
    for component_id in range(1, count + 1):
        ys, xs = np.where(components == component_id)
        if len(xs) < min_pixels:
            continue
        boxes.append((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    return boxes


def add_margin(box: Box, width: int, height: int, fraction: float) -> Box:
    if fraction < 0:
        raise ValueError("margin must be nonnegative")
    amount = int(round(max(width, height) * fraction))
    return max(0, box[0] - amount), max(0, box[1] - amount), min(width, box[2] + amount), min(height, box[3] + amount)


def extract_record(
    record: SegmentationRecord,
    output_root: Path,
    *,
    class_id: int = 2,
    margin: float = 0.05,
    min_fraction: float = 0.005,
    merge: bool = False,
    merge_tolerance: float = 0.0,
    draw_bbox: bool = False,
    materialize_inputs: bool = False,
) -> list[dict[str, object]]:
    image = record.image.convert("RGB")
    if image.size != record.mask.size:
        raise ValueError(f"{record.asset_id}: image/mask dimensions differ")
    width, height = image.size
    binary = class_mask(record.mask, class_id)
    boxes = component_boxes(binary, min_pixels=int(width * height * min_fraction))
    if merge:
        boxes = merge_boxes(boxes, int(round(max(width, height) * merge_tolerance)))
    output_root.mkdir(parents=True, exist_ok=True)
    source_path = Path(record.source_path) if record.source_path else None
    segmentation_path: Path | None = None
    if materialize_inputs or source_path is None:
        source_path = output_root / "sources" / f"{record.asset_id}.png"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(source_path)
        segmentation_path = output_root / "segmentations" / f"{record.asset_id}.png"
        segmentation_path.parent.mkdir(parents=True, exist_ok=True)
        record.mask.save(segmentation_path)
    crop_dir = output_root / ("marginals-bbox" if draw_bbox else "marginals")
    crop_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for region_index, tight in enumerate(boxes):
        expanded = add_margin(tight, width, height, margin)
        crop = image.crop(expanded)
        if draw_bbox:
            draw = ImageDraw.Draw(crop)
            draw.rectangle((tight[0] - expanded[0], tight[1] - expanded[1], tight[2] - expanded[0] - 1, tight[3] - expanded[1] - 1), outline="red", width=2)
        asset_id = stable_id(record.asset_id, class_id, region_index, *tight)
        filename = f"{asset_id}.png"
        crop_path = crop_dir / filename
        crop.save(crop_path)
        rows.append({
            "asset_id": asset_id,
            "filename": filename,
            "page": record.page or record.name,
            "manuscript": record.manuscript,
            "archive": record.archive,
            "location": relative_or_absolute(crop_path, output_root),
            "source_path": relative_or_absolute(source_path, output_root) if source_path else "",
            "segmentation_path": relative_or_absolute(segmentation_path, output_root) if segmentation_path else "",
            "split": record.split,
            "checked": False,
            "include": True,
            "query": False,
            "notes": "",
        })
    return rows


def write_inventory(output_root: Path, rows: list[dict[str, object]], name: str = "marginal_inventory.csv") -> Path:
    path = output_root / name
    identifiers = [str(row["asset_id"]) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate asset_id in marginal inventory")
    write_csv_atomic(path, rows, REVIEW_FIELDS)
    return path


def records_from_predictions(predictions_jsonl: Path, *, split: str = "") -> Iterable[SegmentationRecord]:
    base = predictions_jsonl.parent.resolve()
    with predictions_jsonl.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                source = Path(row["input_image"])
                segmentation = Path(row["segmentation_map"])
            except KeyError as exc:
                raise ValueError(f"{predictions_jsonl}:{line_number} missing {exc.args[0]}") from exc
            if not source.is_absolute():
                source = base / source
            if not segmentation.is_absolute():
                segmentation = base / segmentation
            if not source.is_file() or not segmentation.is_file():
                raise FileNotFoundError(f"prediction row paths do not exist: {source}, {segmentation}")
            with Image.open(source) as image_handle, Image.open(segmentation) as mask_handle:
                image, mask = image_handle.convert("RGB").copy(), mask_handle.copy()
            identity = str(row.get("asset_id") or stable_id(source.as_posix()))
            yield SegmentationRecord(
                asset_id=identity, image=image, mask=mask, split=str(row.get("split", split)),
                name=source.stem, manuscript=str(row.get("manuscript", "")),
                archive=str(row.get("archive", "")), page=str(row.get("page", source.stem)),
                source_path=str(source),
            )
