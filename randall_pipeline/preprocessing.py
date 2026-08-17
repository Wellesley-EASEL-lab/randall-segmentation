"""Materialize normalized MapSAM training tiles and exact retiling metadata."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from .constants import ID2LABEL
from .datasets import SegmentationRecord
from .tiling import prepare_tiles


MANIFEST_FIELDS = (
    "asset_id", "name", "split", "tile_filename", "tile_id", "row", "col",
    "left", "top", "right", "bottom", "pad_left", "pad_right", "pad_top",
    "pad_bottom", "original_width", "original_height",
)


def preprocess_records(
    records: Iterable[SegmentationRecord],
    output_dir: Path,
    *,
    output_size: int = 1024,
    tile_threshold: int = 1024,
    filter_background: bool = False,
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for record in tqdm(records, desc="Preprocessing pages", unit="page"):
        image = record.image.convert("RGB")
        mask_array = np.asarray(record.mask)
        if mask_array.ndim != 2:
            raise ValueError(f"{record.asset_id}: expected class-id mask, got {mask_array.shape}")
        if mask_array.min() < 0 or mask_array.max() >= len(ID2LABEL):
            raise ValueError(f"{record.asset_id}: mask contains class IDs outside 0..7")
        mask = Image.fromarray(mask_array.astype(np.uint8), mode="L")
        image_tiles = prepare_tiles(image, output_size=output_size, threshold=tile_threshold)
        mask_tiles = prepare_tiles(mask, output_size=output_size, threshold=tile_threshold, resample=Image.Resampling.NEAREST)
        for (image_tile, info), (mask_tile, mask_info) in zip(image_tiles, mask_tiles):
            if info != mask_info:
                raise RuntimeError("image and mask tiling metadata diverged")
            tile_mask = np.asarray(mask_tile)
            if filter_background and not np.any(tile_mask):
                continue
            suffix = "full" if info.tile_id == -1 else f"tile-{info.tile_id}"
            filename = f"{record.asset_id}-{suffix}.tif"
            image_dir = output_dir / record.split
            mask_dir = output_dir / "annotation" / record.split
            image_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)
            image_tile.save(image_dir / filename, format="TIFF")
            mask_tile.save(mask_dir / filename, format="TIFF")
            rows.append({"asset_id": record.asset_id, "name": record.name, "split": record.split, "tile_filename": filename, **info.to_dict()})
    manifest = output_dir / "retiling.csv"
    existing: list[dict[str, object]] = []
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    combined = existing + rows
    keys = [(row["split"], row["tile_filename"]) for row in combined]
    if len(keys) != len(set(keys)):
        raise ValueError("preprocessing would create duplicate split/tile rows")
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(combined)
    (output_dir / "id2label.json").write_text(json.dumps({str(k): v for k, v in ID2LABEL.items()}, indent=2) + "\n", encoding="utf-8")
    return rows
