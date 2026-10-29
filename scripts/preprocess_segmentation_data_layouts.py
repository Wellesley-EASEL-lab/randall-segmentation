#!/usr/bin/env python3
"""Prepare Randall images (and masks, when available) with layout-specific tiling.

Copy of preprocess_segmentation_data.py with a --layout option:
  page     2 columns x 3 rows when larger than --tile-threshold (original behavior)
  opening  4 columns x 3 rows when larger than --tile-threshold
  crop     never tile; only pad to square and resize
Also accepts --input-dir for local images without masks (e.g. the crops/openings
experiment sets), in which case only image tiles are written.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from randall_pipeline.constants import ID2LABEL, OPEN_SEGMENTATIONS
from randall_pipeline.datasets import iter_records, resolved_revision
from randall_pipeline.io import write_json_atomic
from randall_pipeline.tiling import TileInfo, pad_to_square

# (columns, rows) per layout; None means never tile, only pad + resize
LAYOUTS = {"page": (2, 3), "opening": (4, 3), "crop": None}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
MANIFEST_FIELDS = (
    "asset_id", "name", "split", "layout", "tile_filename", "tile_id", "row", "col",
    "left", "top", "right", "bottom", "pad_left", "pad_right", "pad_top",
    "pad_bottom", "original_width", "original_height",
)


def tile_bounds(width: int, height: int, cols: int, rows: int) -> list[tuple[int, int, int, int, int, int, int]]:
    """Return cols*rows lossless row-major bounds, assigning remainders to the last cells.
    With cols=2, rows=3 this matches randall_pipeline.tiling.tile_bounds exactly."""
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image dimensions: {width}x{height}")
    tile_w, tile_h = width // cols, height // rows
    if tile_w == 0 or tile_h == 0:
        raise ValueError(f"an image must be at least {cols}x{rows} pixels for this layout")
    result = []
    for row in range(rows):
        for col in range(cols):
            left, top = col * tile_w, row * tile_h
            right = (col + 1) * tile_w if col < cols - 1 else width
            bottom = (row + 1) * tile_h if row < rows - 1 else height
            result.append((row * cols + col, row, col, left, top, right, bottom))
    return result


def prepare_tiles(image: Image.Image, *, layout: str, output_size: int, threshold: int,
                  resample: Image.Resampling = Image.Resampling.LANCZOS) -> list[tuple[Image.Image, TileInfo]]:
    width, height = image.size
    grid = LAYOUTS[layout]
    if grid is None or max(width, height) <= threshold:
        bounds = [(-1, 0, 0, 0, 0, width, height)]
    else:
        bounds = tile_bounds(width, height, *grid)
    result = []
    for tile_id, row, col, left, top, right, bottom in bounds:
        crop = image.crop((left, top, right, bottom))
        padded, pads = pad_to_square(crop)
        resized = padded.resize((output_size, output_size), resample)
        result.append((resized, TileInfo(tile_id, row, col, left, top, right, bottom, *pads, width, height)))
    return result


@dataclass
class LocalRecord:
    """An image from a local folder, with no mask."""
    asset_id: str
    name: str
    split: str
    image: Image.Image
    mask: Image.Image | None = None


def iter_local_records(input_dir: Path, split: str):
    paths = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    for path in paths:
        rel = path.relative_to(input_dir).with_suffix("")
        yield LocalRecord(asset_id="__".join(rel.parts), name=path.stem, split=split, image=Image.open(path))


def preprocess(records, output_dir: Path, *, layout: str, output_size: int,
               tile_threshold: int, filter_background: bool) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for record in tqdm(records, desc=f"Preprocessing pages ({layout})", unit="page"):
        image = record.image.convert("RGB")
        image_tiles = prepare_tiles(image, layout=layout, output_size=output_size, threshold=tile_threshold)
        if getattr(record, "mask", None) is not None:
            mask_array = np.asarray(record.mask)
            if mask_array.ndim != 2:
                raise ValueError(f"{record.asset_id}: expected class-id mask, got {mask_array.shape}")
            if mask_array.min() < 0 or mask_array.max() >= len(ID2LABEL):
                raise ValueError(f"{record.asset_id}: mask contains class IDs outside 0..7")
            mask = Image.fromarray(mask_array.astype(np.uint8), mode="L")
            mask_tiles = prepare_tiles(mask, layout=layout, output_size=output_size,
                                       threshold=tile_threshold, resample=Image.Resampling.NEAREST)
        else:
            mask_tiles = [(None, info) for _, info in image_tiles]
        for (image_tile, info), (mask_tile, mask_info) in zip(image_tiles, mask_tiles):
            if info != mask_info:
                raise RuntimeError("image and mask tiling metadata diverged")
            if filter_background and mask_tile is not None and not np.any(np.asarray(mask_tile)):
                continue
            suffix = "full" if info.tile_id == -1 else f"tile-{info.tile_id}"
            filename = f"{record.asset_id}-{suffix}.tif"
            image_dir = output_dir / record.split
            image_dir.mkdir(parents=True, exist_ok=True)
            image_tile.save(image_dir / filename, format="TIFF")
            if mask_tile is not None:
                mask_dir = output_dir / "annotation" / record.split
                mask_dir.mkdir(parents=True, exist_ok=True)
                mask_tile.save(mask_dir / filename, format="TIFF")
            rows.append({"asset_id": record.asset_id, "name": record.name, "split": record.split,
                         "layout": layout, "tile_filename": filename, **info.to_dict()})
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
    (output_dir / "id2label.json").write_text(
        json.dumps({str(k): v for k, v in ID2LABEL.items()}, indent=2) + "\n", encoding="utf-8")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layout", choices=tuple(LAYOUTS), default="page")
    parser.add_argument("--dataset-repo", default=OPEN_SEGMENTATIONS)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--split", help="train/validation/test for Hugging Face data; any label for --input-dir")
    parser.add_argument("--input-dir", type=Path, help="local folder of images without masks (instead of Hugging Face)")
    parser.add_argument("--image-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-size", type=int, default=1024)
    parser.add_argument("--tile-threshold", type=int, default=1024)
    parser.add_argument("--filter-background", action="store_true")
    parser.add_argument("--cache-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input_dir:
        split = args.split or "experiment"
        records = iter_local_records(args.input_dir, split)
        source = {"input_dir": str(args.input_dir)}
    else:
        if args.split not in ("train", "validation", "test"):
            raise SystemExit("--split must be train, validation, or test when reading from Hugging Face")
        split = args.split
        revision = resolved_revision(args.dataset_repo, args.revision)
        records = iter_records(args.dataset_repo, split=split, revision=revision,
                               image_manifest=args.image_manifest, cache_dir=args.cache_dir)
        source = {"dataset_repo": args.dataset_repo, "resolved_revision": revision}
    rows = preprocess(records, args.output_dir, layout=args.layout, output_size=args.output_size,
                      tile_threshold=args.tile_threshold, filter_background=args.filter_background)
    write_json_atomic(args.output_dir / f"preprocessing_run_{split}.json", {
        **source,
        "split": split,
        "layout": args.layout,
        "output_size": args.output_size,
        "tile_threshold": args.tile_threshold,
        "filter_background": args.filter_background,
        "tiles_written": len(rows),
    })
    print(f"Wrote {len(rows)} tiles to {args.output_dir} ({args.layout} layout)")


if __name__ == "__main__":
    main()