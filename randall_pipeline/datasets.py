"""Adapters for the three public Randall Hugging Face dataset contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from PIL import Image

from .constants import OPEN_SEGMENTATIONS, PREDICTED_SEGMENTATIONS, SEGMENTATION_MASKS
from .io import read_csv, stable_id


@dataclass
class SegmentationRecord:
    asset_id: str
    image: Image.Image
    mask: Image.Image
    split: str
    name: str
    uuid: str = ""
    manuscript: str = ""
    archive: str = ""
    page: str = ""
    source_path: str = ""
    source_image_credit: str = ""


def _pil(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.copy()
    if isinstance(value, np.ndarray):
        return Image.fromarray(value)
    if isinstance(value, dict) and value.get("path"):
        with Image.open(value["path"]) as image:
            return image.copy()
    if isinstance(value, (str, Path)):
        with Image.open(value) as image:
            return image.copy()
    raise TypeError(f"cannot convert {type(value)!r} to an image")


def _manifest_index(path: Path | None, keys: tuple[str, ...]) -> dict[tuple[str, ...], Path]:
    if path is None:
        raise ValueError(f"--image-manifest is required with columns {', '.join((*keys, 'image_path'))}")
    rows = read_csv(path)
    required = set(keys) | {"image_path"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} must contain columns {sorted(required)}")
    result: dict[tuple[str, ...], Path] = {}
    for row in rows:
        key = tuple(str(row[field]).strip() for field in keys)
        if not all(key):
            raise ValueError(f"blank manifest key in {row}")
        if key in result:
            raise ValueError(f"duplicate manifest key {key}")
        image_path = Path(row["image_path"]).expanduser()
        if not image_path.is_absolute():
            image_path = path.parent / image_path
        image_path = image_path.resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        result[key] = image_path
    return result


def load_hub_split(repo_id: str, split: str, revision: str | None = None, cache_dir: Path | None = None) -> Dataset:
    dataset = load_dataset(repo_id, split=split, revision=revision, cache_dir=str(cache_dir) if cache_dir else None)
    if not isinstance(dataset, Dataset):
        raise TypeError(f"expected Dataset for {repo_id}:{split}, got {type(dataset)!r}")
    return dataset


def resolved_revision(repo_id: str, revision: str | None = None) -> str:
    return HfApi().dataset_info(repo_id, revision=revision).sha


def iter_records(
    repo_id: str,
    *,
    split: str = "train",
    revision: str | None = None,
    image_manifest: Path | None = None,
    cache_dir: Path | None = None,
    dataset: Dataset | None = None,
) -> Iterator[SegmentationRecord]:
    """Yield normalized image/mask pairs, requiring explicit joins when images are omitted."""
    source = dataset if dataset is not None else load_hub_split(repo_id, split, revision, cache_dir)
    columns = set(source.column_names)
    if repo_id == OPEN_SEGMENTATIONS or {"pixel_values", "labels", "uuid"}.issubset(columns):
        required = {"pixel_values", "labels", "uuid", "name"}
        if not required.issubset(columns):
            raise ValueError(f"open dataset missing columns {sorted(required - columns)}")
        for row in source:
            image, mask = _pil(row["pixel_values"]).convert("RGB"), _pil(row["labels"])
            if image.size != mask.size:
                raise ValueError(f"{row['uuid']} image/mask dimensions differ: {image.size} != {mask.size}")
            yield SegmentationRecord(
                asset_id=stable_id(repo_id, split, row["uuid"]),
                image=image, mask=mask, split=split, name=str(row["name"]),
                uuid=str(row["uuid"]), page=str(row.get("name", "")),
                source_image_credit=str(row.get("source_image_credit", "")),
            )
        return

    if repo_id == SEGMENTATION_MASKS or {"labels", "uuid"}.issubset(columns):
        required = {"labels", "uuid", "name"}
        if not required.issubset(columns):
            raise ValueError(f"masks dataset missing columns {sorted(required - columns)}")
        index = _manifest_index(image_manifest, ("uuid",))
        seen: set[tuple[str, ...]] = set()
        for row in source:
            key = (str(row["uuid"]).strip(),)
            if key not in index:
                raise KeyError(f"no source image for dataset uuid {key[0]}")
            seen.add(key)
            image, mask = _pil(index[key]).convert("RGB"), _pil(row["labels"])
            if image.size != mask.size:
                raise ValueError(f"{key[0]} image/mask dimensions differ: {image.size} != {mask.size}")
            yield SegmentationRecord(
                asset_id=stable_id(repo_id, split, key[0]),
                image=image, mask=mask, split=split, name=str(row["name"]),
                uuid=key[0], page=str(row["name"]), source_path=str(index[key]),
                source_image_credit=str(row.get("source_image_credit", "")),
            )
        extras = set(index) - seen
        if extras:
            raise ValueError(f"manifest has {len(extras)} unmatched uuid row(s)")
        return

    if repo_id == PREDICTED_SEGMENTATIONS or {"manuscript", "archive", "page", "segmentation_map"}.issubset(columns):
        required = {"manuscript", "archive", "page", "segmentation_map"}
        if not required.issubset(columns):
            raise ValueError(f"predicted dataset missing columns {sorted(required - columns)}")
        index = _manifest_index(image_manifest, ("manuscript", "archive", "page"))
        seen: set[tuple[str, ...]] = set()
        for row in source:
            key = tuple(str(row[field]).strip() for field in ("manuscript", "archive", "page"))
            if key not in index:
                raise KeyError(f"no source image for predicted row {key}")
            seen.add(key)
            image, mask = _pil(index[key]).convert("RGB"), _pil(row["segmentation_map"])
            if image.size != mask.size:
                raise ValueError(f"{key} image/mask dimensions differ: {image.size} != {mask.size}")
            yield SegmentationRecord(
                asset_id=stable_id(repo_id, *key),
                image=image, mask=mask, split=split, name="_".join(key),
                manuscript=key[0], archive=key[1], page=key[2],
                source_path=str(index[key]), source_image_credit=str(row.get("source_image_credit", "")),
            )
        extras = set(index) - seen
        if extras:
            raise ValueError(f"manifest has {len(extras)} unmatched predicted row(s)")
        return

    raise ValueError(f"unsupported dataset schema for {repo_id}: {sorted(columns)}")

