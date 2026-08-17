"""The shared 3-by-2 tiling, padding, and exact reassembly contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class TileInfo:
    tile_id: int
    row: int
    col: int
    left: int
    top: int
    right: int
    bottom: int
    pad_left: int
    pad_right: int
    pad_top: int
    pad_bottom: int
    original_width: int
    original_height: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def tile_bounds(width: int, height: int) -> list[tuple[int, int, int, int, int, int, int]]:
    """Return six lossless row-major bounds, assigning remainders to the last cells."""
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image dimensions: {width}x{height}")
    tile_w, tile_h = width // 2, height // 3
    if tile_w == 0 or tile_h == 0:
        raise ValueError("an image must be at least 2x3 pixels for 3x2 tiling")
    result = []
    for row in range(3):
        for col in range(2):
            left, top = col * tile_w, row * tile_h
            right = (col + 1) * tile_w if col == 0 else width
            bottom = (row + 1) * tile_h if row < 2 else height
            result.append((row * 2 + col, row, col, left, top, right, bottom))
    return result


def pad_to_square(image: Image.Image, fill: int | tuple[int, ...] = 0) -> tuple[Image.Image, tuple[int, int, int, int]]:
    width, height = image.size
    size = max(width, height)
    left = (size - width) // 2
    right = size - width - left
    top = (size - height) // 2
    bottom = size - height - top
    padded = Image.new(image.mode, (size, size), fill)
    padded.paste(image, (left, top))
    return padded, (left, right, top, bottom)


def prepare_tiles(
    image: Image.Image,
    *,
    output_size: int = 1024,
    threshold: int = 1024,
    resample: Image.Resampling = Image.Resampling.LANCZOS,
) -> list[tuple[Image.Image, TileInfo]]:
    width, height = image.size
    bounds = tile_bounds(width, height) if max(width, height) > threshold else [(-1, 0, 0, 0, 0, width, height)]
    result = []
    for tile_id, row, col, left, top, right, bottom in bounds:
        crop = image.crop((left, top, right, bottom))
        padded, pads = pad_to_square(crop)
        resized = padded.resize((output_size, output_size), resample)
        result.append((resized, TileInfo(tile_id, row, col, left, top, right, bottom, *pads, width, height)))
    return result


def unpad_prediction(prediction: np.ndarray, info: TileInfo) -> np.ndarray:
    """Restore one square model prediction to its exact pre-padding tile size."""
    if prediction.ndim != 2:
        raise ValueError(f"prediction must be 2-D, got {prediction.shape}")
    crop_w, crop_h = info.right - info.left, info.bottom - info.top
    square_size = max(crop_w, crop_h)
    restored = Image.fromarray(prediction).resize((square_size, square_size), Image.Resampling.NEAREST)
    array = np.asarray(restored)
    y2 = square_size - info.pad_bottom
    x2 = square_size - info.pad_right
    return array[info.pad_top:y2, info.pad_left:x2]


def retile(predictions: Iterable[tuple[np.ndarray, TileInfo]], width: int, height: int) -> np.ndarray:
    items = list(predictions)
    if not items:
        raise ValueError("no tile predictions supplied")
    dtype = items[0][0].dtype
    output = np.zeros((height, width), dtype=dtype)
    seen: set[int] = set()
    for prediction, info in items:
        if info.tile_id in seen:
            raise ValueError(f"duplicate tile id {info.tile_id}")
        seen.add(info.tile_id)
        restored = unpad_prediction(prediction, info)
        expected = (info.bottom - info.top, info.right - info.left)
        if restored.shape != expected:
            raise ValueError(f"restored tile {info.tile_id} is {restored.shape}, expected {expected}")
        output[info.top:info.bottom, info.left:info.right] = restored
    expected_ids = {-1} if len(items) == 1 and items[0][1].tile_id == -1 else set(range(6))
    if seen != expected_ids:
        raise ValueError(f"incomplete tile set: got {sorted(seen)}, expected {sorted(expected_ids)}")
    return output
