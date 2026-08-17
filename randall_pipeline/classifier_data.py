"""Shared dataset and preprocessing helpers for the include classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import DatasetDict, load_from_disk
from PIL import Image, ImageEnhance
from transformers import AutoImageProcessor


def load_datasets(path: str | Path) -> DatasetDict:
    dataset = load_from_disk(str(path))
    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected a DatasetDict at {path}, got {type(dataset)!r}")

    missing_splits = {"train", "validation", "test"} - set(dataset.keys())
    if missing_splits:
        raise ValueError(f"Dataset is missing required splits: {sorted(missing_splits)}")
    return dataset


def make_processor(model_name: str, image_size: int | None = None) -> Any:
    processor = AutoImageProcessor.from_pretrained(model_name)
    if image_size is not None:
        processor.size = {"height": image_size, "width": image_size}
    processor.do_resize = False
    processor.default_to_square = False
    return processor


def processor_image_size(processor: Any) -> int:
    size = getattr(processor, "size", None)
    if isinstance(size, dict):
        height = size.get("height")
        width = size.get("width")
        if height is not None and width is not None:
            if height != width:
                raise ValueError(f"Expected square processor size, got {size}")
            return int(height)
        if "shortest_edge" in size:
            return int(size["shortest_edge"])
    if isinstance(size, int):
        return size
    raise ValueError(f"Could not determine image size from processor size {size!r}")


def resize_longest_side_and_pad(image: Image.Image, output_size: int) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Cannot preprocess image with invalid size {image.size}")

    scale = output_size / max(width, height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

    padded = Image.new(image.mode, (output_size, output_size), 0)
    left = (output_size - resized_width) // 2
    top = (output_size - resized_height) // 2
    padded.paste(resized, (left, top))
    return padded


def sample_jitter_factor(amount: float) -> float:
    if amount <= 0:
        return 1.0
    lower = max(0.0, 1.0 - amount)
    upper = 1.0 + amount
    return float(torch.empty((), dtype=torch.float32).uniform_(lower, upper).item())


def apply_color_jitter(
    image: Image.Image,
    *,
    brightness: float = 0.0,
    contrast: float = 0.0,
) -> Image.Image:
    brightness_factor = sample_jitter_factor(brightness)
    contrast_factor = sample_jitter_factor(contrast)
    if brightness_factor != 1.0:
        image = ImageEnhance.Brightness(image).enhance(brightness_factor)
    if contrast_factor != 1.0:
        image = ImageEnhance.Contrast(image).enhance(contrast_factor)
    return image


@dataclass
class IncludeCollator:
    processor: Any
    brightness_jitter: float = 0.0
    contrast_jitter: float = 0.0

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        image_size = processor_image_size(self.processor)
        images = []
        for example in examples:
            image = example["image"].convert("RGB")
            image = apply_color_jitter(
                image,
                brightness=self.brightness_jitter,
                contrast=self.contrast_jitter,
            )
            images.append(resize_longest_side_and_pad(image, image_size))
        labels = torch.tensor([example["label"] for example in examples], dtype=torch.long)
        batch = self.processor(images=images, return_tensors="pt")
        batch["labels"] = labels
        return batch
