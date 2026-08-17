"""Hosted Randall adapter loading and shared full-page inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from segment_anything import sam_model_registry

from .constants import SEGMENTATION_MODEL
from .segmentation_model import DoRA_Sam
from .tiling import prepare_tiles, retile


def validate_adapter_config(config: dict[str, Any]) -> None:
    expected = {
        "model_type": "mapsam-dora-sam-vit-l",
        "standalone": False,
        "num_classes": 8,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"incompatible hosted adapter {key}: expected {value!r}, got {config.get(key)!r}")
    if int(config.get("adapter", {}).get("rank", -1)) != 8:
        raise ValueError("incompatible hosted adapter rank: expected 8")
    if str(config.get("adapter", {}).get("spgen_variant", "")) != "depthwise":
        raise ValueError("incompatible hosted self-prompt generator: expected depthwise")
    if int(config.get("preprocessing", {}).get("image_size", -1)) != 1024:
        raise ValueError("incompatible hosted image size: expected 1024")


def resolve_adapter_files(
    *,
    model_repo: str = SEGMENTATION_MODEL,
    revision: str | None = None,
    adapter_checkpoint: Path | None = None,
    config_path: Path | None = None,
    id2label_path: Path | None = None,
) -> tuple[Path, Path, Path]:
    adapter = adapter_checkpoint or Path(hf_hub_download(model_repo, "model.pth", revision=revision))
    config = config_path or Path(hf_hub_download(model_repo, "config.json", revision=revision))
    labels = id2label_path or Path(hf_hub_download(model_repo, "id2label.json", revision=revision))
    return Path(adapter), Path(config), Path(labels)


def load_segmentation_model(
    *,
    base_checkpoint: Path,
    model_repo: str = SEGMENTATION_MODEL,
    revision: str | None = None,
    adapter_checkpoint: Path | None = None,
    config_path: Path | None = None,
    id2label_path: Path | None = None,
    device: str | None = None,
) -> tuple[DoRA_Sam, dict[str, Any], dict[int, str]]:
    if not base_checkpoint.is_file():
        raise FileNotFoundError(base_checkpoint)
    adapter, config_file, label_file = resolve_adapter_files(
        model_repo=model_repo, revision=revision, adapter_checkpoint=adapter_checkpoint,
        config_path=config_path, id2label_path=id2label_path,
    )
    config = json.loads(config_file.read_text(encoding="utf-8"))
    validate_adapter_config(config)
    rank = int(config["adapter"]["rank"])
    num_classes = int(config["num_classes"])
    image_size = int(config["preprocessing"]["image_size"])
    variant = str(config["adapter"]["spgen_variant"])
    sam, _ = sam_model_registry["vit_l"](
        image_size=image_size, num_classes=num_classes, checkpoint=str(base_checkpoint),
        pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1], spgen_variant=variant,
    )
    model = DoRA_Sam(sam, rank)
    model.load_dora_parameters(str(adapter))
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(target_device)
    model.eval()
    labels = {int(key): str(value) for key, value in json.loads(label_file.read_text(encoding="utf-8")).items()}
    return model, config, labels


def predict_image(
    model: DoRA_Sam,
    image: Image.Image,
    *,
    image_size: int = 1024,
    tile_threshold: int = 1024,
    batch_size: int = 6,
) -> np.ndarray:
    tiles = prepare_tiles(image.convert("RGB"), output_size=image_size, threshold=tile_threshold)
    device = next(model.parameters()).device
    predictions: list[tuple[np.ndarray, Any]] = []
    for start in range(0, len(tiles), batch_size):
        batch_tiles = tiles[start:start + batch_size]
        arrays = [np.asarray(tile, dtype=np.float32) / 255.0 for tile, _ in batch_tiles]
        tensor = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).to(device)
        with torch.inference_mode():
            outputs, _ = model(tensor, True, image_size)
            masks = outputs["masks"]
            classes = masks.softmax(dim=1).argmax(dim=1).cpu().numpy().astype(np.uint8)
        predictions.extend((prediction, info) for prediction, (_, info) in zip(classes, batch_tiles))
    return retile(predictions, image.width, image.height)
