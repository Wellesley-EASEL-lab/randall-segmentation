from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from randall_pipeline.constants import OPEN_SEGMENTATIONS, PREDICTED_SEGMENTATIONS, SEGMENTATION_MASKS
from randall_pipeline.datasets import iter_records


class FakeDataset:
    def __init__(self, rows):
        self.rows = rows
        self.column_names = list(rows[0])

    def __iter__(self):
        return iter(self.rows)


def image_and_mask():
    image = Image.new("RGB", (8, 9), "white")
    mask = Image.fromarray(np.zeros((9, 8), dtype=np.uint8), mode="L")
    return image, mask


def write_manifest(path: Path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_open_adapter_needs_no_manifest():
    image, mask = image_and_mask()
    dataset = FakeDataset([{"name": "page", "uuid": "u1", "pixel_values": image, "labels": mask}])
    rows = list(iter_records(OPEN_SEGMENTATIONS, split="test", dataset=dataset))
    assert rows[0].uuid == "u1"
    assert rows[0].split == "test"


def test_masks_adapter_joins_uuid_manifest(tmp_path):
    image, mask = image_and_mask()
    image_path = tmp_path / "page.png"
    image.save(image_path)
    manifest = tmp_path / "images.csv"
    write_manifest(manifest, ["uuid", "image_path"], [{"uuid": "u1", "image_path": "page.png"}])
    dataset = FakeDataset([{"name": "page", "uuid": "u1", "labels": mask}])
    rows = list(iter_records(SEGMENTATION_MASKS, image_manifest=manifest, dataset=dataset))
    assert rows[0].image.size == mask.size


def test_predicted_adapter_joins_composite_manifest(tmp_path):
    image, mask = image_and_mask()
    image_path = tmp_path / "page.png"
    image.save(image_path)
    manifest = tmp_path / "images.csv"
    fields = ["manuscript", "archive", "page", "image_path"]
    write_manifest(manifest, fields, [{"manuscript": "M", "archive": "A", "page": "1r", "image_path": "page.png"}])
    dataset = FakeDataset([{"manuscript": "M", "archive": "A", "page": "1r", "segmentation_map": mask}])
    rows = list(iter_records(PREDICTED_SEGMENTATIONS, image_manifest=manifest, dataset=dataset))
    assert (rows[0].manuscript, rows[0].archive, rows[0].page) == ("M", "A", "1r")


def test_manifest_duplicate_and_dimension_mismatch_fail(tmp_path):
    image, mask = image_and_mask()
    image.resize((3, 3)).save(tmp_path / "page.png")
    manifest = tmp_path / "images.csv"
    write_manifest(manifest, ["uuid", "image_path"], [
        {"uuid": "u1", "image_path": "page.png"},
        {"uuid": "u1", "image_path": "page.png"},
    ])
    dataset = FakeDataset([{"name": "page", "uuid": "u1", "labels": mask}])
    try:
        list(iter_records(SEGMENTATION_MASKS, image_manifest=manifest, dataset=dataset))
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate manifest should fail")

