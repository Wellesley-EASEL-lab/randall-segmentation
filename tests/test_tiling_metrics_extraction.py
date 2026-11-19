from __future__ import annotations

import json
import sys

import numpy as np
import pytest
from PIL import Image

from randall_pipeline.extraction import add_margin, component_boxes, merge_boxes
from randall_pipeline.io import class_mask, colorize, stable_id
from randall_pipeline.merge import merge_boxes_strongest_first
from randall_pipeline.metrics import confusion_matrix, segmentation_metrics
from randall_pipeline.tiling import prepare_tiles, retile
from scripts import extract_marginals


def test_odd_sized_six_tile_round_trip_geometry():
    image = Image.new("RGB", (17, 25), "white")
    tiles = prepare_tiles(image, output_size=12, threshold=1)
    predictions = [(np.full((12, 12), index + 1, dtype=np.uint8), info) for index, (_, info) in enumerate(tiles)]
    restored = retile(predictions, 17, 25)
    assert restored.shape == (25, 17)
    assert set(np.unique(restored)) == {1, 2, 3, 4, 5, 6}
    assert (restored[0, 0], restored[0, 16], restored[24, 0], restored[24, 16]) == (1, 2, 5, 6)


def test_single_tile_round_trip_geometry():
    image = Image.new("RGB", (7, 11), "white")
    tile, info = prepare_tiles(image, output_size=12, threshold=1024)[0]
    restored = retile([(np.full((12, 12), 3, dtype=np.uint8), info)], 7, 11)
    assert restored.shape == (11, 7)
    assert np.all(restored == 3)


def test_palette_and_class_id_masks_match():
    ids = np.zeros((4, 5), dtype=np.uint8)
    ids[1:3, 2:4] = 2
    assert np.array_equal(class_mask(ids, 2), class_mask(colorize(ids), 2))


def test_components_margin_and_transitive_merge():
    mask = np.zeros((10, 20), dtype=bool)
    mask[1:3, 1:4] = True
    mask[5:8, 10:14] = True
    assert component_boxes(mask, min_pixels=2) == [(1, 1, 4, 3), (10, 5, 14, 8)]
    assert add_margin((1, 1, 4, 3), 20, 10, .1) == (0, 0, 6, 5)
    assert merge_boxes([(0, 0, 2, 2), (3, 0, 5, 2), (6, 0, 8, 2)], tolerance=1) == [(0, 0, 8, 2)]


def test_metrics_and_stable_identifier():
    target = np.array([[0, 1], [1, 2]], dtype=np.uint8)
    prediction = np.array([[0, 1], [2, 2]], dtype=np.uint8)
    matrix = confusion_matrix([target], [prediction], 3)
    metrics = segmentation_metrics(matrix)
    assert metrics["overall_accuracy"] == .75
    assert metrics["per_class"]["line filler"]["recall"] == .5
    assert stable_id("a", 1) == stable_id("a", 1)
    assert stable_id("a", 1) != stable_id("a", 2)



@pytest.mark.parametrize(("limit", "value"), [
    ("--max-merged-bbox-area-fraction", "0.04"),
    ("--max-merged-bbox-dimension-fraction", "0.3"),
    ("--min-merged-target-density", "0.12"),
])
def test_cli_checks_padded_crop_and_records_settings(tmp_path, monkeypatch, limit, value):
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:25, 10:15] = 2
    mask[20:25, 28:33] = 2
    Image.fromarray(mask).save(tmp_path / "mask.png")
    Image.new("RGB", (100, 100)).save(tmp_path / "page.png")
    manifest = tmp_path / "pages.jsonl"
    manifest.write_text(json.dumps({"input_image": "page.png", "segmentation_map": "mask.png"}))
    for suffix, args, expected in [("standard", [], 1), ("guarded", [limit, value], 2)]:
        destination = tmp_path / suffix
        monkeypatch.setattr(sys, "argv", [
            "extract_marginals", "--predictions-jsonl", str(manifest),
            "--output-dir", str(destination), "--merge", "--merge-tolerance", "0.1", *args,
        ])
        extract_marginals.main()
        run = json.loads((destination / "extraction_run.json").read_text())
        assert run["regions_written"] == expected
        assert run["min_fraction"] == 0.0005
        assert run["border_class_ids"] == [5]
        assert len(list((destination / "marginals").glob("*.png"))) == expected


def test_border_bridge_wins_when_three_way_merge_is_too_wide():
    boxes = [(x, 10, x + 5, 15) for x in (5, 25, 45)]
    mask = np.zeros((100, 100), dtype=bool)
    for left, top, right, bottom in boxes:
        mask[top:bottom, left:right] = True
    border = np.zeros_like(mask)
    border[10:15, 30:45] = True
    merged = merge_boxes_strongest_first(
        [(box, box) for box in boxes], 12, mask, border, 0.50, 0.30, 0.01,
    )
    assert [tight for tight, crop in merged] == [boxes[0], (25, 10, 50, 15)]
