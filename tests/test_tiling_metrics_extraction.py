from __future__ import annotations

import numpy as np
from PIL import Image

from randall_pipeline.extraction import add_margin, component_boxes, merge_boxes
from randall_pipeline.io import class_mask, colorize, stable_id
from randall_pipeline.metrics import confusion_matrix, segmentation_metrics
from randall_pipeline.tiling import prepare_tiles, retile


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

