from __future__ import annotations

import numpy as np
from PIL import Image

from randall_pipeline import review_db
from randall_pipeline.datasets import SegmentationRecord
from randall_pipeline.extraction import extract_record, write_inventory
from randall_pipeline.preprocessing import preprocess_records
from randall_pipeline.tiling import retile
from scripts.evaluate_segmentation import prepared_pages


def test_synthetic_preprocess_extract_review_export(tmp_path):
    image = Image.new("RGB", (17, 25), "white")
    mask = np.zeros((25, 17), dtype=np.uint8)
    mask[5:14, 2:9] = 2
    record = SegmentationRecord(
        asset_id="page-1", image=image, mask=Image.fromarray(mask, mode="L"),
        split="train", name="page", manuscript="synthetic-ms", page="1r",
    )
    prepared = tmp_path / "prepared"
    rows = preprocess_records([record], prepared, output_size=12, tile_threshold=1)
    assert len(rows) == 6
    prepared_asset_id, prepared_name, prepared_items = next(prepared_pages(prepared, "train"))
    assert (prepared_asset_id, prepared_name) == ("page-1", "page")
    restored_mask = retile(
        [(tile_mask, tile_info) for _, tile_mask, tile_info in prepared_items], 17, 25
    )
    np.testing.assert_array_equal(restored_mask, mask)
    extraction = tmp_path / "extraction"
    crops = extract_record(record, extraction, min_fraction=0, materialize_inputs=True)
    assert len(crops) == 1
    inventory = write_inventory(extraction, crops)
    database = tmp_path / "reviews.sqlite3"
    with review_db.connect(database) as conn:
        review_db.init_schema(conn)
        assert review_db.import_inventory(conn, inventory) == 1
        review_db.update_asset(conn, crops[0]["asset_id"], checked=True, include=True, query=False, notes="synthetic")
        export = tmp_path / "reviews.csv"
        assert review_db.export_reviews(conn, export) == 1
    assert export.is_file()
