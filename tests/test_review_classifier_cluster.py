from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from randall_pipeline import review_db
from randall_pipeline.classifier_build import validate_and_partition
from randall_pipeline.classifier_model import DINOv3ForIncludeClassification
from randall_pipeline.constants import REVIEW_FIELDS
from randall_pipeline.io import read_csv, write_csv_atomic
from scripts.cluster_embeddings import cluster_rows


def review_row(asset_id: str, location: str, manuscript: str, split: str, include: bool, *, checked=True, query=False, notes=""):
    return {
        "asset_id": asset_id, "filename": Path(location).name, "page": asset_id,
        "manuscript": manuscript, "archive": "", "location": location,
        "source_path": "", "segmentation_path": "", "split": split,
        "checked": checked, "include": include, "query": query, "notes": notes,
    }


def create_review_fixture(tmp_path: Path):
    rows = []
    for split_index, split in enumerate(("train", "validation", "test")):
        for label_value in (False, True):
            asset_id = f"{split}-{int(label_value)}"
            location = f"{asset_id}.png"
            Image.new("RGB", (8, 8), "white").save(tmp_path / location)
            rows.append(review_row(asset_id, location, f"ms-{split_index}", split, label_value))
    Image.new("RGB", (8, 8), "white").save(tmp_path / "query.png")
    rows.append(review_row("query", "query.png", "ms-q", "", True, query=True, notes="adjudicate"))
    csv_path = tmp_path / "reviews.csv"
    write_csv_atomic(csv_path, rows, REVIEW_FIELDS)
    return csv_path, rows


def test_review_database_round_trip_and_cluster_suggestion_is_advisory(tmp_path):
    csv_path, rows = create_review_fixture(tmp_path)
    db_path = tmp_path / "reviews.sqlite3"
    assignment_path = tmp_path / "assignments.csv"
    summary_path = tmp_path / "summaries.csv"
    write_csv_atomic(assignment_path, [{"asset_id": rows[0]["asset_id"], "cluster_label": 7}], ("asset_id", "cluster_label"))
    write_csv_atomic(summary_path, [{
        "cluster_label": 7, "size": 1, "labeled_size": 1, "unlabeled_size": 0,
        "majority_label": 1, "majority_label_name": "include", "majority_count": 1, "purity": 1.0,
    }], ("cluster_label", "size", "labeled_size", "unlabeled_size", "majority_label", "majority_label_name", "majority_count", "purity"))
    with review_db.connect(db_path) as conn:
        review_db.init_schema(conn)
        review_db.import_inventory(conn, csv_path)
        review_db.import_clusters(conn, assignment_path, summary_path)
        before = conn.execute("SELECT include FROM assets WHERE asset_id=?", (rows[0]["asset_id"],)).fetchone()[0]
        assert before == int(rows[0]["include"])
        review_db.update_asset(conn, rows[0]["asset_id"], checked=True, include=False, query=True, notes="<safe>")
        output = tmp_path / "export.csv"
        review_db.export_reviews(conn, output)
    exported = {row["asset_id"]: row for row in read_csv(output)}
    assert exported[rows[0]["asset_id"]]["notes"] == "<safe>"
    assert exported[rows[0]["asset_id"]]["include"] == "False"


def test_classifier_partition_requires_explicit_splits_and_makes_queries_unlabeled(tmp_path):
    csv_path, _ = create_review_fixture(tmp_path)
    labeled, unlabeled = validate_and_partition(csv_path, tmp_path)
    assert {key: len(value) for key, value in labeled.items()} == {"train": 2, "validation": 2, "test": 2}
    assert [row["asset_id"] for row in unlabeled] == ["query"]


def test_classifier_rejects_manuscript_overlap(tmp_path):
    csv_path, rows = create_review_fixture(tmp_path)
    rows[2]["manuscript"] = rows[0]["manuscript"]
    write_csv_atomic(csv_path, rows, REVIEW_FIELDS)
    try:
        validate_and_partition(csv_path, tmp_path)
    except ValueError as error:
        assert "cross" in str(error)
    else:
        raise AssertionError("manuscript leakage should fail")


class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"num_register_tokens": 0})()


def test_configured_pooling_controls_exported_embedding_dimension():
    model = DINOv3ForIncludeClassification.__new__(DINOv3ForIncludeClassification)
    nn.Module.__init__(model)
    model.backbone = DummyBackbone()
    hidden = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    model.pooling = "cls"
    assert model.pool_hidden_states(hidden).shape == (2, 3)
    model.pooling = "cls-patch-mean"
    assert model.pool_hidden_states(hidden).shape == (2, 6)


def test_deterministic_clustering_and_majority_suggestions():
    embeddings = np.array([[0, 0], [0, .1], [10, 10], [10, 10.1]], dtype=float)
    rows = [
        {"asset_id": "a", "true_label": "0"},
        {"asset_id": "b", "true_label": ""},
        {"asset_id": "c", "true_label": "1"},
        {"asset_id": "d", "true_label": "1"},
    ]
    first = cluster_rows(embeddings, rows, 2, 1234)
    second = cluster_rows(embeddings, rows, 2, 1234)
    assert first == second
    assert {row["majority_label_name"] for row in first[1]} == {"exclude", "include"}

