"""Fresh local SQLite state shared by both public review applications."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable

from .constants import REVIEW_FIELDS
from .io import normalize_review_row, parse_bool, read_csv, write_csv_atomic


def connect(path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS assets (
            asset_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            page TEXT NOT NULL DEFAULT '',
            manuscript TEXT NOT NULL DEFAULT '',
            archive TEXT NOT NULL DEFAULT '',
            location TEXT NOT NULL,
            source_path TEXT NOT NULL DEFAULT '',
            segmentation_path TEXT NOT NULL DEFAULT '',
            split TEXT NOT NULL DEFAULT '',
            checked INTEGER NOT NULL DEFAULT 0 CHECK (checked IN (0,1)),
            include INTEGER NOT NULL DEFAULT 1 CHECK (include IN (0,1)),
            query INTEGER NOT NULL DEFAULT 0 CHECK (query IN (0,1)),
            notes TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS clusters (
            cluster_label INTEGER PRIMARY KEY,
            size INTEGER,
            labeled_size INTEGER,
            unlabeled_size INTEGER,
            majority_label INTEGER,
            majority_label_name TEXT,
            majority_count INTEGER,
            purity REAL
        );
        CREATE TABLE IF NOT EXISTS cluster_assignments (
            asset_id TEXT PRIMARY KEY REFERENCES assets(asset_id) ON DELETE CASCADE,
            cluster_label INTEGER NOT NULL REFERENCES clusters(cluster_label)
        );
        CREATE INDEX IF NOT EXISTS idx_assets_pending ON assets (checked, manuscript, page);
        CREATE INDEX IF NOT EXISTS idx_cluster_assignments_label ON cluster_assignments (cluster_label, asset_id);
        """
    )
    conn.commit()


def reset(conn: sqlite3.Connection) -> None:
    conn.executescript("DROP TABLE IF EXISTS cluster_assignments; DROP TABLE IF EXISTS clusters; DROP TABLE IF EXISTS assets;")
    conn.commit()
    init_schema(conn)


def import_inventory(conn: sqlite3.Connection, inventory_csv: Path) -> int:
    rows = [normalize_review_row(row) for row in read_csv(inventory_csv)]
    if not rows:
        raise ValueError(f"no rows in {inventory_csv}")
    identifiers = [str(row["asset_id"]) for row in rows]
    if not all(identifiers) or len(identifiers) != len(set(identifiers)):
        raise ValueError("inventory asset_id values must be nonblank and unique")
    for row in rows:
        conn.execute(
            """
            INSERT INTO assets (
                asset_id, filename, page, manuscript, archive, location,
                source_path, segmentation_path, split, checked, include, query, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(asset_id) DO UPDATE SET
                filename=excluded.filename, page=excluded.page,
                manuscript=excluded.manuscript, archive=excluded.archive,
                location=excluded.location, source_path=excluded.source_path,
                segmentation_path=excluded.segmentation_path, split=excluded.split
            """,
            (
                row["asset_id"], row["filename"], row["page"], row["manuscript"],
                row["archive"], row["location"], row["source_path"],
                row["segmentation_path"], row["split"], int(row["checked"]),
                int(row["include"]), int(row["query"]), row["notes"],
            ),
        )
    conn.commit()
    return len(rows)


def import_clusters(conn: sqlite3.Connection, assignments_csv: Path, summaries_csv: Path | None = None) -> None:
    if summaries_csv:
        for row in read_csv(summaries_csv):
            label = int(row["cluster_label"])
            conn.execute(
                """
                INSERT INTO clusters VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cluster_label) DO UPDATE SET
                    size=excluded.size, labeled_size=excluded.labeled_size,
                    unlabeled_size=excluded.unlabeled_size,
                    majority_label=excluded.majority_label,
                    majority_label_name=excluded.majority_label_name,
                    majority_count=excluded.majority_count, purity=excluded.purity
                """,
                (label, _int(row.get("size")), _int(row.get("labeled_size")),
                 _int(row.get("unlabeled_size")), _int(row.get("majority_label")),
                 row.get("majority_label_name") or None, _int(row.get("majority_count")),
                 _float(row.get("purity"))),
            )
    for row in read_csv(assignments_csv):
        asset_id, label = row["asset_id"], int(row["cluster_label"])
        if conn.execute("SELECT 1 FROM assets WHERE asset_id=?", (asset_id,)).fetchone() is None:
            raise KeyError(f"cluster assignment references unknown asset_id {asset_id}")
        conn.execute("INSERT OR IGNORE INTO clusters(cluster_label) VALUES (?)", (label,))
        conn.execute(
            "INSERT INTO cluster_assignments(asset_id, cluster_label) VALUES (?, ?) "
            "ON CONFLICT(asset_id) DO UPDATE SET cluster_label=excluded.cluster_label",
            (asset_id, label),
        )
    conn.commit()


def _int(value: Any) -> int | None:
    text = str(value or "").strip()
    return int(text) if text not in {"", "NONE"} else None


def _float(value: Any) -> float | None:
    text = str(value or "").strip()
    return float(text) if text not in {"", "NONE"} else None


def update_asset(conn: sqlite3.Connection, asset_id: str, *, checked: bool, include: bool, query: bool, notes: str) -> None:
    cursor = conn.execute(
        "UPDATE assets SET checked=?, include=?, query=?, notes=? WHERE asset_id=?",
        (int(checked), int(include), int(query), str(notes), asset_id),
    )
    if cursor.rowcount != 1:
        raise KeyError(asset_id)
    conn.commit()


def export_reviews(conn: sqlite3.Connection, output_csv: Path) -> int:
    rows = [dict(row) for row in conn.execute(
        "SELECT " + ", ".join(REVIEW_FIELDS) + " FROM assets ORDER BY manuscript, page, filename, asset_id"
    )]
    for row in rows:
        for field in ("checked", "include", "query"):
            row[field] = bool(row[field])
    write_csv_atomic(output_csv, rows, REVIEW_FIELDS)
    return len(rows)


def list_assets(conn: sqlite3.Connection, *, include_checked: bool = True) -> list[sqlite3.Row]:
    where = "" if include_checked else "WHERE checked=0"
    return conn.execute(
        f"SELECT * FROM assets {where} ORDER BY manuscript, page, filename, asset_id"
    ).fetchall()


def cluster_labels(conn: sqlite3.Connection, *, include_checked: bool = False) -> list[sqlite3.Row]:
    visibility = "" if include_checked else "AND a.checked=0"
    return conn.execute(
        f"""
        SELECT ca.cluster_label, COUNT(*) AS visible_count, c.*
        FROM cluster_assignments ca
        JOIN assets a ON a.asset_id=ca.asset_id
        LEFT JOIN clusters c ON c.cluster_label=ca.cluster_label
        WHERE 1=1 {visibility}
        GROUP BY ca.cluster_label
        ORDER BY c.purity IS NULL, c.purity, ca.cluster_label
        """
    ).fetchall()


def cluster_assets(conn: sqlite3.Connection, cluster_label: int, *, include_checked: bool = False, limit: int = 50, offset: int = 0) -> list[sqlite3.Row]:
    visibility = "" if include_checked else "AND a.checked=0"
    return conn.execute(
        f"""
        SELECT a.*, ca.cluster_label, c.majority_label, c.majority_label_name, c.purity
        FROM cluster_assignments ca JOIN assets a ON a.asset_id=ca.asset_id
        LEFT JOIN clusters c ON c.cluster_label=ca.cluster_label
        WHERE ca.cluster_label=? {visibility}
        ORDER BY a.filename, a.asset_id LIMIT ? OFFSET ?
        """,
        (cluster_label, limit, offset),
    ).fetchall()
