#!/usr/bin/env python3
"""Run deterministic K-means on exported marginal-image embeddings."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from randall_pipeline.io import read_csv, write_csv_atomic, write_json_atomic


def cluster_rows(embeddings: np.ndarray, rows: list[dict[str, str]], n_clusters: int, seed: int):
    if embeddings.ndim != 2 or len(embeddings) != len(rows):
        raise ValueError("embedding matrix and index CSV row count differ")
    if not 1 <= n_clusters <= len(rows):
        raise ValueError("n_clusters must be between 1 and the number of rows")
    labels = KMeans(n_clusters=n_clusters, n_init="auto", random_state=seed).fit_predict(embeddings)
    assignments = []
    summaries = []
    correct, labeled_total = 0, 0
    for row, cluster in zip(rows, labels):
        assignments.append({**row, "cluster_label": int(cluster)})
    for cluster in sorted(set(int(value) for value in labels)):
        members = [row for row, value in zip(rows, labels) if int(value) == cluster]
        labeled = [row for row in members if str(row.get("true_label", "")).strip() in {"0", "1"}]
        counts = Counter(int(row["true_label"]) for row in labeled)
        if counts:
            majority, majority_count = counts.most_common(1)[0]
            purity = majority_count / len(labeled)
            correct += majority_count
            labeled_total += len(labeled)
        else:
            majority, majority_count, purity = "", "", ""
        summaries.append({
            "cluster_label": cluster, "size": len(members), "labeled_size": len(labeled),
            "unlabeled_size": len(members) - len(labeled), "majority_label": majority,
            "majority_label_name": ("include" if majority == 1 else "exclude") if majority != "" else "",
            "majority_count": majority_count, "purity": purity,
        })
    return assignments, summaries, float(correct / labeled_total) if labeled_total else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--index-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-clusters", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    embeddings = np.load(args.embeddings, allow_pickle=False)
    rows = read_csv(args.index_csv)
    assignments, summaries, purity = cluster_rows(embeddings, rows, args.n_clusters, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assignment_fields = tuple(rows[0]) + ("cluster_label",) if rows else ("asset_id", "cluster_label")
    write_csv_atomic(args.output_dir / "cluster_assignments.csv", assignments, assignment_fields)
    write_csv_atomic(
        args.output_dir / "cluster_summaries.csv", summaries,
        ("cluster_label", "size", "labeled_size", "unlabeled_size", "majority_label",
         "majority_label_name", "majority_count", "purity"),
    )
    write_json_atomic(args.output_dir / "clustering_metrics.json", {
        "embeddings": str(args.embeddings), "rows": len(rows), "n_clusters": args.n_clusters,
        "seed": args.seed, "labeled_purity": purity,
    })
    print(f"Wrote {len(assignments)} assignments in {args.n_clusters} clusters")


if __name__ == "__main__":
    main()
