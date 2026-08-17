#!/usr/bin/env python3
"""Extract marginal-image components and create the canonical review inventory."""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm.auto import tqdm

from randall_pipeline.constants import OPEN_SEGMENTATIONS
from randall_pipeline.datasets import iter_records, resolved_revision
from randall_pipeline.extraction import extract_record, records_from_predictions, write_inventory
from randall_pipeline.io import write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions-jsonl", type=Path)
    source.add_argument("--dataset-repo")
    parser.add_argument("--revision")
    parser.add_argument("--split", default="train")
    parser.add_argument("--image-manifest", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-id", type=int, default=2)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--min-fraction", type=float, default=0.005)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--merge-tolerance", type=float, default=0.0)
    parser.add_argument("--draw-bbox", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    revision = None
    if args.predictions_jsonl:
        records = records_from_predictions(args.predictions_jsonl, split=args.split)
        materialize = True
        source_description = str(args.predictions_jsonl)
    else:
        repo_id = args.dataset_repo or OPEN_SEGMENTATIONS
        revision = resolved_revision(repo_id, args.revision)
        records = iter_records(
            repo_id, split=args.split, revision=revision,
            image_manifest=args.image_manifest, cache_dir=args.cache_dir,
        )
        materialize = True
        source_description = repo_id
    rows = []
    for record in tqdm(records, desc="Extracting marginals", unit="page"):
        rows.extend(extract_record(
            record, args.output_dir, class_id=args.class_id, margin=args.margin,
            min_fraction=args.min_fraction, merge=args.merge,
            merge_tolerance=args.merge_tolerance, draw_bbox=args.draw_bbox,
            materialize_inputs=materialize,
        ))
    inventory = write_inventory(args.output_dir, rows)
    write_json_atomic(args.output_dir / "extraction_run.json", {
        "source": source_description,
        "resolved_revision": revision,
        "split": args.split,
        "class_id": args.class_id,
        "margin": args.margin,
        "min_fraction": args.min_fraction,
        "merge": args.merge,
        "merge_tolerance": args.merge_tolerance,
        "regions_written": len(rows),
    })
    print(f"Wrote {len(rows)} marginal crops and {inventory}")


if __name__ == "__main__":
    main()
