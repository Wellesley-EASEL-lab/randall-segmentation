#!/usr/bin/env python3
"""Prepare public Randall image/mask pairs for multiclass MapSAM training."""

from __future__ import annotations

import argparse
from pathlib import Path

from randall_pipeline.constants import OPEN_SEGMENTATIONS
from randall_pipeline.datasets import iter_records, resolved_revision
from randall_pipeline.io import write_json_atomic
from randall_pipeline.preprocessing import preprocess_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-repo", default=OPEN_SEGMENTATIONS)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--split", required=True, choices=("train", "validation", "test"))
    parser.add_argument("--image-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-size", type=int, default=1024)
    parser.add_argument("--tile-threshold", type=int, default=1024)
    parser.add_argument("--filter-background", action="store_true")
    parser.add_argument("--cache-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    revision = resolved_revision(args.dataset_repo, args.revision)
    records = iter_records(
        args.dataset_repo, split=args.split, revision=revision,
        image_manifest=args.image_manifest, cache_dir=args.cache_dir,
    )
    rows = preprocess_records(
        records, args.output_dir, output_size=args.output_size,
        tile_threshold=args.tile_threshold, filter_background=args.filter_background,
    )
    write_json_atomic(args.output_dir / f"preprocessing_run_{args.split}.json", {
        "dataset_repo": args.dataset_repo,
        "resolved_revision": revision,
        "split": args.split,
        "output_size": args.output_size,
        "tile_threshold": args.tile_threshold,
        "filter_background": args.filter_background,
        "tiles_written": len(rows),
    })
    print(f"Wrote {len(rows)} tiles to {args.output_dir} from {args.dataset_repo}@{revision}")


if __name__ == "__main__":
    main()
