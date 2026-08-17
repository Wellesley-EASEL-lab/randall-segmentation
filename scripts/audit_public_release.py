#!/usr/bin/env python3
"""Fail when the release contains private state, path leakage, or unavailable Hub assets."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from huggingface_hub import HfApi

from randall_pipeline.constants import (
    OPEN_SEGMENTATIONS, PREDICTED_SEGMENTATIONS, SEGMENTATION_MASKS, SEGMENTATION_MODEL,
)

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {
    ".sqlite", ".sqlite3", ".db", ".db3", ".pth", ".pt", ".ckpt", ".bin",
    ".safetensors", ".npy", ".npz", ".parquet", ".jpg", ".jpeg", ".png",
    ".gif", ".bmp", ".webp", ".tif", ".tiff",
}
FORBIDDEN_NAMES = {"predictions.jsonl"}
PRIVATE_PATTERNS = (
    b"/home/", b"/hdd/", b"/scratch/", b"file://", b":\\Users\\",
)
PRIVATE_REPOSITORY_PATTERNS = (
    b"mapSAM-medieval", b"medieval-clustering",
)
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".html", ".css", ".js"}


def candidate_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT, check=True, capture_output=True,
    )
    return [ROOT / value.decode("utf-8") for value in result.stdout.split(b"\0") if value]


def audit_files() -> list[str]:
    errors: list[str] = []
    for path in candidate_files():
        relative = path.relative_to(ROOT)
        if path == Path(__file__).resolve():
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name in FORBIDDEN_NAMES:
            errors.append(f"forbidden release artifact: {relative}")
            continue
        lowered = path.name.lower()
        if (lowered.endswith("_reviews.csv") or lowered.endswith("_inventory.csv")) and "tests" not in relative.parts:
            errors.append(f"review/inventory data must not be released: {relative}")
        if path.suffix.lower() == ".csv" and path.is_file() and "tests" not in relative.parts:
            header = path.open(encoding="utf-8").readline().lower().split(",")
            if {"asset_id", "checked", "include", "query", "notes"}.issubset(set(header)):
                errors.append(f"review/inventory data must not be released: {relative}")
        if path.suffix.lower() in TEXT_SUFFIXES and path.is_file():
            payload = path.read_bytes()
            for pattern in (*PRIVATE_PATTERNS, *PRIVATE_REPOSITORY_PATTERNS):
                if pattern in payload:
                    errors.append(f"private reference {pattern.decode()} in {relative}")
    return errors


def audit_hub() -> list[str]:
    api = HfApi()
    errors = []
    for repo_id in (OPEN_SEGMENTATIONS, SEGMENTATION_MASKS, PREDICTED_SEGMENTATIONS):
        info = api.dataset_info(repo_id)
        if info.private:
            errors.append(f"dataset is not public: {repo_id}")
    model = api.model_info(SEGMENTATION_MODEL)
    if model.private:
        errors.append(f"model is not public: {SEGMENTATION_MODEL}")
    card_data = model.card_data
    if card_data is None:
        card = {}
    elif hasattr(card_data, "to_dict"):
        card = card_data.to_dict()
    elif isinstance(card_data, dict):
        card = card_data
    else:
        card = vars(card_data)
    license_value = str(card.get("license", "")).lower()
    license_name = str(card.get("license_name", "")).lower()
    if not license_value or "no-public-release" in license_name:
        errors.append("model card does not declare suitable public redistribution terms")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-hub", action="store_true", help="Also require every named Hub repository to be public")
    args = parser.parse_args()
    errors = audit_files()
    if args.check_hub:
        errors.extend(audit_hub())
    if errors:
        raise SystemExit("Public-release audit failed:\n- " + "\n- ".join(errors))
    print(f"Public-release audit passed for {len(candidate_files())} files")


if __name__ == "__main__":
    main()
