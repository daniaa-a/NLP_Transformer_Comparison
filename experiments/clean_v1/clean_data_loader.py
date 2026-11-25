"""Shared ordered manifest loader for both clean_v1 model pipelines."""

from __future__ import annotations

import csv
from pathlib import Path

from dataset_builder import EXPERIMENT_ROOT, MANIFEST_DIR, PROJECT_ROOT, text_hash

EXPECTED_COLUMNS = ["source_path", "source_filename", "label", "normalized_text_hash"]
SPLITS = {"train", "validation", "test"}


def load_clean_split(split: str):
    """Load one manifest, preserving row order and original review text."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {sorted(SPLITS)}; received {split!r}")
    manifest = MANIFEST_DIR / f"{split}.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Clean {split} manifest is missing: {manifest}")

    rows = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(f"Unexpected columns in {manifest}: {reader.fieldnames}")
        for line_number, item in enumerate(reader, start=2):
            try:
                label = int(item["label"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid label at {manifest}:{line_number}") from exc
            if label not in (0, 1):
                raise ValueError(f"Label must be 0 or 1 at {manifest}:{line_number}; got {label}")
            source = (PROJECT_ROOT / item["source_path"]).resolve()
            if not source.is_relative_to(PROJECT_ROOT):
                raise ValueError(f"Manifest path escapes the project directory: {item['source_path']}")
            if not source.is_file():
                raise FileNotFoundError(f"Review source from {manifest}:{line_number} is missing: {source}")
            try:
                text = source.read_text(encoding="utf-8", errors="strict")
            except (OSError, UnicodeError) as exc:
                raise ValueError(f"Cannot read review source {source}: {exc}") from exc
            if not text.strip():
                raise ValueError(f"Empty review text at {source}")
            computed_hash = text_hash(text)
            if computed_hash != item["normalized_text_hash"]:
                raise ValueError(f"Review text hash mismatch at {manifest}:{line_number}")
            if Path(item["source_path"]).name != item["source_filename"]:
                raise ValueError(f"source_filename mismatch at {manifest}:{line_number}")
            rows.append({
                "text": text,
                "label": label,
                "normalized_text_hash": computed_hash,
                "source_path": item["source_path"],
                "source_filename": item["source_filename"],
            })
    if not rows:
        raise ValueError(f"Clean {split} manifest is empty: {manifest}")
    return rows


def load_clean_train_validation():
    """Load only training and validation; keeps the held-out test unaccessed."""
    return load_clean_split("train"), load_clean_split("validation")


def load_clean_test():
    """Load the official-test-derived manifest for final evaluation only."""
    return load_clean_split("test")


def load_dataset_summary() -> dict:
    import json

    path = MANIFEST_DIR / "dataset_summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Clean dataset summary is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
