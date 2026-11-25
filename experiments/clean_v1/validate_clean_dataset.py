"""Validate clean_v1 paths, labels, group isolation, ordering, and reproducibility."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from clean_data_loader import load_clean_split
from dataset_builder import (
    MANIFEST_COLUMNS, MANIFEST_DIR, SEED, SOURCE_ROOT, VALIDATION_FRACTION,
    _assert_no_label_conflicts, _group_records, _representatives, collect_source,
)
from sklearn.model_selection import train_test_split

REPORT_PATH = EXPERIMENT_ROOT / "dataset_validation_report.md"
EXPECTED_SIZES = {"train": 19_824, "validation": 4_957, "test": 24_801}
EXPECTED_LABELS = {
    "train": {0: 9_869, 1: 9_955},
    "validation": {0: 2_468, 1: 2_489},
    "test": {0: 12_361, 1: 12_440},
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_manifest_rows():
    """Recreate the deterministic manifest records in memory without writing files."""
    original_train = collect_source("train")
    original_test = collect_source("test")
    train_groups = _group_records(original_train)
    test_groups = _group_records(original_test)
    global_groups = _group_records(original_train + original_test)
    _assert_no_label_conflicts(global_groups)
    test_records = _representatives(test_groups)
    test_hashes = set(test_groups)
    eligible = [r for r in _representatives(train_groups) if r["normalized_text_hash"] not in test_hashes]
    train_records, validation_records = train_test_split(
        eligible,
        test_size=VALIDATION_FRACTION,
        random_state=SEED,
        stratify=[r["label"] for r in eligible],
        shuffle=True,
    )
    return {
        "train": list(train_records),
        "validation": list(validation_records),
        "test": test_records,
    }, global_groups, len(train_groups), len(test_groups), len(set(train_groups) & set(test_groups))


def read_manifest_metadata(split):
    path = MANIFEST_DIR / f"{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {split} manifest: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != MANIFEST_COLUMNS:
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames}")
        rows = list(reader)
    return rows, path


def validate():
    checks = []
    details = {}
    summary_path = MANIFEST_DIR / "dataset_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing dataset summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("dataset_version") != "clean_v1":
        raise ValueError("dataset_summary.json has the wrong dataset version.")

    expected, global_groups, original_train_unique, original_test_unique, source_overlap = expected_manifest_rows()
    _assert_no_label_conflicts(global_groups)
    split_data = {}
    manifest_hashes = {}
    for split in ("train", "validation", "test"):
        metadata, manifest_path = read_manifest_metadata(split)
        # The clean loader resolves every source path, reads UTF-8 text and verifies its text hash.
        data = load_clean_split(split)
        split_data[split] = data
        manifest_hashes[split] = sha256_file(manifest_path)
        if len(metadata) != len(data):
            raise ValueError(f"Row count changed while loading {split} manifest.")
        if len(data) != EXPECTED_SIZES[split]:
            raise ValueError(f"Expected {EXPECTED_SIZES[split]} {split} rows, found {len(data)}.")
        label_counts = dict(Counter(row["label"] for row in data))
        if label_counts != EXPECTED_LABELS[split]:
            raise ValueError(f"Unexpected {split} label counts: {label_counts}.")
        hashes = [row["normalized_text_hash"] for row in data]
        if len(hashes) != len(set(hashes)):
            raise ValueError(f"Duplicate normalized hashes within {split}.")
        for row in data:
            source_path = Path(row["source_path"]).as_posix()
            if split == "test":
                valid_source = source_path.startswith("data/raw/aclImdb/test/")
            else:
                valid_source = (
                    source_path.startswith("data/raw/aclImdb/train/neg/")
                    or source_path.startswith("data/raw/aclImdb/train/pos/")
                )
            if not valid_source or "/unsup/" in source_path:
                raise ValueError(f"Invalid source membership for {split}: {source_path}")
            expected_label = 0 if "/neg/" in source_path else 1
            if row["label"] != expected_label:
                raise ValueError(f"Label/source mismatch for {source_path}: label={row['label']}")

        # Verify each manifest has the deterministic expected identities and ordering.
        expected_meta = [
            (r["source_path"], r["source_filename"], str(r["label"]), r["normalized_text_hash"])
            for r in expected[split]
        ]
        actual_meta = [
            (r["source_path"], r["source_filename"], r["label"], r["normalized_text_hash"])
            for r in metadata
        ]
        if actual_meta != expected_meta:
            raise ValueError(f"{split} manifest does not reproduce from source, seed, and algorithm in identical order.")
        details[split] = {
            "samples": len(data),
            "negative": label_counts[0],
            "positive": label_counts[1],
            "negative_fraction": label_counts[0] / len(data),
            "manifest_sha256": manifest_hashes[split],
        }
        checks.append(f"{split} manifest paths resolve, texts are readable/non-empty, hashes and labels match")
        checks.append(f"{split} count/class distribution verified: {len(data)} / {label_counts}")

    train_hashes = {row["normalized_text_hash"] for row in split_data["train"]}
    val_hashes = {row["normalized_text_hash"] for row in split_data["validation"]}
    test_hashes = {row["normalized_text_hash"] for row in split_data["test"]}
    overlaps = {
        "train_validation": len(train_hashes & val_hashes),
        "train_test": len(train_hashes & test_hashes),
        "validation_test": len(val_hashes & test_hashes),
    }
    if any(overlaps.values()):
        raise ValueError(f"Clean manifest leakage detected: {overlaps}")
    checks.append("all pairwise normalized-hash overlaps are zero")
    if len(test_hashes) != original_test_unique:
        raise ValueError("Clean test does not retain every unique official-test review.")
    if len(train_hashes | val_hashes) != original_train_unique - source_overlap:
        raise ValueError("Clean train/validation do not match the test-isolated original-train unique pool.")
    checks.append("test is sourced only from official test; shared train/test hashes are excluded from train/validation")
    checks.append("no train/unsup source path is present")

    if summary.get("manifest_sha256") != manifest_hashes:
        raise ValueError("Manifest file hashes do not match dataset_summary.json.")
    if summary.get("overlap_counts") != overlaps:
        raise ValueError("Summary overlap counts do not match validated manifests.")

    # A second manifest load confirms stable row ordering independent of caller state.
    for split in ("train", "validation", "test"):
        first = [(r["source_path"], r["label"], r["normalized_text_hash"]) for r in split_data[split]]
        second = load_clean_split(split)
        again = [(r["source_path"], r["label"], r["normalized_text_hash"]) for r in second]
        if first != again:
            raise ValueError(f"Repeated {split} manifest loads changed row order or contents.")
    checks.append("repeated manifest loads produce identical ordering and identities")
    checks.append("reconstructed manifests match the saved seed and deterministic stratified split")

    details["global"] = {
        "total_unique_texts": len(global_groups),
        "negative": summary["global_unique_class_counts"]["negative"],
        "positive": summary["global_unique_class_counts"]["positive"],
        "source_shared_texts": source_overlap,
        "overlap_counts": overlaps,
        "ratio_train_to_validation": len(split_data["train"]) / len(split_data["validation"]),
    }
    return {"checks": checks, "details": details, "manifest_hashes": manifest_hashes}


def main():
    try:
        result = validate()
    except Exception as exc:
        REPORT_PATH.write_text(
            "# clean_v1 Dataset Validation\n\n**Status: FAILED**\n\n"
            f"Validation stopped with: `{type(exc).__name__}: {exc}`\n",
            encoding="utf-8",
        )
        print(f"DATASET VALIDATION FAILED: {exc}")
        raise

    details = result["details"]
    lines = [
        "# clean_v1 Dataset Validation",
        "",
        "**Status: PASSED**",
        "",
        "All checks below were run against the manifests and source review files.",
        "",
        "| Split | Samples | Negative | Positive | Negative fraction | Manifest SHA-256 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for split in ("train", "validation", "test"):
        d = details[split]
        lines.append(
            f"| {split} | {d['samples']:,} | {d['negative']:,} | {d['positive']:,} | "
            f"{d['negative_fraction']:.4%} | `{d['manifest_sha256']}` |"
        )
    global_data = details["global"]
    lines.extend([
        "",
        f"Unique content groups: {global_data['total_unique_texts']:,}; "
        f"negative={global_data['negative']:,}; positive={global_data['positive']:,}.",
        f"Source overlap removed before splitting: {global_data['source_shared_texts']:,} hashes.",
        f"Pairwise manifest overlap counts: `{json.dumps(global_data['overlap_counts'], sort_keys=True)}`.",
        f"Train/validation size ratio: {global_data['ratio_train_to_validation']:.6f}:1.",
        "",
        "## Checks",
        "",
    ])
    lines.extend(f"- PASS: {check}." for check in result["checks"])
    lines.extend([
        "",
        "Normalization: Unicode NFKC, casefold, and collapse whitespace. Original review text is read unchanged for model input.",
        "The official test manifest contains only unique source-test paths; all hashes shared with original train were excluded from train and validation.",
    ])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result["details"], indent=2))
    print(f"Validation passed; report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
