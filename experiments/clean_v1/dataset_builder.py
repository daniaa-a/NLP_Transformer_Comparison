"""Build immutable, content-deduplicated clean_v1 manifests from aclImdb."""

from __future__ import annotations

import csv
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "data" / "raw" / "aclImdb"
EXPERIMENT_ROOT = Path(__file__).resolve().parent
MANIFEST_DIR = EXPERIMENT_ROOT / "manifests"
SEED = 42
VALIDATION_FRACTION = 0.20
LABELS = {"neg": 0, "pos": 1}
MANIFEST_COLUMNS = ["source_path", "source_filename", "label", "normalized_text_hash"]


def normalize_review(text: str) -> str:
    """Normalize identity only; punctuation and words are retained."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def text_hash(text: str) -> str:
    normalized = normalize_review(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _read_source_file(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Cannot read UTF-8 review {path}: {exc}") from exc
    if not text.strip():
        raise ValueError(f"Empty review text: {path}")
    return text


def collect_source(split: str) -> list[dict]:
    """Read labeled files in the shared loader's stable neg-then-pos order."""
    records = []
    for sentiment, label in LABELS.items():
        folder = SOURCE_ROOT / split / sentiment
        if not folder.is_dir():
            raise FileNotFoundError(f"Required labeled directory is missing: {folder}")
        files = sorted(folder.glob("*.txt"))
        if not files:
            raise ValueError(f"No labeled .txt reviews found in {folder}")
        for path in files:
            text = _read_source_file(path)
            records.append({
                "source_path": path.relative_to(PROJECT_ROOT).as_posix(),
                "source_filename": path.name,
                "label": label,
                "text": text,
                "normalized_text_hash": text_hash(text),
                "source_split": split,
                "sentiment_dir": sentiment,
            })
    return records


def _group_records(records: list[dict]) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for record in records:
        groups[record["normalized_text_hash"]].append(record)
    return dict(groups)


def _assert_no_label_conflicts(groups: dict[str, list[dict]]) -> None:
    conflicts = []
    for digest, records in groups.items():
        labels = {record["label"] for record in records}
        if len(labels) > 1:
            conflicts.append((digest, records))
    if conflicts:
        preview = "\n".join(
            f"{digest}: " + ", ".join(
                f"{record['source_path']} (label={record['label']})" for record in records
            )
            for digest, records in conflicts[:10]
        )
        raise ValueError(
            f"Found {len(conflicts)} normalized review groups with conflicting labels; "
            f"no manifests were written. First groups:\n{preview}"
        )


def _representatives(groups: dict[str, list[dict]]) -> list[dict]:
    """Select the first stable source path for each same-label content group."""
    selected = []
    for digest, records in groups.items():
        representative = dict(records[0])
        representative["normalized_text_hash"] = digest
        selected.append(representative)
    return selected


def _write_manifest(name: str, records: list[dict]) -> str:
    path = MANIFEST_DIR / f"{name}.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({column: record[column] for column in MANIFEST_COLUMNS})
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifests() -> dict:
    if not SOURCE_ROOT.is_dir():
        raise FileNotFoundError(f"IMDb source dataset is missing: {SOURCE_ROOT}")
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    original_train = collect_source("train")
    original_test = collect_source("test")
    train_groups = _group_records(original_train)
    test_groups = _group_records(original_test)
    global_groups = _group_records(original_train + original_test)
    _assert_no_label_conflicts(global_groups)

    # Reserve all unique official-test hashes first; no matching original-train
    # text can enter train or validation.
    test_records = _representatives(test_groups)
    test_hashes = set(test_groups)
    train_unique = _representatives(train_groups)
    eligible_train = [record for record in train_unique if record["normalized_text_hash"] not in test_hashes]
    if not eligible_train:
        raise ValueError("No eligible training records remain after official-test isolation.")

    train_records, validation_records = train_test_split(
        eligible_train,
        test_size=VALIDATION_FRACTION,
        random_state=SEED,
        stratify=[record["label"] for record in eligible_train],
        shuffle=True,
    )
    train_records = list(train_records)
    validation_records = list(validation_records)
    split_records = {
        "train": train_records,
        "validation": validation_records,
        "test": test_records,
    }

    manifest_hashes = {
        split: _write_manifest(split, records)
        for split, records in split_records.items()
    }

    counts = {}
    for split, records in split_records.items():
        label_counts = Counter(record["label"] for record in records)
        counts[split] = {
            "samples": len(records),
            "negative": label_counts[0],
            "positive": label_counts[1],
            "ratio_of_train_eligible_or_official_test": (
                len(records) / len(eligible_train) if split != "test" else len(records) / len(test_records)
            ),
        }

    duplicate_groups = []
    for digest, records in sorted(global_groups.items()):
        if len(records) < 2:
            continue
        duplicate_groups.append({
            "normalized_text_hash": digest,
            "label": records[0]["label"],
            "conflicting_labels": False,
            "group_file_count": len(records),
            "source_memberships": json.dumps(sorted({record["source_split"] for record in records})),
            "source_files": json.dumps([record["source_path"] for record in records]),
        })
    duplicate_path = MANIFEST_DIR / "duplicate_groups.csv"
    with duplicate_path.open("w", encoding="utf-8", newline="") as stream:
        columns = ["normalized_text_hash", "label", "conflicting_labels", "group_file_count", "source_memberships", "source_files"]
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(duplicate_groups)

    overlaps = {
        "train_validation": len({r["normalized_text_hash"] for r in train_records} & {r["normalized_text_hash"] for r in validation_records}),
        "train_test": len({r["normalized_text_hash"] for r in train_records} & {r["normalized_text_hash"] for r in test_records}),
        "validation_test": len({r["normalized_text_hash"] for r in validation_records} & {r["normalized_text_hash"] for r in test_records}),
    }
    summary = {
        "dataset_version": "clean_v1",
        "normalization": "Unicode NFKC, casefold, collapse all whitespace; original text is unchanged in source files.",
        "hash_algorithm": "SHA-256 of UTF-8 normalized text",
        "seed": SEED,
        "validation_fraction": VALIDATION_FRACTION,
        "split_strategy": "unique original-train groups stratified 80/20; unique official-test groups kept isolated",
        "original_labeled_file_counts": {"train": len(original_train), "official_test": len(original_test)},
        "unique_source_counts": {
            "original_train": len(train_groups),
            "official_test": len(test_groups),
            "shared_train_test_texts": len(set(train_groups) & set(test_groups)),
            "global_unique_texts": len(global_groups),
            "train_eligible_after_test_isolation": len(eligible_train),
        },
        "global_unique_class_counts": {
            "negative": sum(1 for records in global_groups.values() if records[0]["label"] == 0),
            "positive": sum(1 for records in global_groups.values() if records[0]["label"] == 1),
        },
        "splits": counts,
        "overlap_counts": overlaps,
        "conflicting_label_groups": 0,
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_file_instances_in_duplicate_groups": sum(group["group_file_count"] for group in duplicate_groups),
        "manifest_sha256": manifest_hashes,
    }
    (MANIFEST_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(build_manifests(), indent=2))
