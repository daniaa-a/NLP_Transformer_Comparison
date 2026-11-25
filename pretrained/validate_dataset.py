"""Read-only readiness checks for the local IMDb corpus and shared split."""

from collections import Counter
from hashlib import sha256
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loader import DATA_DIR, load_imdb_data, load_reviews

EXPECTED_COUNTS = {
    ("train", "neg"): 12_500,
    ("train", "pos"): 12_500,
    ("test", "neg"): 12_500,
    ("test", "pos"): 12_500,
}


def digest(text):
    return sha256(text.encode("utf-8")).hexdigest()


def validate_directories_and_files(data_dir):
    required = [
        data_dir / "train" / "neg", data_dir / "train" / "pos",
        data_dir / "train" / "unsup", data_dir / "test" / "neg",
        data_dir / "test" / "pos",
    ]
    missing = [path for path in required if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Missing required IMDb directories: " + ", ".join(map(str, missing)))

    labeled_texts = {}
    paths_by_hash = {}
    for (split, sentiment), expected in EXPECTED_COUNTS.items():
        folder = data_dir / split / sentiment
        files = sorted(folder.glob("*.txt"))
        if len(files) != expected:
            raise ValueError(f"{folder}: expected {expected:,} .txt reviews, found {len(files):,}.")
        texts = []
        for file_path in files:
            try:
                text = file_path.read_text(encoding="utf-8", errors="strict")
            except (OSError, UnicodeError) as exc:
                raise ValueError(f"Unreadable review file {file_path}: {exc}") from exc
            if not text.strip():
                raise ValueError(f"Empty review file: {file_path}")
            texts.append(text)
            paths_by_hash.setdefault(digest(text), []).append(file_path)
        labeled_texts[(split, sentiment)] = texts
        print(f"{split}/{sentiment}: {len(files):,} readable reviews; label={0 if sentiment == 'neg' else 1}")
    unsup_count = sum(1 for path in (data_dir / "train" / "unsup").glob("*.txt") if path.is_file())
    print(f"train/unsup: {unsup_count:,} unlabeled review files (not included in supervised splits)")
    return labeled_texts, paths_by_hash


def validate_split(labeled_texts, paths_by_hash):
    train_df, val_df, test_df = load_imdb_data()
    expected = [(train_df, 20_000, "train"), (val_df, 5_000, "validation"), (test_df, 25_000, "test")]
    for frame, size, name in expected:
        if len(frame) != size:
            raise ValueError(f"Shared {name} split expected {size:,} reviews, found {len(frame):,}.")
        distribution = Counter(int(label) for label in frame["label"])
        expected_distribution = {0: size // 2, 1: size // 2}
        if dict(distribution) != expected_distribution:
            raise ValueError(f"Shared {name} label distribution is {dict(distribution)}, expected {expected_distribution}.")
        if frame["text"].isna().any():
            raise ValueError(f"Shared {name} split contains missing review text.")
        print(f"Shared {name}: {len(frame):,} reviews; negative={distribution[0]:,}, positive={distribution[1]:,}")

    train_hashes = {digest(text) for text in train_df["text"]}
    val_hashes = {digest(text) for text in val_df["text"]}
    test_hashes = {digest(text) for text in test_df["text"]}
    overlaps = {
        "train_validation": train_hashes & val_hashes,
        "train_test": train_hashes & test_hashes,
        "validation_test": val_hashes & test_hashes,
    }
    print("Exact review-text overlaps across splits:")
    for pair, hashes in overlaps.items():
        print(f"  {pair}: {len(hashes):,} unique reviews")

    train_source_hashes = {digest(text) for split in ("train",) for sentiment in ("neg", "pos")
                           for text in labeled_texts[(split, sentiment)]}
    test_source_hashes = {digest(text) for sentiment in ("neg", "pos")
                          for text in labeled_texts[("test", sentiment)]}
    if not train_hashes <= train_source_hashes or not val_hashes <= train_source_hashes:
        raise ValueError("Train or validation contains text outside the labeled original training reviews.")
    if test_hashes != test_source_hashes:
        raise ValueError("Shared test rows do not match the labeled official test source reviews.")
    if any(overlaps.values()):
        sample_pair = next(name for name, hashes in overlaps.items() if hashes)
        sample_hash = next(iter(overlaps[sample_pair]))
        examples = paths_by_hash.get(sample_hash, [])
        print("Example duplicate source files:")
        for path in examples:
            print(f"  {path}")
        raise ValueError("Exact duplicate review text crosses split boundaries; leakage check failed.")
    print("Leakage check passed: split texts are disjoint and match their labeled source splits.")
    print("Unsupervised exclusion confirmed from data_loader.py: load_reviews reads only pos/neg folders.")


def main():
    data_dir = (PROJECT_ROOT / DATA_DIR).resolve() if not DATA_DIR.is_absolute() else DATA_DIR
    print(f"IMDb data directory: {data_dir}")
    labeled_texts, paths_by_hash = validate_directories_and_files(data_dir)
    validate_split(labeled_texts, paths_by_hash)
    print("Dataset readiness checks passed.")


if __name__ == "__main__":
    main()
