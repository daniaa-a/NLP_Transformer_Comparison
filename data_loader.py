
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


DATA_DIR = Path("data/raw/aclImdb")
RANDOM_STATE = 42
VALIDATION_SIZE = 0.20


def load_reviews(split):
    """
    Load positive and negative IMDb reviews.

    Labels:
        0 = Negative
        1 = Positive
    """

    reviews = []

    split_dir = DATA_DIR / split

    for sentiment, label in [("neg", 0), ("pos", 1)]:

        folder = split_dir / sentiment

        if not folder.exists():
            raise FileNotFoundError(
                f"Dataset folder not found: {folder}"
            )

        for file_path in sorted(folder.glob("*.txt")):

            text = file_path.read_text(
                encoding="utf-8",
                errors="replace"
            )

            reviews.append({
                "text": text,
                "label": label
            })

    return pd.DataFrame(reviews)


def load_imdb_data():
    """
    Load IMDb dataset and create fixed train,
    validation, and test sets.
    """

    # Load original training data
    full_train = load_reviews("train")

    # Load official test data
    test_df = load_reviews("test")

    # Split training data into train and validation
    train_df, val_df = train_test_split(
        full_train,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=full_train["label"],
        shuffle=True
    )

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df


if __name__ == "__main__":

    train_df, val_df, test_df = load_imdb_data()

    print("\n========== IMDb Dataset ==========")

    for name, df in [
        ("Training", train_df),
        ("Validation", val_df),
        ("Testing", test_df)
    ]:

        print(f"\n{name} samples: {len(df)}")

        print("Negative:", (df["label"] == 0).sum())
        print("Positive:", (df["label"] == 1).sum())

    print("\nSample review:")
    print(train_df.iloc[0]["text"][:300])

    print("\nDataset loaded successfully!")