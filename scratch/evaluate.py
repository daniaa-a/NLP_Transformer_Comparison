
import sys
import json
import time

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_loader import load_imdb_data
from scratch.model import ScratchTransformer
from scratch.tokenizer import ScratchTokenizer


DEVICE = torch.device("cuda")

CHECKPOINT_DIR = PROJECT_ROOT / "scratch" / "checkpoints"
BATCH_SIZE = 16


def main():

    print("\nLoading best checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_DIR / "best_model.pth",
        map_location=DEVICE
    )

    config = checkpoint["config"]

    # Load saved vocabulary
    with open(
        CHECKPOINT_DIR / "tokenizer.json",
        "r",
        encoding="utf-8"
    ) as file:
        word_to_id = json.load(file)

    tokenizer = ScratchTokenizer(
        vocab_size=len(word_to_id),
        max_length=config["max_length"]
    )

    tokenizer.word_to_id = {
        word: int(index)
        for word, index in word_to_id.items()
    }

    tokenizer.id_to_word = {
        index: word
        for word, index in tokenizer.word_to_id.items()
    }

    print("Vocabulary loaded:", len(word_to_id))

    # Load official test set
    print("\nLoading official test data...")

    _, _, test_df = load_imdb_data()

    X_test = tokenizer.encode_batch(
        test_df["text"].tolist()
    )

    y_test = torch.tensor(
        test_df["label"].values,
        dtype=torch.long
    )

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Rebuild model
    model = ScratchTransformer(
        vocab_size=checkpoint["vocab_size"],
        embedding_dim=config["embedding_dim"],
        num_heads=config["num_heads"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        max_length=config["max_length"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
        padding_idx=0
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("\nEvaluating on GPU...")

    all_predictions = []
    all_labels = []

    total_inference_time = 0.0

    with torch.inference_mode():

        for inputs, labels in test_loader:

            inputs = inputs.to(
                DEVICE,
                non_blocking=True
            )

            torch.cuda.synchronize()
            start = time.perf_counter()

            outputs = model(inputs)

            torch.cuda.synchronize()
            total_inference_time += time.perf_counter() - start

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.tolist()
            )

    # Calculate metrics
    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        average="macro"
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="macro"
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro"
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions
    )

    avg_latency_ms = (
        total_inference_time / len(all_labels)
    ) * 1000

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    results = {
        "model": "Scratch Transformer",
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "macro_f1": macro_f1,
        "confusion_matrix": matrix.tolist(),
        "average_inference_latency_ms_per_review": avg_latency_ms,
        "parameters": parameter_count,
        "test_samples": len(all_labels)
    }

    print("\n========== TEST RESULTS ==========")

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Macro F1:  {macro_f1:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")
    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=["Negative", "Positive"]
        )
    )

    print(f"Parameters: {parameter_count:,}")

    print(
        f"Average inference latency: "
        f"{avg_latency_ms:.4f} ms/review"
    )

    print(
        f"Total forward-pass time: "
        f"{total_inference_time:.2f} seconds"
    )

    with open(
        CHECKPOINT_DIR / "test_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print("\nTest results saved successfully!")


if __name__ == "__main__":

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    main()
