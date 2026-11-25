"""Evaluate the saved best DistilBERT checkpoint on the shared IMDb test set."""

import json
import sys
import time
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRETRAINED_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PRETRAINED_DIR) not in sys.path:
    sys.path.insert(0, str(PRETRAINED_DIR))

from data_loader import load_imdb_data
from model import MAX_LENGTH, PRETRAINED_CHECKPOINT, load_finetuned_model, load_tokenizer

CHECKPOINT_DIR = PRETRAINED_DIR / "checkpoints"
BEST_MODEL_DIR = CHECKPOINT_DIR / "best_model"
BATCH_SIZE = 8


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _parameter_counts(model):
    return (
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )


def main():
    if not BEST_MODEL_DIR.is_dir():
        raise FileNotFoundError(
            f"Fine-tuned checkpoint not found at {BEST_MODEL_DIR}. Run `python pretrained/train.py` first."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    _, _, test_df = load_imdb_data()
    counts = test_df["label"].value_counts().to_dict()
    if len(test_df) != 25_000 or counts != {0: 12_500, 1: 12_500}:
        raise ValueError(f"Expected the shared 25,000-example balanced test split; got {len(test_df)} and {counts}.")
    labels = test_df["label"].astype(int).tolist()

    tokenizer = load_tokenizer(BEST_MODEL_DIR)
    model = load_finetuned_model(BEST_MODEL_DIR, device=device)
    model.eval()

    # Tokenize before timing. Store CPU tensors so measured intervals cover model forward passes only.
    encoded_batches = []
    texts = test_df["text"].astype(str).tolist()
    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Tokenizing test set"):
        encoded = tokenizer(texts[start:start + BATCH_SIZE], max_length=MAX_LENGTH,
                            truncation=True, padding=True, return_tensors="pt")
        encoded_batches.append({key: value for key, value in encoded.items()})

    # Warm-up is excluded from all reported durations.
    warmup = encoded_batches[0]
    with torch.inference_mode():
        model(**{key: value.to(device) for key, value in warmup.items()})
    _sync(device)

    predictions = []
    total_forward_seconds = 0.0
    with torch.inference_mode():
        for batch in tqdm(encoded_batches, desc="Evaluating test set"):
            device_batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            _sync(device)
            start = time.perf_counter()
            logits = model(**device_batch).logits
            _sync(device)
            total_forward_seconds += time.perf_counter() - start
            predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())

    # Interactive single-review latency is measured separately with batch size one.
    single_total_seconds = 0.0
    with torch.inference_mode():
        for index, text in enumerate(tqdm(texts, desc="Timing batch-size-1 forwards")):
            # Encode each review before starting its timer, avoiding a large duplicate token cache.
            encoded = tokenizer(text, max_length=MAX_LENGTH, truncation=True,
                                padding=True, return_tensors="pt")
            batch = {key: value for key, value in encoded.items()}
            device_batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            if index == 0:
                model(**device_batch)  # untimed single-review warm-up
                _sync(device)
            _sync(device)
            start = time.perf_counter()
            model(**device_batch)
            _sync(device)
            single_total_seconds += time.perf_counter() - start

    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    report = classification_report(labels, predictions, labels=[0, 1],
                                   target_names=["Negative", "Positive"], output_dict=True,
                                   zero_division=0)
    total_params, trainable_params = _parameter_counts(model)
    results = {
        "model": "DistilBERT", "pretrained_checkpoint": PRETRAINED_CHECKPOINT,
        "accuracy": accuracy_score(labels, predictions),
        "precision_macro": precision_score(labels, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, predictions, average="macro", zero_division=0),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "confusion_matrix": matrix.tolist(), "classification_report": report,
        "parameters": total_params, "trainable_parameters": trainable_params,
        "test_samples": len(labels), "total_forward_pass_time_seconds": total_forward_seconds,
        "average_inference_latency_ms_per_review": total_forward_seconds / len(labels) * 1000,
        "batch_size_1_total_forward_pass_time_seconds": single_total_seconds,
        "batch_size_1_latency_ms_per_review": single_total_seconds / len(labels) * 1000,
        "batch_averaged_latency_ms_per_review": total_forward_seconds / len(labels) * 1000,
        "inference_batch_size": BATCH_SIZE, "device": str(device),
        "timing_method": "Tokenization excluded; one untimed warm-up; perf_counter around forward only; CUDA synchronized before and after each batch; batch-size-1 measured separately over all test reviews.",
    }
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = CHECKPOINT_DIR / "test_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved actual test metrics to {metrics_path}")


if __name__ == "__main__":
    main()
