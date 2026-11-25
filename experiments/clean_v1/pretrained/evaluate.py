"""Evaluate a trained clean_v1 DistilBERT checkpoint on the clean test manifest."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CLEAN_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, CLEAN_ROOT, CLEAN_ROOT / "pretrained"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from clean_data_loader import load_clean_test, load_dataset_summary
from model import MAX_LENGTH, PRETRAINED_CHECKPOINT, load_finetuned_model, load_tokenizer

CHECKPOINT_DIR = CLEAN_ROOT / "pretrained" / "checkpoints"
BEST_MODEL_DIR = CHECKPOINT_DIR / "best_model"
BATCH_SIZE = 8


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    output_path = CHECKPOINT_DIR / "test_metrics.json"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite clean_v1 test metrics: {output_path}")
    if not BEST_MODEL_DIR.is_dir():
        raise FileNotFoundError(f"Train clean_v1 DistilBERT before final evaluation: {BEST_MODEL_DIR}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(BEST_MODEL_DIR, local_files_only=True)
    model = load_finetuned_model(BEST_MODEL_DIR, device=device, local_files_only=True).eval()
    test_rows = load_clean_test()  # Test manifest is read only by this post-training evaluator.
    summary = load_dataset_summary()
    labels = [row["label"] for row in test_rows]
    texts = [row["text"] for row in test_rows]

    encoded_batches = []
    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Tokenizing clean test"):
        encoded = tokenizer(texts[start:start + BATCH_SIZE], max_length=MAX_LENGTH,
                            truncation=True, padding=True, return_tensors="pt")
        encoded_batches.append(dict(encoded))
    with torch.inference_mode():
        model(**{key: value.to(device) for key, value in encoded_batches[0].items()})
    sync(device)

    predictions = []
    batch_forward_seconds = 0.0
    with torch.inference_mode():
        for encoded in tqdm(encoded_batches, desc="Evaluating clean test"):
            values = {key: value.to(device, non_blocking=True) for key, value in encoded.items()}
            sync(device)
            started = time.perf_counter()
            logits = model(**values).logits
            sync(device)
            batch_forward_seconds += time.perf_counter() - started
            predictions.extend(logits.argmax(dim=1).cpu().tolist())

    single_forward_seconds = 0.0
    with torch.inference_mode():
        for index, text in enumerate(tqdm(texts, desc="Timing batch-size-1")):
            encoded = tokenizer(text, max_length=MAX_LENGTH, truncation=True,
                                padding=True, return_tensors="pt")
            values = {key: value.to(device, non_blocking=True) for key, value in encoded.items()}
            if index == 0:
                model(**values)
                sync(device)
            sync(device)
            started = time.perf_counter()
            model(**values)
            sync(device)
            single_forward_seconds += time.perf_counter() - started

    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    result = {
        "model": "DistilBERT", "dataset_version": "clean_v1",
        "pretrained_checkpoint": PRETRAINED_CHECKPOINT,
        "manifest_sha256": summary["manifest_sha256"],
        "accuracy": accuracy_score(labels, predictions),
        "precision_macro": precision_score(labels, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, predictions, average="macro", zero_division=0),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            labels, predictions, labels=[0, 1], target_names=["Negative", "Positive"],
            output_dict=True, zero_division=0,
        ),
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "test_samples": len(labels), "device": str(device), "inference_batch_size": BATCH_SIZE,
        "total_forward_pass_time_seconds": batch_forward_seconds,
        "batch_averaged_latency_ms_per_review": batch_forward_seconds / len(labels) * 1000,
        "batch_size_1_total_forward_pass_time_seconds": single_forward_seconds,
        "batch_size_1_latency_ms_per_review": single_forward_seconds / len(labels) * 1000,
        "timing_method": "Tokenization excluded from timed forwards; one warm-up; perf_counter timing; CUDA synchronized before and after every timed forward batch/review.",
    }
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
