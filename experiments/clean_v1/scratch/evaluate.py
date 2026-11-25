"""Evaluate a trained clean_v1 Scratch checkpoint on the clean official-test manifest."""

import json
import sys
import time
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CLEAN_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, CLEAN_ROOT, CLEAN_ROOT / "scratch"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from clean_data_loader import load_clean_test
from model import ScratchTransformer
from tokenizer import ScratchTokenizer

CHECKPOINT_DIR = CLEAN_ROOT / "scratch" / "checkpoints"
EVAL_BATCH_SIZE = 8


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    checkpoint_path = CHECKPOINT_DIR / "best_model.pth"
    tokenizer_path = CHECKPOINT_DIR / "tokenizer.json"
    if not checkpoint_path.is_file() or not tokenizer_path.is_file():
        raise FileNotFoundError("Train clean_v1 Scratch before evaluating its test manifest.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    word_to_id = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    tokenizer = ScratchTokenizer(vocab_size=len(word_to_id), max_length=config["max_length"])
    tokenizer.word_to_id = {word: int(index) for word, index in word_to_id.items()}
    tokenizer.id_to_word = {index: word for word, index in tokenizer.word_to_id.items()}

    test_rows = load_clean_test()
    x_test = tokenizer.encode_batch([row["text"] for row in test_rows])
    y_test = torch.tensor([row["label"] for row in test_rows], dtype=torch.long)
    loader = DataLoader(TensorDataset(x_test, y_test), batch_size=EVAL_BATCH_SIZE, shuffle=False)
    model = ScratchTransformer(
        vocab_size=checkpoint["vocab_size"], embedding_dim=config["embedding_dim"],
        num_heads=config["num_heads"], hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"], max_length=config["max_length"],
        num_classes=config["num_classes"], dropout=config["dropout"],
        padding_idx=config.get("padding_idx", 0),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions, labels = [], []
    # Warm-up is outside the measured pass.
    warmup_x, _ = next(iter(loader))
    with torch.inference_mode():
        model(warmup_x[:1].to(device))
    sync(device)
    total_forward_seconds = 0.0
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            sync(device)
            started = time.perf_counter()
            logits = model(inputs)
            sync(device)
            total_forward_seconds += time.perf_counter() - started
            predictions.extend(logits.argmax(1).cpu().tolist())
            labels.extend(targets.tolist())

    # Batch-size-1 latency is a separate full-test forward pass.
    single_seconds = 0.0
    with torch.inference_mode():
        for index in range(len(test_rows)):
            x = x_test[index:index + 1].to(device)
            if index == 0:
                model(x)
                sync(device)
            sync(device)
            started = time.perf_counter()
            model(x)
            sync(device)
            single_seconds += time.perf_counter() - started

    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    result = {
        "model": "Scratch Transformer", "dataset_version": "clean_v1",
        "manifest_sha256": checkpoint["manifest_sha256"],
        "accuracy": accuracy_score(labels, predictions),
        "precision_macro": precision_score(labels, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, predictions, average="macro", zero_division=0),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            labels, predictions, labels=[0, 1], target_names=["Negative", "Positive"],
            output_dict=True, zero_division=0,
        ),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "test_samples": len(labels), "device": str(device), "batch_size": EVAL_BATCH_SIZE,
        "total_forward_pass_time_seconds": total_forward_seconds,
        "batch_averaged_latency_ms_per_review": total_forward_seconds / len(labels) * 1000,
        "batch_size_1_total_forward_pass_time_seconds": single_seconds,
        "batch_size_1_latency_ms_per_review": single_seconds / len(labels) * 1000,
        "timing_method": "Tokenization/encoding excluded; one warm-up; perf_counter around model forwards; CUDA synchronized before/after each timed pass.",
    }
    output_path = CHECKPOINT_DIR / "test_metrics.json"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite clean_v1 test metrics: {output_path}")
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
