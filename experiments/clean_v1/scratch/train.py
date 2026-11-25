"""Train the custom Transformer only on clean_v1 train/validation manifests."""

from __future__ import annotations

import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CLEAN_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, CLEAN_ROOT, CLEAN_ROOT / "scratch"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from clean_data_loader import load_clean_train_validation, load_dataset_summary
from model import ScratchTransformer
from tokenizer import ScratchTokenizer

SEED = 42
VOCAB_SIZE = 30_000
MAX_LENGTH = 256
EMBEDDING_DIM = 128
NUM_HEADS = 4
HIDDEN_DIM = 256
NUM_LAYERS = 3
DROPOUT = 0.1
BATCH_SIZE = 16
EPOCHS = 6
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.01
CHECKPOINT_DIR = CLEAN_ROOT / "scratch" / "checkpoints"


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _protect_existing_outputs():
    if CHECKPOINT_DIR.exists() and any(CHECKPOINT_DIR.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing clean_v1 Model A artifacts: {CHECKPOINT_DIR}")


def _metrics(labels, predictions):
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision_macro": precision_score(labels, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, predictions, average="macro", zero_division=0),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
    }


def run_epoch(model, loader, device, optimizer=None, criterion=None, desc="Train"):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels_all, predictions_all = [], []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for inputs, labels in tqdm(loader, desc=desc, leave=False):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.detach().item() * labels.size(0)
            predictions_all.extend(logits.argmax(dim=1).detach().cpu().tolist())
            labels_all.extend(labels.detach().cpu().tolist())
    result = _metrics(labels_all, predictions_all)
    result["loss"] = total_loss / len(labels_all)
    return result


def main():
    set_seed()
    _protect_existing_outputs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.get_device_name(device)

    # This API deliberately does not load the held-out test manifest.
    train_rows, validation_rows = load_clean_train_validation()
    train_texts = [row["text"] for row in train_rows]
    tokenizer = ScratchTokenizer(vocab_size=VOCAB_SIZE, max_length=MAX_LENGTH)
    tokenizer.fit(train_texts)  # Fit on clean training examples only.

    x_train = tokenizer.encode_batch(train_texts)
    y_train = torch.tensor([row["label"] for row in train_rows], dtype=torch.long)
    x_val = tokenizer.encode_batch([row["text"] for row in validation_rows])
    y_val = torch.tensor([row["label"] for row in validation_rows], dtype=torch.long)
    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(SEED),
    )
    validation_loader = DataLoader(
        TensorDataset(x_val, y_val), batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda",
    )

    model = ScratchTransformer(
        vocab_size=len(tokenizer.word_to_id), embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
        max_length=MAX_LENGTH, num_classes=2, dropout=DROPOUT, padding_idx=0,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    dataset_summary = load_dataset_summary()
    config = {
        "experiment": "clean_v1", "model": "Scratch Transformer",
        "seed": SEED, "device": str(device), "python_version": platform.python_version(),
        "pytorch_version": torch.__version__, "numpy_version": np.__version__,
        "sklearn_version": __import__("sklearn").__version__,
        "vocab_size_limit": VOCAB_SIZE, "actual_vocab_size": len(tokenizer.word_to_id),
        "max_length": MAX_LENGTH, "embedding_dim": EMBEDDING_DIM,
        "num_heads": NUM_HEADS, "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS,
        "dropout": DROPOUT, "batch_size": BATCH_SIZE, "epochs": EPOCHS,
        "optimizer": "AdamW", "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY, "loss": "CrossEntropyLoss",
        "best_checkpoint_metric": "validation_macro_f1",
        "train_samples": len(train_rows), "validation_samples": len(validation_rows),
        "manifest_sha256": dataset_summary["manifest_sha256"],
    }
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    (CHECKPOINT_DIR / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    history = []
    best_f1 = -1.0
    best_epoch = None
    start_all = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        start_epoch = time.perf_counter()
        try:
            training = run_epoch(model, train_loader, device, optimizer, criterion, f"Train {epoch}/{EPOCHS}")
            validation = run_epoch(model, validation_loader, device, criterion=criterion,
                                   desc=f"Validation {epoch}/{EPOCHS}")
        except torch.cuda.OutOfMemoryError as exc:
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            raise RuntimeError("CUDA OOM during clean_v1 Scratch training; lower the physical batch size and record it.") from exc
        row = {
            "epoch": epoch, "train_loss": training["loss"], "train_accuracy": training["accuracy"],
            "train_macro_f1": training["macro_f1"], "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"], "validation_precision": validation["precision_macro"],
            "validation_recall": validation["recall_macro"], "validation_macro_f1": validation["macro_f1"],
            "epoch_duration_seconds": time.perf_counter() - start_epoch,
        }
        history.append(row)
        print(json.dumps(row))
        if validation["macro_f1"] > best_f1:
            best_f1, best_epoch = validation["macro_f1"], epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab_size": len(tokenizer.word_to_id),
                "config": {"embedding_dim": EMBEDDING_DIM, "num_heads": NUM_HEADS,
                           "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS,
                           "max_length": MAX_LENGTH, "num_classes": 2, "dropout": DROPOUT,
                           "padding_idx": 0},
                "best_epoch": epoch, "best_validation_macro_f1": best_f1,
                "manifest_sha256": dataset_summary["manifest_sha256"],
            }, CHECKPOINT_DIR / "best_model.pth")

    total_seconds = time.perf_counter() - start_all
    tokenizer_path = CHECKPOINT_DIR / "tokenizer.json"
    tokenizer_path.write_text(json.dumps(tokenizer.word_to_id, ensure_ascii=False), encoding="utf-8")
    (CHECKPOINT_DIR / "training_history.json").write_text(json.dumps({
        "epochs": history, "total_training_time_seconds": total_seconds,
        "best_epoch": best_epoch, "best_validation_macro_f1": best_f1,
        "training_config": config,
    }, indent=2), encoding="utf-8")
    print(f"clean_v1 Scratch training complete: best epoch {best_epoch}, val macro-F1={best_f1:.6f}")


if __name__ == "__main__":
    main()
