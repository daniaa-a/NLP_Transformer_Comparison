"""Fine-tune DistilBERT on the repository's shared IMDb split."""

import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import transformers

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import load_imdb_data
from model import ID2LABEL, LABEL2ID, MAX_LENGTH, PRETRAINED_CHECKPOINT, load_model, load_tokenizer

SEED = 42
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
CHECKPOINT_DIR = PROJECT_ROOT / "pretrained" / "checkpoints"
BEST_MODEL_DIR = CHECKPOINT_DIR / "best_model"


class ReviewDataset(Dataset):
    """Keep raw reviews in memory and tokenize only the current batch."""

    def __init__(self, frame):
        self.texts = frame["text"].astype(str).tolist()
        self.labels = frame["label"].astype(int).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]


def make_collator(tokenizer):
    def collate(examples):
        texts, labels = zip(*examples)
        encoded = tokenizer(
            list(texts), max_length=MAX_LENGTH, truncation=True, padding=True,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        return encoded
    return collate


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic kernels improve repeatability; some CUDA kernels may be slower.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _validate_split(train_df, val_df, test_df):
    expected = ((train_df, 20_000, "training"), (val_df, 5_000, "validation"), (test_df, 25_000, "test"))
    for frame, size, name in expected:
        if len(frame) != size:
            raise ValueError(f"Expected {size:,} {name} reviews, found {len(frame):,}.")
        counts = frame["label"].value_counts().to_dict()
        if counts != {0: size // 2, 1: size // 2}:
            raise ValueError(f"Unexpected {name} label distribution: {counts}")


def _ensure_output_paths_available():
    """Protect existing P2 artifacts; training must not silently replace a prior run."""
    protected = [
        BEST_MODEL_DIR,
        CHECKPOINT_DIR / "training_history.json",
        CHECKPOINT_DIR / "training_config.json",
    ]
    existing = [path for path in protected if path.exists()]
    if existing:
        paths = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "Existing P2 training artifacts are protected from overwrite. "
            "Choose or archive an output directory explicitly before training:\n" + paths
        )


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    sample_count = 0
    predictions, labels_all = [], []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in tqdm(loader, desc="Training" if training else "Validation", leave=False):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            if training:
                loss.backward()
                optimizer.step()
            count = batch["labels"].size(0)
            total_loss += loss.detach().item() * count
            sample_count += count
            predictions.extend(outputs.logits.argmax(dim=1).detach().cpu().tolist())
            labels_all.extend(batch["labels"].detach().cpu().tolist())
    return {
        "loss": total_loss / sample_count,
        "accuracy": accuracy_score(labels_all, predictions),
        "precision": precision_score(labels_all, predictions, average="macro", zero_division=0),
        "recall": recall_score(labels_all, predictions, average="macro", zero_division=0),
        "macro_f1": f1_score(labels_all, predictions, average="macro", zero_division=0),
    }


def main(pretrained_checkpoint=PRETRAINED_CHECKPOINT):
    set_seed()
    _ensure_output_paths_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; CUDA available: {torch.cuda.is_available()}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA unavailable; using explicit CPU fallback (training may be very slow).")

    config = {
        "model": "DistilBERT", "pretrained_checkpoint": str(pretrained_checkpoint),
        "num_labels": 2, "id2label": ID2LABEL, "label2id": LABEL2ID,
        "max_length": MAX_LENGTH, "train_batch_size": TRAIN_BATCH_SIZE,
        "evaluation_batch_size": EVAL_BATCH_SIZE, "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "optimizer": "AdamW", "seed": SEED, "device": str(device),
        "deterministic_cudnn": True, "mixed_precision": False,
        "python_version": platform.python_version(), "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__, "numpy_version": np.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    print("Training configuration:")
    print(json.dumps(config, indent=2))

    # The test frame is inspected only for its expected shape/count and never tokenized or passed to training.
    train_df, val_df, test_df = load_imdb_data()
    _validate_split(train_df, val_df, test_df)
    print("Verified shared split: 20,000 train, 5,000 validation, 25,000 test.")

    tokenizer = load_tokenizer(pretrained_checkpoint)
    collator = make_collator(tokenizer)
    train_loader = DataLoader(ReviewDataset(train_df), batch_size=TRAIN_BATCH_SIZE,
                              shuffle=True, num_workers=0, pin_memory=device.type == "cuda",
                              collate_fn=collator, generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(ReviewDataset(val_df), batch_size=EVAL_BATCH_SIZE,
                            shuffle=False, num_workers=0, pin_memory=device.type == "cuda",
                            collate_fn=collator)

    model = load_model(str(pretrained_checkpoint)).to(device)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.update({"parameters": total_params, "trainable_parameters": trainable_params,
                   "train_samples": len(train_df), "validation_samples": len(val_df), "test_samples": len(test_df)})
    (CHECKPOINT_DIR / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    history = []
    best_f1 = -1.0
    best_epoch = None
    total_start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.perf_counter()
        try:
            train_metrics = run_epoch(model, train_loader, device, optimizer)
            val_metrics = run_epoch(model, val_loader, device)
        except torch.cuda.OutOfMemoryError as exc:
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA ran out of memory during fine-tuning. Reduce TRAIN_BATCH_SIZE "
                "or MAX_LENGTH, record the changed configuration, then rerun."
            ) from exc
        epoch_seconds = time.perf_counter() - epoch_start
        row = {
            "epoch": epoch, "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"], "train_macro_f1": train_metrics["macro_f1"],
            "validation_loss": val_metrics["loss"], "validation_accuracy": val_metrics["accuracy"],
            "validation_precision": val_metrics["precision"], "validation_recall": val_metrics["recall"],
            "validation_macro_f1": val_metrics["macro_f1"], "epoch_duration_seconds": epoch_seconds,
        }
        history.append(row)
        print(json.dumps(row, indent=2))
        if val_metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = val_metrics["macro_f1"], epoch
            BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(BEST_MODEL_DIR)
            tokenizer.save_pretrained(BEST_MODEL_DIR)

    total_seconds = time.perf_counter() - total_start
    history_payload = {
        "epochs": history, "total_finetuning_time_seconds": total_seconds,
        "best_epoch": best_epoch, "best_validation_macro_f1": best_f1,
        "best_validation_accuracy": history[best_epoch - 1]["validation_accuracy"],
        "training_config": config,
    }
    (CHECKPOINT_DIR / "training_history.json").write_text(json.dumps(history_payload, indent=2), encoding="utf-8")
    print(f"Training complete in {total_seconds:.1f}s; best epoch {best_epoch}, validation macro-F1 {best_f1:.4f}.")
    print(f"Saved model and tokenizer to {BEST_MODEL_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pretrained-checkpoint", default=PRETRAINED_CHECKPOINT,
        help="Hugging Face model ID or local from_pretrained() directory.",
    )
    main(parser.parse_args().pretrained_checkpoint)
