"""Fine-tune DistilBERT on clean_v1 train/validation manifests only."""

from __future__ import annotations

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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CLEAN_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, CLEAN_ROOT, CLEAN_ROOT / "pretrained"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from clean_data_loader import load_clean_train_validation, load_dataset_summary
from model import (
    DEFAULT_CHECKPOINT_PATH, ID2LABEL, LABEL2ID, MAX_LENGTH, PRETRAINED_CHECKPOINT,
    load_model, load_tokenizer,
)

SEED = 42
EFFECTIVE_BATCH_SIZE = 8
MICRO_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 1
EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
CHECKPOINT_DIR = CLEAN_ROOT / "pretrained" / "checkpoints"
BEST_MODEL_DIR = CHECKPOINT_DIR / "best_model"


class ReviewDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        return row["text"], row["label"]


def make_collator(tokenizer):
    def collate(batch):
        texts, labels = zip(*batch)
        values = tokenizer(
            list(texts), max_length=MAX_LENGTH, truncation=True,
            padding=True, return_tensors="pt",
        )
        values["labels"] = torch.tensor(labels, dtype=torch.long)
        return values
    return collate


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _protect_outputs():
    if CHECKPOINT_DIR.exists() and any(CHECKPOINT_DIR.iterdir()):
        raise FileExistsError(f"Refusing to overwrite clean_v1 Model B artifacts: {CHECKPOINT_DIR}")


def _metrics(labels, predictions):
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision_macro": precision_score(labels, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, predictions, average="macro", zero_division=0),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
    }


def run_epoch(model, loader, device, optimizer=None, accumulation_steps=1,
              amp=False, scaler=None, desc="Train"):
    training = optimizer is not None
    model.train(training)
    labels_all, predictions_all = [], []
    loss_sum = 0.0
    total_batches = len(loader)
    if training:
        optimizer.zero_grad(set_to_none=True)
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for step, batch in enumerate(tqdm(loader, desc=desc, leave=False)):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            active_group_size = min(
                accumulation_steps,
                total_batches - (step // accumulation_steps) * accumulation_steps,
            )
            amp_context = torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"
            )
            with amp_context:
                outputs = model(**batch)
                loss = outputs.loss
            if training:
                scaled_loss = loss / active_group_size
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                should_step = ((step + 1) % accumulation_steps == 0) or (step + 1 == total_batches)
                if should_step:
                    if scaler is not None and scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            count = batch["labels"].size(0)
            loss_sum += loss.detach().float().item() * count
            predictions_all.extend(outputs.logits.argmax(dim=1).detach().cpu().tolist())
            labels_all.extend(batch["labels"].detach().cpu().tolist())
    result = _metrics(labels_all, predictions_all)
    result["loss"] = loss_sum / len(labels_all)
    return result


def main(checkpoint=DEFAULT_CHECKPOINT_PATH, micro_batch_size=MICRO_BATCH_SIZE,
         accumulation_steps=GRADIENT_ACCUMULATION_STEPS, amp=False,
         gradient_checkpointing=False):
    if micro_batch_size * accumulation_steps != EFFECTIVE_BATCH_SIZE:
        raise ValueError(
            f"Physical batch {micro_batch_size} x accumulation {accumulation_steps} "
            f"must equal effective batch {EFFECTIVE_BATCH_SIZE}."
        )
    set_seed()
    _protect_outputs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Device: {device}; GPU: {torch.cuda.get_device_name(device)}")
        print(f"GPU memory: {torch.cuda.mem_get_info(device)} (free,total bytes)")
    else:
        print("CUDA unavailable; explicit CPU fallback.")

    # Deliberately never loads the held-out test manifest here.
    train_rows, validation_rows = load_clean_train_validation()
    tokenizer = load_tokenizer(checkpoint, local_files_only=True)
    collator = make_collator(tokenizer)
    train_loader = DataLoader(
        ReviewDataset(train_rows), batch_size=micro_batch_size, shuffle=True,
        num_workers=0, pin_memory=device.type == "cuda", collate_fn=collator,
        generator=torch.Generator().manual_seed(SEED),
    )
    validation_loader = DataLoader(
        ReviewDataset(validation_rows), batch_size=micro_batch_size, shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda", collate_fn=collator,
    )

    model = load_model(checkpoint, local_files_only=True).to(device)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")

    summary = load_dataset_summary()
    config = {
        "experiment": "clean_v1", "model": "DistilBERT",
        "pretrained_checkpoint": PRETRAINED_CHECKPOINT,
        "pretrained_checkpoint_path": str(Path(checkpoint).resolve()),
        "num_labels": 2, "id2label": ID2LABEL, "label2id": LABEL2ID,
        "max_length": MAX_LENGTH, "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "physical_micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "mixed_precision_fp16": bool(amp and device.type == "cuda"),
        "gradient_checkpointing": gradient_checkpointing,
        "epochs": EPOCHS, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "optimizer": "AdamW", "seed": SEED, "best_checkpoint_metric": "validation_macro_f1",
        "device": str(device), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_runtime": torch.version.cuda, "python_version": platform.python_version(),
        "pytorch_version": torch.__version__, "transformers_version": transformers.__version__,
        "numpy_version": np.__version__,
        "train_samples": len(train_rows), "validation_samples": len(validation_rows),
        "manifest_sha256": summary["manifest_sha256"],
        "parameters": total_params, "trainable_parameters": trainable_params,
    }
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    (CHECKPOINT_DIR / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    history = []
    best_f1 = -1.0
    best_epoch = None
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.perf_counter()
        try:
            training = run_epoch(
                model, train_loader, device, optimizer=optimizer,
                accumulation_steps=accumulation_steps, amp=amp, scaler=scaler,
                desc=f"Train {epoch}/{EPOCHS}",
            )
            validation = run_epoch(model, validation_loader, device, desc=f"Validation {epoch}/{EPOCHS}")
        except torch.cuda.OutOfMemoryError as exc:
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA OOM. Do not reduce effective batch size; retry with micro-batch 4 and accumulation 2, "
                "then consider FP16 or gradient checkpointing if needed."
            ) from exc
        row = {
            "epoch": epoch, "train_loss": training["loss"], "train_accuracy": training["accuracy"],
            "train_macro_f1": training["macro_f1"], "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "validation_precision": validation["precision_macro"],
            "validation_recall": validation["recall_macro"],
            "validation_macro_f1": validation["macro_f1"],
            "epoch_duration_seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        print(json.dumps(row))
        if validation["macro_f1"] > best_f1:
            best_f1, best_epoch = validation["macro_f1"], epoch
            BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(BEST_MODEL_DIR)
            tokenizer.save_pretrained(BEST_MODEL_DIR)

    total_seconds = time.perf_counter() - started
    (CHECKPOINT_DIR / "training_history.json").write_text(json.dumps({
        "epochs": history, "total_training_time_seconds": total_seconds,
        "best_epoch": best_epoch, "best_validation_macro_f1": best_f1,
        "training_config": config,
    }, indent=2), encoding="utf-8")
    print(f"clean_v1 DistilBERT training complete: best epoch {best_epoch}, val macro-F1={best_f1:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained-checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--micro-batch-size", type=int, default=MICRO_BATCH_SIZE)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()
    main(args.pretrained_checkpoint, args.micro_batch_size,
         args.gradient_accumulation_steps, args.fp16, args.gradient_checkpointing)
