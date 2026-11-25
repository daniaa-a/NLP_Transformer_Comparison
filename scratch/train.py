
import sys
import json
import random
import time

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# Make project root accessible
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_loader import load_imdb_data
from model import ScratchTransformer
from tokenizer import ScratchTokenizer


# ==========================================
# CONFIGURATION
# ==========================================

SEED = 42

VOCAB_SIZE = 30000
MAX_LENGTH = 256

EMBEDDING_DIM = 128
NUM_HEADS = 4
HIDDEN_DIM = 256
NUM_LAYERS = 3
DROPOUT = 0.1

BATCH_SIZE = 16
EPOCHS = 6
LEARNING_RATE = 0.0003

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_DIR = PROJECT_ROOT / "scratch" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# REPRODUCIBILITY
# ==========================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==========================================
# DATA PREPARATION
# ==========================================

def prepare_data():

    print("\nLoading IMDb dataset...")

    train_df, val_df, test_df = load_imdb_data()

    print("Training reviews:", len(train_df))
    print("Validation reviews:", len(val_df))
    print("Test reviews:", len(test_df))

    print("\nBuilding custom vocabulary...")

    tokenizer = ScratchTokenizer(
        vocab_size=VOCAB_SIZE,
        max_length=MAX_LENGTH
    )

    # Fit vocabulary ONLY on training reviews
    tokenizer.fit(train_df["text"])

    print("\nEncoding reviews...")

    X_train = tokenizer.encode_batch(
        train_df["text"].tolist()
    )

    X_val = tokenizer.encode_batch(
        val_df["text"].tolist()
    )

    y_train = torch.tensor(
        train_df["label"].values,
        dtype=torch.long
    )

    y_val = torch.tensor(
        val_df["label"].values,
        dtype=torch.long
    )

    train_dataset = TensorDataset(
        X_train,
        y_train
    )

    val_dataset = TensorDataset(
        X_val,
        y_val
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return tokenizer, train_loader, val_loader


# ==========================================
# MODEL INITIALIZATION
# ==========================================

def create_model(vocab_size):

    model = ScratchTransformer(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        max_length=MAX_LENGTH,
        num_classes=2,
        dropout=DROPOUT,
        padding_idx=0
    )

    return model.to(DEVICE)


# ==========================================
# TRAINING FUNCTION
# ==========================================

def train_one_epoch(model, loader, criterion, optimizer):

    model.train()

    total_loss = 0
    all_predictions = []
    all_labels = []

    progress = tqdm(
        loader,
        desc="Training",
        leave=False
    )

    for inputs, labels in progress:

        inputs = inputs.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.detach().cpu().tolist()
        )

        all_labels.extend(
            labels.detach().cpu().tolist()
        )

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    avg_loss = total_loss / len(loader)

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro"
    )

    return avg_loss, accuracy, macro_f1


# ==========================================
# VALIDATION FUNCTION
# ==========================================

def validate(model, loader, criterion):

    model.eval()

    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for inputs, labels in tqdm(
            loader,
            desc="Validation",
            leave=False
        ):

            inputs = inputs.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

    avg_loss = total_loss / len(loader)

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro"
    )

    return avg_loss, accuracy, macro_f1


# ==========================================
# MAIN TRAINING PIPELINE
# ==========================================

def main():

    print("\n================================")
    print("SCRATCH TRANSFORMER TRAINING")
    print("================================")

    print("Device:", DEVICE)

    if DEVICE.type != "cuda":
        raise RuntimeError(
            "CUDA GPU is not available. Training stopped."
        )

    print("GPU:", torch.cuda.get_device_name(0))

    tokenizer, train_loader, val_loader = prepare_data()

    model = create_model(
        vocab_size=len(tokenizer.word_to_id)
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print("\nModel parameters:", f"{parameter_count:,}")

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_f1 = -1.0

    total_start = time.time()

    history = []

    # ======================================
    # EPOCHS
    # ======================================

    for epoch in range(EPOCHS):

        epoch_start = time.time()

        print(f"\n========== EPOCH {epoch + 1}/{EPOCHS} ==========")

        train_loss, train_acc, train_f1 = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_acc, val_f1 = validate(
            model,
            val_loader,
            criterion
        )

        epoch_time = time.time() - epoch_start

        print(f"\nTraining Loss: {train_loss:.4f}")
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Training Macro F1: {train_f1:.4f}")

        print(f"\nValidation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print(f"Validation Macro F1: {val_f1:.4f}")

        print(f"Epoch Time: {epoch_time / 60:.2f} minutes")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "epoch_time_seconds": epoch_time
        })

        # Save best model based on validation Macro F1
        if val_f1 > best_f1:

            best_f1 = val_f1

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab_size": len(tokenizer.word_to_id),
                    "config": {
                        "embedding_dim": EMBEDDING_DIM,
                        "num_heads": NUM_HEADS,
                        "hidden_dim": HIDDEN_DIM,
                        "num_layers": NUM_LAYERS,
                        "max_length": MAX_LENGTH,
                        "num_classes": 2,
                        "dropout": DROPOUT
                    }
                },
                CHECKPOINT_DIR / "best_model.pth"
            )

            print("\nBest model checkpoint saved!")

    # ======================================
    # SAVE TOKENIZER AND HISTORY
    # ======================================

    with open(
        CHECKPOINT_DIR / "tokenizer.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            tokenizer.word_to_id,
            file,
            ensure_ascii=False
        )

    with open(
        CHECKPOINT_DIR / "training_history.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            history,
            file,
            indent=4
        )

    total_time = time.time() - total_start

    print("\n================================")
    print("TRAINING COMPLETED")
    print("================================")

    print(f"Total Training Time: {total_time / 60:.2f} minutes")
    print(f"Best Validation Macro F1: {best_f1:.4f}")

    print("\nSaved files:")
    print("scratch/checkpoints/best_model.pth")
    print("scratch/checkpoints/tokenizer.json")
    print("scratch/checkpoints/training_history.json")


if __name__ == "__main__":
    main()
