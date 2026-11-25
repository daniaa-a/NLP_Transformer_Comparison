"""Reusable DistilBERT loading helpers for IMDb sentiment classification."""

from pathlib import Path
from typing import Union

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PRETRAINED_CHECKPOINT = "distilbert-base-uncased"
NUM_LABELS = 2
MAX_LENGTH = 256
ID2LABEL = {0: "NEGATIVE", 1: "POSITIVE"}
LABEL2ID = {label: index for index, label in ID2LABEL.items()}
PathLike = Union[str, Path]


def load_tokenizer(checkpoint: PathLike = PRETRAINED_CHECKPOINT, **kwargs):
    """Load the base tokenizer or a tokenizer saved with a fine-tuned model."""
    return AutoTokenizer.from_pretrained(str(checkpoint), **kwargs)


def load_model(checkpoint: str = PRETRAINED_CHECKPOINT, **kwargs):
    """Initialize a two-label classifier from a pretrained checkpoint."""
    options = {
        "num_labels": NUM_LABELS,
        "id2label": ID2LABEL,
        "label2id": LABEL2ID,
    }
    options.update(kwargs)
    if options["num_labels"] != NUM_LABELS:
        raise ValueError(f"This sentiment classifier requires exactly {NUM_LABELS} labels.")
    if options["id2label"] != ID2LABEL or options["label2id"] != LABEL2ID:
        raise ValueError("The sentiment label mapping must remain NEGATIVE=0 and POSITIVE=1.")
    return AutoModelForSequenceClassification.from_pretrained(checkpoint, **options)


def load_finetuned_model(
    checkpoint: PathLike,
    device: Union[str, torch.device, None] = None,
    **kwargs,
):
    """Load a locally saved fine-tuned model and place it on the selected device."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Fine-tuned checkpoint directory not found: {checkpoint}")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint), **kwargs)
    if model.config.num_labels != NUM_LABELS:
        raise ValueError(f"Expected {NUM_LABELS} labels, found {model.config.num_labels}")
    id2label = {int(index): label for index, label in model.config.id2label.items()}
    if id2label != ID2LABEL:
        raise ValueError(f"Unexpected fine-tuned checkpoint label mapping: {id2label}")
    model.to(device)
    return model
