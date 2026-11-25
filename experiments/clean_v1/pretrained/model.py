"""Local/offline DistilBERT helpers for the isolated clean_v1 experiment."""

from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

PRETRAINED_CHECKPOINT = "distilbert-base-uncased"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "distilbert-base-uncased"
ID2LABEL = {0: "NEGATIVE", 1: "POSITIVE"}
LABEL2ID = {label: index for index, label in ID2LABEL.items()}
NUM_LABELS = 2
MAX_LENGTH = 256


def validate_base_checkpoint(checkpoint, local_files_only=True):
    """Fail early unless a DistilBERT config, weights, and tokenizer are available."""
    path = Path(checkpoint)
    if not path.is_dir():
        if local_files_only:
            raise FileNotFoundError(f"Local DistilBERT checkpoint directory not found: {path}")
        return AutoConfig.from_pretrained(str(checkpoint), local_files_only=False)
    missing = []
    if not (path / "config.json").is_file():
        missing.append("config.json")
    if not any((path / filename).is_file() for filename in ("model.safetensors", "pytorch_model.bin")):
        missing.append("model.safetensors or pytorch_model.bin")
    if not any((path / filename).is_file() for filename in ("tokenizer.json", "vocab.txt")):
        missing.append("tokenizer.json or vocab.txt")
    if missing:
        raise FileNotFoundError(f"Checkpoint {path} is missing required files: {', '.join(missing)}")
    config = AutoConfig.from_pretrained(str(path), local_files_only=local_files_only)
    if config.model_type != "distilbert":
        raise ValueError(f"Expected a DistilBERT checkpoint; config model_type is {config.model_type!r}.")
    return config


def load_tokenizer(checkpoint=DEFAULT_CHECKPOINT_PATH, local_files_only=True):
    validate_base_checkpoint(checkpoint, local_files_only=local_files_only)
    return AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=local_files_only)


def load_model(checkpoint=DEFAULT_CHECKPOINT_PATH, local_files_only=True, **kwargs):
    validate_base_checkpoint(checkpoint, local_files_only=local_files_only)
    options = {
        "num_labels": NUM_LABELS,
        "id2label": ID2LABEL,
        "label2id": LABEL2ID,
        "local_files_only": local_files_only,
    }
    options.update(kwargs)
    if options.get("num_labels") != NUM_LABELS:
        raise ValueError("clean_v1 DistilBERT must have exactly two output labels.")
    if options.get("id2label") != ID2LABEL or options.get("label2id") != LABEL2ID:
        raise ValueError("clean_v1 label mapping must remain NEGATIVE=0, POSITIVE=1.")
    return AutoModelForSequenceClassification.from_pretrained(str(checkpoint), **options)


def load_finetuned_model(checkpoint, device=None, local_files_only=True):
    checkpoint = Path(checkpoint)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Fine-tuned clean_v1 checkpoint missing: {checkpoint}")
    model = AutoModelForSequenceClassification.from_pretrained(
        str(checkpoint), local_files_only=local_files_only,
    )
    if model.config.num_labels != NUM_LABELS:
        raise ValueError(f"Expected two output labels; found {model.config.num_labels}.")
    actual_mapping = {int(k): v for k, v in model.config.id2label.items()}
    if actual_mapping != ID2LABEL:
        raise ValueError(f"Unexpected checkpoint label mapping: {actual_mapping}")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)
