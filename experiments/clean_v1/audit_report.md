# clean_v1 Stage 1: Project and Artifact Audit

Date: 2026-11-25

## Scope and preservation

The relevant original source files, requirements, ignore rules, and checkpoint directories were inspected before clean-experiment implementation. The historical experiment remains under `scratch/` and the original `data_loader.py`; the new experiment will live only under `experiments/clean_v1/`. No original checkpoint or dataset file was changed during this audit.

## Existing data behavior

`data_loader.py` reads sorted `.txt` files from `train/neg`, `train/pos`, `test/neg`, and `test/pos`, mapping negative to 0 and positive to 1. It excludes `train/unsup` by only reading the two labeled sentiment folders. It creates train and validation using `train_test_split(test_size=0.20, random_state=42, stratify=full_train["label"], shuffle=True)`, then returns the official test folder unchanged. Its `DATA_DIR` is relative to the process working directory. The original split has documented exact-review duplication across boundaries; it is unsuitable for clean_v1.

## Model A source audit

- `scratch/model.py`: custom PyTorch encoder with token embeddings, learned positional embeddings, custom scaled dot-product attention and multi-head attention, feed-forward blocks, residual connections, LayerNorm, dropout, padding masks, masked mean pooling, and a two-class head. It does not use `nn.TransformerEncoder` or `nn.MultiheadAttention`.
- `scratch/tokenizer.py`: lowercase regex tokenizer; vocabulary is fitted from the training text passed by its caller; PAD=0 and UNK=1; encodes to a fixed maximum length.
- `scratch/train.py`: vocabulary 30,000; sequence length 256; embedding 128; 4 heads; hidden size 256; 3 layers; dropout 0.1; batch 16; 6 epochs; AdamW at 0.0003; seed 42. It uses the old loader, requires CUDA, and writes directly to `scratch/checkpoints/` (including creating that directory on import). It must not be used for clean_v1 training.
- `scratch/evaluate.py`: loads the old checkpoint/tokenizer, uses the old loader's test split, requires CUDA, and reports batched model timing. It writes old test metrics; it must not be used for clean_v1 evaluation.

The clean implementation will reuse the inspected architecture and tokenizer in new files and use only clean_v1 manifests and output paths.

## Model B source audit

- Existing root `pretrained/model.py` uses `AutoTokenizer` and `AutoModelForSequenceClassification`, defines `distilbert-base-uncased`, two labels, and the NEGATIVE=0/POSITIVE=1 mapping. It includes local fine-tuned checkpoint loading.
- Existing root `pretrained/train.py` uses the old `data_loader`, saves under `pretrained/checkpoints/`, and has the expected 256 length, batch 8, 3 epochs, learning rate 2e-5, weight decay 0.01, AdamW, seed 42, and validation Macro-F1 selection. It cannot be used unchanged for clean_v1 because it loads the old split and writes to the old artifact location.
- Existing root `pretrained/evaluate.py` also loads the old data split and saves results in the old pretrained checkpoint directory. It is not the clean_v1 evaluator.
- `pretrained/validate_dataset.py` validates the old split, and correctly reports its duplicate leakage; it is not a clean-manifest validator.
- There is no populated `pretrained/checkpoints/` directory, so no prior Model B checkpoint/history/metrics currently need preservation.

## Historical Model A artifacts

These values were measured before clean_v1 work and will be checked again after it:

| File | SHA-256 |
|---|---|
| `scratch/checkpoints/best_model.pth` | `b949e25eaaf917afd1965dba5689617fa0922bd5fe21fe2e0c469de995874b8e` |
| `scratch/checkpoints/tokenizer.json` | `f6636f427966f013d0da3bb2b672b55a9f08334d1d3ce6860d8260d6c674aea0` |
| `scratch/checkpoints/training_history.json` | `a92e05ba85fce1851b824ff60170ead8926f730aa73c23a246e1cce5ebbeb221` |
| `scratch/checkpoints/test_metrics.json` | `6c7323b2b3fdb71c576d6c0a327af15f951f25a79247116a9efc9f682315459d` |

Historical metrics record 84.164% accuracy, 0.84162 Macro-F1, 4,270,722 parameters, and 25,000 original-split test examples. These are not clean_v1 results.

## Environment and pretrained-model audit

- At the initial audit, Python 3.12.0 was installed outside the project environment; its pip was 23.2.1. It currently has none of torch, Transformers, NumPy, pandas, scikit-learn, or tqdm installed.
- The active Python 3.14 environment has PyTorch 2.11.0+cpu, Transformers 5.6.2, NumPy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0, and tqdm 4.67.3. Imports pass there, but that environment is not the requested Python 3.12 experiment environment and PyTorch reports no CUDA runtime/device.
- `nvidia-smi` detects an NVIDIA GeForce RTX 3050 Laptop GPU with 4,096 MiB VRAM, driver 591.86, and CUDA driver support 13.1. This does not establish that a CUDA-enabled PyTorch build works.
- The Hugging Face cache contains `distilbert-base-cased`, but not the required `distilbert-base-uncased`. A previous Hub request failed with Windows `WinError 10061`; the uncased model must be downloaded or supplied locally before the DistilBERT smoke test.
- Current official PyTorch instructions support Python 3.9–3.12 on Windows. Official previous-version commands list a Python 3.12 Windows CUDA 13.0 wheel for PyTorch 2.11.0. The reported NVIDIA driver is newer than the CUDA 13.0 compatibility example minimum, but CUDA operation must still be verified with `torch.cuda.is_available()` in the new environment.
- The isolated `.venv312` was created, but it has only pip 23.2.1 installed. This process has `PIP_NO_INDEX=1`; pip reports “Ignoring indexes” even when passed the official CUDA wheel index. Package installation requires an environment with approved package-index access or locally supplied wheels. This is a Stage 4 stop condition.

## Risks and required clean_v1 adaptations

1. Keep all clean data manifests, code, and model artifacts under `experiments/clean_v1/`.
2. Derive the test manifest exclusively from unique official-test content; remove all matching normalized hashes from eligible original training content before splitting.
3. Abort dataset generation on any conflicting-label normalized hash.
4. Reconstruct the train/validation assignment deterministically and validate manifest order, path readability, class counts, hashes, and zero within/across-split duplicate hashes before training.
5. Fit the scratch vocabulary only on the clean training manifest.
6. Make both clean model pipelines load the same manifest rows and record manifest hashes in configuration.
7. The 4 GB GPU is memory-constrained. DistilBERT needs a real CUDA/AMP and effective-batch smoke test before full fine-tuning.
8. The Python 3.12 environment and uncased model weights are not yet available. Full training is prohibited until data validation, environment verification, and both model smoke tests pass.

## Stage 1 status

**COMPLETED.** The audit is recorded. No historical data loader, Model A source, checkpoint, tokenizer, history, or metrics were modified. Dataset construction and clean training remain subsequent stages.
