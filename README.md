# NLP Transformer Comparison

This repository compares two PyTorch sentiment classifiers on the IMDb Large Movie Review Dataset:

- **Model A:** a custom Transformer encoder implemented with PyTorch attention and encoder layers.
- **Model B:** pretrained `distilbert-base-uncased`, fine-tuned with a two-class classification head.

The historical experiment is in `scratch/`, `pretrained/`, and the root `data_loader.py`. Its existing Model A checkpoint and metrics are historical artifacts. For the leakage-controlled comparison, use **`experiments/clean_v1/`**. The clean manifests group duplicate review text before splitting. Clean training has not been run; the checkpoint and test metric paths are created by training and evaluation.

Labels are `0 = Negative` and `1 = Positive`. The clean dataset has 19,824 training examples, 4,957 validation examples, and 24,801 official test examples. Counts differ from the original split because duplicate review texts were removed while keeping the official test set separate.

## Dataset setup

Download the IMDb Large Movie Review Dataset (`aclImdb_v1.tar.gz`) from the [Stanford AI Lab dataset page](https://ai.stanford.edu/~amaas/data/sentiment/). Extract it from the repository root so the layout is:

```text
data/raw/aclImdb/
├── train/
│   ├── neg/
│   ├── pos/
│   └── unsup/
└── test/
    ├── neg/
    └── pos/
```

The clean CSV manifests contain labels, hashes, and repository-relative paths to review files; they do not contain full review text. Keep the extracted dataset at `data/raw/aclImdb/`. `train/unsup` is never used. Raw data is excluded from Git and must be obtained separately on each computer. The source paths in the manifests are relative to the repository root, so run commands below from that root.

## Python and CUDA setup (Windows PowerShell)

Use 64-bit Python 3.12. Create and activate a project environment:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the CUDA-enabled PyTorch build separately. The tested environment used PyTorch `2.12.1+cu130`:

```powershell
python -m pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu130
```

This is the CUDA 13.0 wheel index; a compatible NVIDIA driver is still required. If this exact wheel is unavailable for the teammate's hardware/driver, select a supported CUDA build using the [official PyTorch installer](https://pytorch.org/get-started/locally/). Install this before the remaining dependencies so pip does not substitute a CPU-only PyTorch build.

Install the versions recorded from the working `.venv312` environment:

```powershell
python -m pip install -r experiments/clean_v1/requirements.txt
python -m pip check
```

Versions measured in the working environment (Python 3.12.0, Windows):

| Package | Version |
|---|---|
| PyTorch | `2.12.1+cu130` (CUDA build `13.0`) |
| Transformers | `5.24.1` |
| NumPy | `2.5.5` |
| pandas | `3.1.1` |
| scikit-learn | `1.9.1` |
| tqdm | `4.70.1` |
| tokenizers | `0.23.3` |
| safetensors | `0.8.0` |
| huggingface_hub | `2.4.0` |
| matplotlib | `3.11.2` |

The clean_v1 requirements file pins the packages used by its scripts and model download; pandas and matplotlib are recorded from the existing project environment but are not needed by the clean training scripts.

Verify PyTorch sees the GPU:

```powershell
python -c "import torch; print('torch', torch.__version__); print('CUDA build', torch.version.cuda); print('CUDA available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU detected')"
```

The root `requirements.txt` serves the older project files and includes an unpinned `torch` entry. For clean_v1, install its pinned requirements file above and preserve the already installed CUDA build.

## DistilBERT checkpoint

Download the base model into the repository-relative directory expected by clean_v1. This downloads weights and tokenizer files; it does not train the model:

```powershell
hf download distilbert/distilbert-base-uncased --local-dir models/distilbert-base-uncased
```

The clean training script loads this directory in offline/local-only mode. The model directory is ignored by Git because it contains large weight files. If the download command is not on `PATH`, use `python -m pip install huggingface_hub` or invoke `hf.exe` from `.venv312\Scripts`.

## Validate and run clean_v1

Run validation before training. It checks manifest hashes, source paths, counts, labels, and duplicate leakage:

```powershell
python experiments/clean_v1/validate_clean_dataset.py
```

Train and evaluate Model A (Scratch Transformer):

```powershell
python experiments/clean_v1/scratch/train.py
python experiments/clean_v1/scratch/evaluate.py
```

Train and evaluate Model B (DistilBERT):

```powershell
python experiments/clean_v1/pretrained/train.py --pretrained-checkpoint models/distilbert-base-uncased --micro-batch-size 8 --gradient-accumulation-steps 1
python experiments/clean_v1/pretrained/evaluate.py
```

The training scripts use seed 42, select the checkpoint using validation Macro-F1, and do not load the test manifest. Evaluation uses the held-out test set after training. Each training script refuses to overwrite a non-empty clean_v1 checkpoint directory; review or move an existing experiment output deliberately before starting a new run.

Outputs are kept separate from historical artifacts:

```text
experiments/clean_v1/scratch/checkpoints/
experiments/clean_v1/pretrained/checkpoints/
```

Each contains the best checkpoint, training history, configuration, and (after evaluation only) `test_metrics.json` as applicable. Do not expect metric files before completing training and evaluation.

## GPU memory and run order

The tested machine had an RTX 3050 Laptop GPU with 4 GB VRAM. DistilBERT at sequence length 256, physical batch 8, FP32 completed a forward/backward/optimizer-step smoke test; peak usage was 1,302 MiB allocated and 1,376 MiB reserved. Available memory varies with other applications. Run the two full trainings **sequentially**, never concurrently, so they do not compete for GPU memory.

If DistilBERT reports CUDA out-of-memory, preserve effective batch size 8 with:

```powershell
python experiments/clean_v1/pretrained/train.py --pretrained-checkpoint models/distilbert-base-uncased --micro-batch-size 4 --gradient-accumulation-steps 2
```

If that still does not fit, try `--fp16` and then `--gradient-checkpointing` (either or both). Scratch training uses batch size 16 and has its own CUDA out-of-memory message; do not run the experiments at the same time.

## References

- [IMDb Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- [DistilBERT model](https://huggingface.co/distilbert/distilbert-base-uncased)
- [PyTorch installation selector](https://pytorch.org/get-started/locally/)
- [Hugging Face download CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli)

Do not commit the raw IMDb dataset, local model weights, virtual environments, or temporary caches. The dataset is distributed by its authors; consult its source page and terms.
