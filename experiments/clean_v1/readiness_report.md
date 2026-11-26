# clean_v1 Readiness Report

Date: 2026-11-25 
Scope: verify clean_v1 readiness only. **No full training, final test evaluation, or comparison was run.** Smoke-test losses below are technical checks, not experiment metrics.

## Status summary

| Check | Status | Result |
|---|---|---|
| Python 3.12 environment and package consistency | **COMPLETED** | Python 3.12.0; `pip check`: no broken requirements |
| CUDA device/runtime detection | **COMPLETED** | RTX 3050 Laptop GPU, CUDA runtime 13.0, CUDA available |
| Clean dataset validation | **COMPLETED** | Expected counts; zero duplicate-hash overlap; validation passed |
| Shared manifest records for both models | **COMPLETED** | Both train scripts use the same `load_clean_train_validation()` function; both evaluators use `load_clean_test()` |
| Model A smoke test | **COMPLETED** | Custom forward/backward, pad behavior, optimizer step, checkpoint round-trip passed |
| Model B offline load and smoke test | **COMPLETED** | Encoder equality, two-class forward/backward, save/load round-trip on CUDA passed |
| DistilBERT batch-8 GPU memory test | **COMPLETED** | FP32 physical batch 8 passed; effective batch 8 |
| Clean training script review/import checks | **COMPLETED** | Train/evaluate modules compile and import; no test manifest loaded during training |
| Full training and final evaluation | **NOT TESTED** | Intentionally not started; awaiting review of this report |

## 1. Environment verification

| Item | Verified value |
|---|---|
| Python | 3.12.0 |
| Executable | `.venv312\Scripts\python.exe` |
| PyTorch | 2.12.1+cu130 |
| Transformers | 5.24.1 |
| NumPy | 2.5.5 |
| pandas | 3.1.1 |
| scikit-learn | 1.9.1 |
| tqdm | 4.70.1 |
| CUDA available | `True` |
| PyTorch CUDA runtime | 13.0 |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| GPU VRAM | 4,294,443,008 bytes (reported as 4095 MiB) |
| Current CUDA free memory | 3,459,986,228 bytes (about 3300 MiB) |
| Current PyTorch allocated/reserved | 0 / 0 bytes after smoke process exit |
| Package compatibility | `pip check`: `No broken requirements found.` |

Observed environment log:

```text
python 3.12.0
executable .venv312\Scripts\python.exe
torch 2.12.1+cu130
transformers 5.24.1
cuda_available True
cuda_runtime 13.0
gpu NVIDIA GeForce RTX 3050 Laptop GPU
No broken requirements found.
```

`nvidia-smi` reported driver 591.86, CUDA driver support 13.1, and 0 MiB currently in use when checked.

## 2. Clean dataset verification

Ran `experiments/clean_v1/validate_clean_dataset.py` under `.venv312`; it exited successfully and wrote `dataset_validation_report.md`.

| Split | Samples | Negative | Positive | Negative fraction |
|---|---:|---:|---:|---:|
| Train | 19,824 | 9,869 | 9,955 | 49.7831% |
| Validation | 4,957 | 2,468 | 2,489 | 49.7882% |
| Official test | 24,801 | 12,361 | 12,440 | 49.8407% |

Validation confirmed the following:

- Train/validation normalized-hash overlap: **0**.
- Train/test normalized-hash overlap: **0**.
- Validation/test normalized-hash overlap: **0**.
- No `train/unsup` path is present in any manifest.
- Source paths exist, decode as UTF-8, and yield non-empty review text.
- Manifest rows reproduce in identical order from the saved seed and splitting procedure.
- Both clean training scripts call the same `load_clean_train_validation()` API; both evaluators call `load_clean_test()`. The manifest fingerprints are carried in each model's training configuration.

Manifest SHA-256 values were unchanged during validation and smoke checks:

| Manifest | SHA-256 |
|---|---|
| `train.csv` | `b6993f416b753e62ebf1c2dc304bfdea2a25366a0ad8e9424e9e74502c033140` |
| `validation.csv` | `bf4a228fc67000384d9864e3b7c4d81f79facdc0baf0f71fb16c3f45bf10ceba` |
| `test.csv` | `be25f9d0c1ecd276b7974f5f12e475405dd81173f31c8e347eb30db568933d6d` |

## 3. Model A smoke test

The clean_v1 wrapper reuses the inspected P1 `ScratchTransformer` and `ScratchTokenizer` source without editing `scratch/`. The clean training script fits the vocabulary from `train_texts` returned by the clean train manifest only; it does not load the test manifest.

Smoke test used 128 training-manifest examples to fit a temporary tokenizer, then checked a batch of two at sequence length 256 on CUDA:

```text
Vocabulary size: 5959
device: cuda
logits shape: [2, 2]
cross-entropy loss: 0.7216928005218506 (smoke-only)
finite loss and gradients: passed
AdamW optimizer step: passed
custom MultiHeadAttention / ScaledDotProductAttention: passed
built-in TransformerEncoder/MultiheadAttention absent: passed
right-padding invariance check: passed
temporary checkpoint save/load and matching logits: passed
```

The checkpoint round-trip used a temporary file outside the historical checkpoint directory. No historical Model A artifact was loaded for this smoke test or overwritten.

## 4. Model B smoke test and pretrained-weight confirmation

The local checkpoint `models/distilbert-base-uncased` loaded with `local_files_only=True`. `AutoTokenizer` loaded as a Bert tokenizer with vocabulary size 30,522. The sequence classifier has two outputs and the expected mappings:

```text
id2label: {0: NEGATIVE, 1: POSITIVE}
label2id: {NEGATIVE: 0, POSITIVE: 1}
```

The Transformers load report is expected for a masked-language-model checkpoint being converted to sequence classification:

```text
MISSING: classifier.bias, classifier.weight,
         pre_classifier.bias, pre_classifier.weight
UNEXPECTED: vocab_transform.bias, vocab_transform.weight,
            vocab_layer_norm.bias, vocab_layer_norm.weight,
            vocab_projector.bias
```

To verify the encoder rather than relying on the warning text, the base `AutoModel` and the sequence-classification model were loaded from the same local checkpoint. Their **100 encoder state tensors matched exactly** (`torch.equal` for every key). There were no missing base-encoder keys, size mismatches, or load errors. Thus the unexpected keys belong to the source MLM head, while the absent task-specific classification layers were newly initialized as intended.

CUDA smoke results:

```text
logits shape: [1, 2]
cross-entropy loss: 0.6784078478813171 (smoke-only)
finite loss and backward gradients: passed
CUDA execution: passed
save_pretrained model/tokenizer round-trip: passed
reloaded config labels: two classes with NEGATIVE=0, POSITIVE=1
```

The round-trip used a temporary directory. No clean_v1 training checkpoint or test metrics were created.

## 5. GPU memory assessment

The controlled trial used the local model in FP32, input length 256, labels, AdamW, one forward/backward pass, and one optimizer update. Batch 8 was attempted first and succeeded; no fallback was needed.

| Physical micro-batch | Accumulation | Effective batch | Result | Peak allocated | Peak reserved |
|---:|---:|---:|---|---:|---:|
| 8 | 1 | 8 | **PASSED** | 1,365,398,528 bytes (1302.1 MiB) | 1,442,840,576 bytes (1376.0 MiB) |

Mixed precision, micro-batch 4 with accumulation 2, and gradient checkpointing were **NOT TESTED** because the requested batch-8 step passed with headroom. They remain disabled in the default training configuration.

## 6. Clean training script review

Both training scripts compiled and imported successfully. The clean evaluators also imported successfully. Static review confirmed:

| Requirement | Model A | Model B |
|---|---|---|
| Clean training/validation source | `load_clean_train_validation()` | `load_clean_train_validation()` |
| Test data during training | Not loaded | Not loaded |
| Seed | 42 | 42 |
| Optimizer | AdamW | AdamW |
| Learning rate | 0.0003 | 2e-5 |
| Weight decay | 0.01 (AdamW default made explicit) | 0.01 |
| Batch | 16 | Effective 8; physical 8, accumulation 1 |
| Epochs | 6 | 3 |
| Best checkpoint criterion | Validation Macro-F1 | Validation Macro-F1 |
| Training config includes clean manifest hashes | Yes | Yes |
| Output location | `experiments/clean_v1/scratch/checkpoints/` | `experiments/clean_v1/pretrained/checkpoints/` |
| Historical paths protected | Yes; clean-only path and overwrite guard | Yes; clean-only path and overwrite guard |

Both training scripts load train and validation only. Their separate evaluators load the clean official-test manifest; they save metrics only after evaluation and refuse to overwrite an existing clean_v1 metrics file. No test evaluation or test metric generation occurred during this readiness check.

## 7. Preservation check

SHA-256 checks of all historical Model A artifacts still match the Stage 1 audit:

```text
scratch/checkpoints/best_model.pth        UNCHANGED
scratch/checkpoints/tokenizer.json        UNCHANGED
scratch/checkpoints/training_history.json UNCHANGED
scratch/checkpoints/test_metrics.json     UNCHANGED
```

No full-training checkpoint directories, histories, or test metrics exist under the clean_v1 model paths. The only clean_v1 model files currently present are the source wrappers/scripts and temporary smoke outputs were cleaned up.

## Stage status and remaining work

- Stages 1–5 (audit, dataset creation/validation, environment, and both model smoke tests): **COMPLETED**.
- Stage 6 (full clean_v1 Model A training and final test evaluation): **NOT STARTED**.
- Stage 7 (full clean_v1 DistilBERT training and final test evaluation): **NOT STARTED**.
- Stage 8 (comparison reports and plots): **NOT STARTED**.

**Readiness checks passed. Full training is deliberately not started; it awaits the user's review of this report.**

---

# Final local readiness check (2026-11-26)

**Scope:** readiness verification only. No full experiment training, official-test evaluation, metric generation, or manifest modification was performed. The two model smoke checks did include one transient in-memory AdamW update each to exercise the forward/backward path; both test models and checkpoints stayed in temporary test scope and were discarded after round-trip verification. The official test split was read only by the dataset validator to verify its paths, text hashes, counts, and leakage status. This dated section records the current computer and supersedes older machine-specific environment values above; the earlier report content is retained.

| Check | Status | Exact command/script | Actual result and evidence | Remaining blocker |
|---|---|---|---|---|
| Repository and branch | PASS | `git branch --show-current; git status --short --branch; git log -1 --oneline` | Branch `clean-v1-training`; HEAD `a0ba10c`. Existing worktree changes are `dataset_validation_report.md` and `validate_clean_dataset.py`; both preserved. | None |
| Python/CUDA/GPU | PASS | `.\.venv\Scripts\python.exe --version`; `.\.venv\Scripts\python.exe -c "import torch,sys; ..."`; `nvidia-smi` | Python 3.12.10; PyTorch 2.14.1+cu126; CUDA available, runtime 12.6; NVIDIA GeForce RTX 4060 Laptop GPU; 8,188 MiB reported; idle at 0 MiB before smoke checks. | None |
| Clean dataset validator | PASS | `.\.venv\Scripts\python.exe -` with `validate_clean_dataset.REPORT_PATH` redirected to `%TEMP%\clean_v1_readiness_validation_tmp.md`, then `main()`; temporary report removed after success | Validator passed source readability/UTF-8/hash checks, labels and source membership, deterministic ordering/reconstruction, no `train/unsup`, and all expected counts. Temporary validation report said **PASSED**. | None |
| Manifest fingerprints and counts | PASS | Same validator invocation; canonical-LF SHA-256 values compared to `dataset_summary.json` | Train: 19,824, SHA-256 `b6993f416b753e62ebf1c2dc304bfdea2a25366a0ad8e9424e9e74502c033140`; validation: 4,957, `bf4a228fc67000384d9864e3b7c4d81f79facdc0baf0f71fb16c3f45bf10ceba`; official test: 24,801, `be25f9d0c1ecd276b7974f5f12e475405dd81173f31c8e347eb30db568933d6d`. All match the approved summary. The validator’s LF normalization handles this checkout’s `core.autocrlf=true`; manifest files were not changed. | None |
| Normalized-text leakage | PASS | Same validator invocation | Train/validation overlap 0; train/test overlap 0; validation/test overlap 0. 123 train/test-shared source hashes were excluded before train/validation construction. | None |
| Scratch Transformer smoke | PASS | Inline smoke script piped to `.\.venv\Scripts\python.exe -u -`; used `load_clean_train_validation()`, training-only 128-review tokenizer fit, approved architecture dimensions, CUDA, cross-entropy/backward/AdamW update, and temporary `torch.save`/load round-trip | CUDA logits `[2, 2]`; finite loss `0.7353824`; backward and optimizer step passed; checkpoint logits matched after reload. Temporary checkpoint directory removed. Peak CUDA allocated 49,204,224 bytes (~46.9 MiB), reserved 56,623,104 bytes (54 MiB). | None |
| DistilBERT smoke | PASS | Same inline smoke script; local `load_tokenizer`/`load_model` with `local_files_only=True`, micro-batch 8, max length 256, cross-entropy/backward/AdamW update, `save_pretrained()` and reload round-trip | Local tokenizer/model loaded; output `[8, 2]`; finite loss `0.7288257`; round-trip logits matched. Total parameters 66,955,010 including the task head; encoder parameters 66,362,880. Expected newly initialized classifier and unused MLM-head load notices appeared; no CUDA error/OOM. Temporary checkpoint/tokenizer directory removed. Peak allocated 1,456,469,504 bytes (~1,389 MiB), reserved 1,566,571,544 bytes (~1,494 MiB). | None |
| Training pipeline data isolation | PASS | Inspected `experiments/clean_v1/scratch/train.py`, `experiments/clean_v1/pretrained/train.py`, and `clean_data_loader.py` | Both training scripts call the same `load_clean_train_validation()` loader for train and validation and do not load test. Both share the same manifest directory and dataset summary fingerprints. Test loading is isolated to the separate evaluator scripts. | None |
| Dependencies | PASS | `.\.venv\Scripts\python.exe -c "import importlib.metadata ..."`; `.\.venv\Scripts\python.exe -m pip check` | Installed: torch 2.14.1+cu126, NumPy 2.5.5, scikit-learn 1.9.1, tqdm 4.70.1, Transformers 5.24.1, tokenizers 0.23.3, safetensors 0.8.0, huggingface_hub 2.4.0. `pip check`: no broken requirements. It emitted a warning for an orphaned `~ip-25.0.1.dist-info` remnant; no required package is missing. | None that prevents training |
| Output safety and historical artifacts | PASS | Inspected clean checkpoint paths and training output guards; SHA-256 checked the four historical `scratch/checkpoints/*` files against `audit_report.md` | Clean Model A and Model B checkpoint directories do not exist yet. Both train scripts refuse to write into a non-empty clean output directory. Historical checkpoint, tokenizer, history, and metrics hashes all match the audit. | None |
| Full training and official-test evaluation | NOT RUN (by request) | No training or evaluator command executed | Readiness only; test data was not used for model predictions or checkpoint selection. | Training remains to be started separately after this readiness review. |

## Final verdict

**READY FOR TRAINING**

This verdict means the local data, environment, model smoke checks, memory check, dependencies, and output protections passed. No training was started as part of this check.
