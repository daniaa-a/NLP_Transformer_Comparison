# clean_v1 Dataset Validation

**Status: PASSED**

All checks below were run against the manifests and source review files.

| Split | Samples | Negative | Positive | Negative fraction | Manifest SHA-256 |
|---|---:|---:|---:|---:|---|
| train | 19,824 | 9,869 | 9,955 | 49.7831% | `b6993f416b753e62ebf1c2dc304bfdea2a25366a0ad8e9424e9e74502c033140` |
| validation | 4,957 | 2,468 | 2,489 | 49.7882% | `bf4a228fc67000384d9864e3b7c4d81f79facdc0baf0f71fb16c3f45bf10ceba` |
| test | 24,801 | 12,361 | 12,440 | 49.8407% | `be25f9d0c1ecd276b7974f5f12e475405dd81173f31c8e347eb30db568933d6d` |

Unique content groups: 49,582; negative=24,698; positive=24,884.
Source overlap removed before splitting: 123 hashes.
Pairwise manifest overlap counts: `{"train_test": 0, "train_validation": 0, "validation_test": 0}`.
Train/validation size ratio: 3.999193:1.

## Checks

- PASS: train manifest paths resolve, texts are readable/non-empty, hashes and labels match.
- PASS: train count/class distribution verified: 19824 / {1: 9955, 0: 9869}.
- PASS: validation manifest paths resolve, texts are readable/non-empty, hashes and labels match.
- PASS: validation count/class distribution verified: 4957 / {0: 2468, 1: 2489}.
- PASS: test manifest paths resolve, texts are readable/non-empty, hashes and labels match.
- PASS: test count/class distribution verified: 24801 / {0: 12361, 1: 12440}.
- PASS: all pairwise normalized-hash overlaps are zero.
- PASS: test is sourced only from official test; shared train/test hashes are excluded from train/validation.
- PASS: no train/unsup source path is present.
- PASS: repeated manifest loads produce identical ordering and identities.
- PASS: reconstructed manifests match the saved seed and deterministic stratified split.

Normalization: Unicode NFKC, casefold, and collapse whitespace. Original review text is read unchanged for model input.
The official test manifest contains only unique source-test paths; all hashes shared with original train were excluded from train and validation.
