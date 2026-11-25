# Transformer: From Scratch vs. Pretrained Model

A Natural Language Processing course project comparing a custom Transformer encoder implemented from scratch with a pretrained Transformer fine-tuned for binary sentiment analysis.

## 1. Project Overview

This project compares two Transformer approaches for classifying IMDb movie reviews as positive or negative.

- **Model A:** A custom Transformer encoder implemented and trained from scratch using PyTorch.
- **Model B:** A pretrained DistilBERT model fine-tuned using Hugging Face Transformers.

Both models use the same dataset, data split, and held-out test examples to ensure a fair comparison.

The project also includes a Streamlit application that allows users to enter a movie review and compare both models' predictions.

## 2. Project Objectives

- Understand the internal architecture of Transformer models.
- Implement attention and multi-head attention without built-in Transformer blocks.
- Train a Transformer using randomly initialized weights.
- Fine-tune a pretrained Transformer.
- Compare both approaches using standard classification metrics.
- Demonstrate both models through an interactive application.

## 3. Dataset

**Dataset:** IMDb Large Movie Review Dataset (aclImdb)

**Source:** Stanford AI Lab

https://ai.stanford.edu/~amaas/data/sentiment/

The dataset contains 50,000 labeled movie reviews:

- 25,000 training reviews.
- 25,000 official test reviews.

The dataset is balanced between positive and negative reviews.

The additional unlabeled reviews in `train/unsup` are excluded.

## 4. Data Splitting

The original 25,000 labeled training reviews are divided into training and validation sets.

| Split | Samples | Negative | Positive |
|---|---:|---:|---:|
| Training | 20,000 | 10,000 | 10,000 |
| Validation | 5,000 | 2,500 | 2,500 |
| Testing | 25,000 | 12,500 | 12,500 |

The split uses stratified sampling with:

- Random seed: 42
- Validation proportion: 20%
- Shuffling: Enabled
- Stratification: Enabled

Both models use the same `data_loader.py` implementation to guarantee identical splits.

## 5. Models

### Model A — Transformer from Scratch

A custom encoder-only Transformer built using basic PyTorch layers.

Implemented components:

- Token embeddings
- Positional embeddings
- Scaled dot-product attention
- Custom multi-head attention
- Feed-forward layers
- Residual connections
- Layer normalization
- Dropout
- Padding masks
- Masked mean pooling
- Classification head

No pretrained weights or built-in Transformer blocks are used.

**Configuration:**

| Parameter | Value |
|---|---:|
| Vocabulary size | 30,000 |
| Maximum sequence length | 256 |
| Embedding dimension | 128 |
| Attention heads | 4 |
| Hidden dimension | 256 |
| Encoder layers | 3 |
| Dropout | 0.1 |
| Batch size | 16 |
| Epochs | 6 |
| Learning rate | 0.0003 |
| Optimizer | AdamW |
| Random seed | 42 |

**Current measured test accuracy:** 84.16%

### Model B — Pretrained DistilBERT

A pretrained DistilBERT model fine-tuned for binary sentiment classification.

**Checkpoint:** `distilbert-base-uncased`

Fine-tuning configuration and final evaluation results will be added after training is completed.

## 6. Evaluation Metrics

Both models are evaluated using:

- Accuracy
- Precision
- Recall
- Macro-F1
- Confusion matrix
- Parameter count
- Average inference latency
- Training or fine-tuning time

All final quality metrics are calculated on the same 25,000 held-out test reviews.

## 7. Current Model A Results

| Metric | Result |
|---|---:|
| Accuracy | 84.16% |
| Precision | 84.18% |
| Recall | 84.16% |
| Macro-F1 | 84.16% |
| Parameters | 4,270,722 |
| Training time | 2.14 minutes |
| Average inference latency | 0.2882 ms/review |

The reported inference latency is from the current batched evaluation procedure. See the final experiment report for the exact measurement protocol.

## 8. Project Structure

```text
NLP_Transformer_Comparison/
│
├── data/
│   └── raw/
│       └── aclImdb/
│
├── scratch/
│   ├── model.py
│   ├── tokenizer.py
│   ├── train.py
│   ├── evaluate.py
│   └── checkpoints/
│
├── pretrained/
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── checkpoints/
│
├── app.py
├── data_loader.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 9. Installation

### Requirements

- Python 3.12
- NVIDIA GPU with CUDA support recommended
- PyTorch
- Hugging Face Transformers
- Streamlit

### Clone the repository

```bash
git clone https://github.com/daniaa-a/NLP_Transformer_Comparison.git
cd NLP_Transformer_Comparison
```

### Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Install PyTorch

Install a PyTorch build compatible with your system and CUDA version.

For the development environment used for this project, the installation command was:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

If this CUDA build is unavailable for a different machine, use the official PyTorch installation selector:

https://pytorch.org/get-started/locally/

### Install the remaining dependencies

```powershell
pip install -r requirements.txt
```

## 10. Dataset Setup

Download the IMDb Large Movie Review Dataset from:

https://ai.stanford.edu/~amaas/data/sentiment/

Extract the dataset so that the directory structure is:

```text
data/
└── raw/
    └── aclImdb/
        ├── train/
        │   ├── pos/
        │   ├── neg/
        │   └── unsup/
        └── test/
            ├── pos/
            └── neg/
```

The project loader reads labeled positive and negative reviews only.

## 11. Training

### Train Model A

```powershell
python scratch/train.py
```

### Evaluate Model A

```powershell
python scratch/evaluate.py
```

### Train Model B

```powershell
python pretrained/train.py
```

### Evaluate Model B

```powershell
python pretrained/evaluate.py
```

The pretrained checkpoint is downloaded through Hugging Face when first required, and the fine-tuned checkpoint is saved locally.

Training should be performed before launching the application if the required model checkpoints are not already available.

## 12. Run the Streamlit Application

From the project root:

```powershell
streamlit run app.py
```

The application allows users to:

- Enter movie reviews.
- Obtain predictions from both models.
- Compare predicted sentiments and confidence scores.
- View inference latency.
- Review test metrics and confusion matrices.
- Explore example reviews.

## 13. Hardware and Experiment Environment

| Component | Configuration |
|---|---|
| Operating system | Windows |
| Python | 3.12 |
| Deep learning framework | PyTorch |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| CUDA | 13.2 |

## 14. Model Artifacts

Model A artifacts:

```text
scratch/checkpoints/
├── best_model.pth
├── tokenizer.json
├── training_history.json
└── test_metrics.json
```

Model B artifacts:

```text
pretrained/checkpoints/
├── best_model/
├── training_history.json
└── test_metrics.json
```

Large model files may be distributed through a documented download procedure rather than committed directly to GitHub.

## 15. Results and Discussion

The final comparison table will be completed after Model B is trained and evaluated.

The discussion will cover:

- Accuracy and Macro-F1 differences.
- Confusion matrix comparison.
- Model parameter counts.
- Training and fine-tuning time.
- Inference latency.
- Benefits of pretrained language representations.
- Limitations of training from scratch on a limited dataset.

Only experimentally measured results will be reported.

## 16. References

1. Maas, A. L., et al. (2011). Learning Word Vectors for Sentiment Analysis. ACL.
2. IMDb Large Movie Review Dataset: https://ai.stanford.edu/~amaas/data/sentiment/
3. Hugging Face DistilBERT: https://huggingface.co/distilbert-base-uncased
4. PyTorch Documentation: https://pytorch.org/docs/
5. Hugging Face Transformers Documentation: https://huggingface.co/docs/transformers/

## 17. License and Dataset Terms

The dataset is distributed by its original authors for research use. Consult the original dataset page and accompanying terms before redistribution. Do not include the raw dataset in the GitHub repository.