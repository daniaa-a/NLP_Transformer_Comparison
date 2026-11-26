
import json
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F


# =========================================================
# 1. STREAMLIT CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Transformer Comparison | Sentiment Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. PROJECT PATHS AND CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent
CLEAN_ROOT = PROJECT_ROOT / "experiments" / "clean_v1"

SCRATCH_CHECKPOINT = (
    CLEAN_ROOT / "scratch" / "checkpoints" / "best_model.pth"
)

SCRATCH_TOKENIZER_PATH = (
    CLEAN_ROOT / "scratch" / "checkpoints" / "tokenizer.json"
)

DISTILBERT_CHECKPOINT = (
    CLEAN_ROOT / "pretrained" / "checkpoints" / "best_model"
)

SCRATCH_METRICS_PATH = (
    CLEAN_ROOT / "scratch" / "checkpoints" / "test_metrics.json"
)

DISTILBERT_METRICS_PATH = (
    CLEAN_ROOT / "pretrained" / "checkpoints" / "test_metrics.json"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# 3. CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    .hero {
        padding: 2.2rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #172554, #312e81);
        margin-bottom: 2rem;
    }

    .hero h1 {
        color: white;
        margin-bottom: 0.5rem;
        font-size: 2.5rem;
    }

    .hero p {
        color: #dbeafe;
        font-size: 1.05rem;
    }

    .model-card {
        padding: 1.3rem;
        border: 1px solid #334155;
        border-radius: 14px;
        background-color: #161b27;
        min-height: 125px;
        margin-bottom: 0.5rem;
    }

    .model-card h3 {
        color: white;
        margin-bottom: 0.5rem;
    }

    .model-card p {
        color: #94a3b8;
        font-size: 0.9rem;
    }

    .section-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
    }

    .positive {
        color: #22c55e;
        font-weight: 700;
        font-size: 1.4rem;
    }

    .negative {
        color: #ef4444;
        font-weight: 700;
        font-size: 1.4rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 4. LOAD SCRATCH TRANSFORMER
# =========================================================

@st.cache_resource(show_spinner=False)
def load_scratch_model():

    from experiments.clean_v1.scratch.model import (
        ScratchTransformer,
    )

    from experiments.clean_v1.scratch.tokenizer import (
        ScratchTokenizer,
    )

    if not SCRATCH_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Scratch checkpoint not found: {SCRATCH_CHECKPOINT}"
        )

    if not SCRATCH_TOKENIZER_PATH.is_file():
        raise FileNotFoundError(
            f"Scratch tokenizer not found: {SCRATCH_TOKENIZER_PATH}"
        )

    checkpoint = torch.load(
        SCRATCH_CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    config = checkpoint["config"]

    word_to_id = json.loads(
        SCRATCH_TOKENIZER_PATH.read_text(encoding="utf-8")
    )

    tokenizer = ScratchTokenizer(
        vocab_size=len(word_to_id),
        max_length=config["max_length"],
    )

    tokenizer.word_to_id = {
        word: int(index)
        for word, index in word_to_id.items()
    }

    tokenizer.id_to_word = {
        index: word
        for word, index in tokenizer.word_to_id.items()
    }

    model = ScratchTransformer(
        vocab_size=checkpoint["vocab_size"],
        embedding_dim=config["embedding_dim"],
        num_heads=config["num_heads"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        max_length=config["max_length"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
        padding_idx=config.get("padding_idx", 0),
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    return model, tokenizer


# =========================================================
# 5. LOAD FINE-TUNED DISTILBERT
# =========================================================

@st.cache_resource(show_spinner=False)
def load_distilbert_model():

    from experiments.clean_v1.pretrained.model import (
        load_finetuned_model,
        load_tokenizer,
    )

    if not DISTILBERT_CHECKPOINT.is_dir():
        raise FileNotFoundError(
            "Fine-tuned DistilBERT checkpoint not found: "
            f"{DISTILBERT_CHECKPOINT}"
        )

    tokenizer = load_tokenizer(
        DISTILBERT_CHECKPOINT,
        local_files_only=True,
    )

    model = load_finetuned_model(
        DISTILBERT_CHECKPOINT,
        device=DEVICE,
        local_files_only=True,
    )

    model.eval()

    return model, tokenizer


# =========================================================
# 6. LOAD OFFICIAL EVALUATION METRICS
# =========================================================

@st.cache_data
def load_evaluation_metrics():

    if not SCRATCH_METRICS_PATH.is_file():
        raise FileNotFoundError(
            f"Scratch metrics not found: {SCRATCH_METRICS_PATH}"
        )

    if not DISTILBERT_METRICS_PATH.is_file():
        raise FileNotFoundError(
            f"DistilBERT metrics not found: {DISTILBERT_METRICS_PATH}"
        )

    with open(
        SCRATCH_METRICS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        scratch = json.load(file)

    with open(
        DISTILBERT_METRICS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        distilbert = json.load(file)

    scratch_hash = scratch["manifest_sha256"]["test"]
    distilbert_hash = distilbert["manifest_sha256"]["test"]

    if scratch_hash != distilbert_hash:
        raise ValueError(
            "The two models were evaluated on different test splits."
        )

    if scratch["test_samples"] != distilbert["test_samples"]:
        raise ValueError(
            "Test sample counts do not match."
        )

    return scratch, distilbert


# =========================================================
# 7. INFERENCE UTILITIES
# =========================================================

def synchronize_device():

    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)


def predict_scratch(text, model, tokenizer):

    encoded = tokenizer.encode(text)

    input_ids = torch.tensor(
        [encoded],
        dtype=torch.long,
        device=DEVICE,
    )

    with torch.inference_mode():

        synchronize_device()

        start_time = time.perf_counter()

        logits = model(input_ids)

        synchronize_device()

        latency = (
            time.perf_counter() - start_time
        ) * 1000

        probabilities = F.softmax(logits, dim=1)[0]

    predicted_class = int(
        torch.argmax(probabilities).item()
    )

    labels = {
        0: "Negative",
        1: "Positive",
    }

    return {
        "sentiment": labels[predicted_class],
        "confidence": float(
            probabilities[predicted_class].item()
        ),
        "probabilities": probabilities.cpu().tolist(),
        "latency": latency,
    }


def predict_distilbert(text, model, tokenizer):

    encoded = tokenizer(
        text,
        max_length=256,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )

    encoded = {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }

    with torch.inference_mode():

        synchronize_device()

        start_time = time.perf_counter()

        outputs = model(**encoded)

        synchronize_device()

        latency = (
            time.perf_counter() - start_time
        ) * 1000

        probabilities = F.softmax(
            outputs.logits,
            dim=1,
        )[0]

    predicted_class = int(
        torch.argmax(probabilities).item()
    )

    labels = {
        0: "Negative",
        1: "Positive",
    }

    return {
        "sentiment": labels[predicted_class],
        "confidence": float(
            probabilities[predicted_class].item()
        ),
        "probabilities": probabilities.cpu().tolist(),
        "latency": latency,
    }



# =========================================================
# 8. HEADER
# =========================================================

st.markdown(
    """
    <style>
    .hero-container {
        background: linear-gradient(120deg, #172554, #312e81);
        padding: 35px;
        border-radius: 18px;
        margin-bottom: 30px;
    }

    .hero-title {
        color: white;
        font-size: 38px;
        font-weight: 800;
        margin: 0;
    }

    .hero-description {
        color: #dbeafe;
        font-size: 17px;
        margin-top: 12px;
        line-height: 1.7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">
            🧠 Transformer Comparison
        </div>
        <div class="hero-description">
            Scratch Transformer vs. Pretrained DistilBERT
            <br>
            An Interactive IMDb Sentiment Analysis Platform
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 9. SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:

    st.title("🧠 Project Navigation")

    page = st.radio(
        "Select a Section",
        [
            "Live Prediction",
            "Model Comparison",
            "About the Project",
        ],
    )

    st.divider()

    st.caption("NLP & Deep Learning Project")

    st.caption(f"Running device: {DEVICE}")


# =========================================================
# 10. LIVE SENTIMENT PREDICTION
# =========================================================

if page == "Live Prediction":

    st.title("Live Sentiment Prediction")

    st.write(
        "Enter a movie review and compare the predictions "
        "of both Transformer models."
    )

    examples = {
        "Choose an example": "",

        "Positive Review": (
            "I absolutely loved this movie! "
            "The acting was amazing and the story was wonderful."
        ),

        "Negative Review": (
            "This was one of the worst movies I have ever watched. "
            "The story was boring and the acting was terrible."
        ),

        "Mixed Review": (
            "The movie had some interesting scenes, "
            "but the story was slow and disappointing."
        ),
    }

    selected_example = st.selectbox(
        "Try an example review",
        list(examples.keys()),
    )

    review = st.text_area(
        "Movie Review",
        value=examples[selected_example],
        height=160,
        placeholder="Write your movie review here...",
    )

    predict_button = st.button(
        "Compare Models",
        type="primary",
        use_container_width=True,
    )

    if predict_button:

        if not review.strip():

            st.warning(
                "Please enter a movie review first."
            )

        else:

            try:

                with st.spinner(
                    "Loading Scratch Transformer..."
                ):

                    scratch_model, scratch_tokenizer = (
                        load_scratch_model()
                    )

                with st.spinner(
                    "Loading fine-tuned DistilBERT..."
                ):

                    distilbert_model, distilbert_tokenizer = (
                        load_distilbert_model()
                    )

                with st.spinner(
                    "Analyzing your review..."
                ):

                    scratch_result = predict_scratch(
                        review,
                        scratch_model,
                        scratch_tokenizer,
                    )

                    distilbert_result = predict_distilbert(
                        review,
                        distilbert_model,
                        distilbert_tokenizer,
                    )

                st.success(
                    "Prediction completed successfully!"
                )

                st.subheader("Model Predictions")

                col1, col2 = st.columns(2)

                results = [
                    (
                        col1,
                        "Scratch Transformer",
                        "Custom Transformer trained from scratch",
                        scratch_result,
                    ),
                    (
                        col2,
                        "DistilBERT",
                        "Pretrained Transformer fine-tuned on IMDb",
                        distilbert_result,
                    ),
                ]

                for (
                    column,
                    model_name,
                    description,
                    result,
                ) in results:

                    with column:

                        st.markdown(
                            f"""
                            <div class="model-card">
                                <h3>{model_name}</h3>
                                <p>{description}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        sentiment = result["sentiment"]

                        if sentiment == "Positive":

                            st.markdown(
                                '<p class="positive">'
                                'Positive 😊</p>',
                                unsafe_allow_html=True,
                            )

                        else:

                            st.markdown(
                                '<p class="negative">'
                                'Negative ☹️</p>',
                                unsafe_allow_html=True,
                            )

                        st.metric(
                            "Confidence",
                            f'{result["confidence"] * 100:.2f}%',
                        )

                        st.progress(
                            result["confidence"]
                        )

                        st.metric(
                            "Model Forward Latency",
                            f'{result["latency"]:.3f} ms',
                        )

                        st.caption(
                            "Negative: "
                            f'{result["probabilities"][0] * 100:.2f}%'
                        )

                        st.caption(
                            "Positive: "
                            f'{result["probabilities"][1] * 100:.2f}%'
                        )

                st.divider()

                st.subheader("Prediction Agreement")

                if (
                    scratch_result["sentiment"]
                    == distilbert_result["sentiment"]
                ):

                    st.success(
                        "Both models predicted the same sentiment."
                    )

                else:

                    st.warning(
                        "The two models predicted different sentiments."
                    )

                st.caption(
                    "Latency measures model forward-pass time only. "
                    "Tokenization and application overhead are excluded."
                )

            except Exception as error:

                st.error(
                    "An error occurred while loading or running "
                    f"the models: {error}"
                )

                st.exception(error)


# =========================================================
# 11. MODEL COMPARISON DASHBOARD
# =========================================================

elif page == "Model Comparison":

    st.title("📊 Model Performance Dashboard")

    st.write(
        "Official evaluation results from the clean_v1 "
        "IMDb held-out test dataset."
    )

    try:

        scratch, distilbert = load_evaluation_metrics()

        st.success(
            "Both models were evaluated on the same "
            f"{scratch['test_samples']:,} test reviews."
        )

        # -------------------------------------------------
        # PERFORMANCE OVERVIEW
        # -------------------------------------------------

        st.header("Performance Overview")

        metrics = [
            ("Accuracy", "accuracy"),
            ("Macro Precision", "precision_macro"),
            ("Macro Recall", "recall_macro"),
            ("Macro-F1", "macro_f1"),
        ]

        for row_start in range(0, len(metrics), 2):

            col1, col2 = st.columns(2)

            for column, (label, key) in zip(
                [col1, col2],
                metrics[row_start:row_start + 2],
            ):

                scratch_value = scratch[key] * 100
                distilbert_value = distilbert[key] * 100

                with column:

                    st.subheader(label)

                    metric_col1, metric_col2 = st.columns(2)

                    metric_col1.metric(
                        "Scratch",
                        f"{scratch_value:.2f}%",
                    )

                    metric_col2.metric(
                        "DistilBERT",
                        f"{distilbert_value:.2f}%",
                        delta=(
                            f"{distilbert_value - scratch_value:+.2f} pp"
                        ),
                    )

        st.divider()

        # -------------------------------------------------
        # PERFORMANCE COMPARISON CHART
        # -------------------------------------------------

        st.header("Performance Comparison")

        comparison_data = pd.DataFrame(
            {
                "Scratch Transformer": {
                    "Accuracy": scratch["accuracy"] * 100,
                    "Precision": scratch["precision_macro"] * 100,
                    "Recall": scratch["recall_macro"] * 100,
                    "Macro-F1": scratch["macro_f1"] * 100,
                },

                "DistilBERT": {
                    "Accuracy": distilbert["accuracy"] * 100,
                    "Precision": distilbert["precision_macro"] * 100,
                    "Recall": distilbert["recall_macro"] * 100,
                    "Macro-F1": distilbert["macro_f1"] * 100,
                },
            }
        )

        st.bar_chart(
            comparison_data,
            y_label="Score (%)",
            height=400,
        )

        st.dataframe(
            comparison_data.style.format("{:.2f}%"),
            use_container_width=True,
        )

        st.divider()

        # -------------------------------------------------
        # CONFUSION MATRICES
        # -------------------------------------------------

        st.header("Confusion Matrices")

        st.caption(
            "Rows represent actual labels; "
            "columns represent predicted labels."
        )

        cm_col1, cm_col2 = st.columns(2)

        with cm_col1:

            st.subheader("Scratch Transformer")

            scratch_cm = pd.DataFrame(
                scratch["confusion_matrix"],
                index=[
                    "Actual Negative",
                    "Actual Positive",
                ],
                columns=[
                    "Predicted Negative",
                    "Predicted Positive",
                ],
            )

            st.dataframe(
                scratch_cm,
                use_container_width=True,
            )

            st.caption(
                "Correct predictions: "
                f"{scratch_cm.iloc[0, 0] + scratch_cm.iloc[1, 1]:,}"
            )

        with cm_col2:

            st.subheader("DistilBERT")

            distilbert_cm = pd.DataFrame(
                distilbert["confusion_matrix"],
                index=[
                    "Actual Negative",
                    "Actual Positive",
                ],
                columns=[
                    "Predicted Negative",
                    "Predicted Positive",
                ],
            )

            st.dataframe(
                distilbert_cm,
                use_container_width=True,
            )

            st.caption(
                "Correct predictions: "
                f"{distilbert_cm.iloc[0, 0] + distilbert_cm.iloc[1, 1]:,}"
            )

        st.divider()

        # -------------------------------------------------
        # MODEL COMPLEXITY
        # -------------------------------------------------

        st.header("Model Complexity")

        complexity_col1, complexity_col2 = st.columns(2)

        with complexity_col1:

            st.subheader("Scratch Transformer")

            st.metric(
                "Parameters",
                f"{scratch['parameters']:,}",
            )

            st.metric(
                "Parameters (Millions)",
                f"{scratch['parameters'] / 1_000_000:.2f} M",
            )

        with complexity_col2:

            st.subheader("DistilBERT")

            st.metric(
                "Parameters",
                f"{distilbert['parameters']:,}",
            )

            st.metric(
                "Parameters (Millions)",
                f"{distilbert['parameters'] / 1_000_000:.2f} M",
            )

        parameter_data = pd.DataFrame(
            {
                "Parameters (Millions)": [
                    scratch["parameters"] / 1_000_000,
                    distilbert["parameters"] / 1_000_000,
                ]
            },
            index=[
                "Scratch Transformer",
                "DistilBERT",
            ],
        )

        st.bar_chart(
            parameter_data,
            y_label="Parameters (Millions)",
        )

        st.divider()

        # -------------------------------------------------
        # INFERENCE LATENCY
        # -------------------------------------------------

        st.header("Inference Latency")

        st.caption(
            "Model-forward latency per review. "
            "Tokenization and application overhead are excluded."
        )

        latency_data = pd.DataFrame(
            {
                "Batch Size 8 (ms)": [
                    scratch["batch_averaged_latency_ms_per_review"],
                    distilbert["batch_averaged_latency_ms_per_review"],
                ],

                "Batch Size 1 (ms)": [
                    scratch["batch_size_1_latency_ms_per_review"],
                    distilbert["batch_size_1_latency_ms_per_review"],
                ],
            },

            index=[
                "Scratch Transformer",
                "DistilBERT",
            ],
        )

        st.bar_chart(
            latency_data,
            y_label="Milliseconds per review",
        )

        st.dataframe(
            latency_data.style.format("{:.3f} ms"),
            use_container_width=True,
        )

        st.divider()

        # -------------------------------------------------
        # EXPERIMENT INFORMATION
        # -------------------------------------------------

        st.header("Experiment Information")

        info_col1, info_col2 = st.columns(2)

        with info_col1:

            st.subheader("Dataset")

            st.write(
                "Dataset: IMDb Large Movie Review Dataset"
            )

            st.write(
                f"Test samples: {scratch['test_samples']:,}"
            )

            st.write("Dataset version: clean_v1")

            st.write("Labels: Negative / Positive")

        with info_col2:

            st.subheader("Hardware")

            st.write(
                f"Scratch device: {scratch['device']}"
            )

            st.write(
                f"DistilBERT device: {distilbert['device']}"
            )

            st.write(
                f"Scratch batch size: {scratch['batch_size']}"
            )

            st.write(
                "DistilBERT batch size: "
                f"{distilbert['inference_batch_size']}"
            )

        st.caption(
            "Test manifest SHA-256: "
            + scratch["manifest_sha256"]["test"]
        )

        st.divider()

        # -------------------------------------------------
        # KEY FINDINGS
        # -------------------------------------------------

        st.header("Key Findings")

        accuracy_gap = (
            distilbert["accuracy"] - scratch["accuracy"]
        ) * 100

        st.write(
            f"DistilBERT achieved an accuracy improvement of "
            f"{accuracy_gap:.2f} percentage points over "
            "the Scratch Transformer on the same test set."
        )

        st.write(
            "The Scratch Transformer has fewer parameters and "
            "lower measured forward latency, while DistilBERT "
            "achieves higher classification performance."
        )

        st.info(
            "The two models were evaluated on identical held-out "
            "test samples. The reported metrics are loaded directly "
            "from the saved evaluation JSON files."
        )

    except Exception as error:

        st.error(
            "Could not load the official evaluation results: "
            f"{error}"
        )

        st.exception(error)


# =========================================================
# 12. ABOUT THE PROJECT
# =========================================================

elif page == "About the Project":

    st.title("About the Project")

    st.write(
        """
        Transformer: From Scratch vs. Pretrained Model

        This project compares two Transformer approaches
        for binary sentiment analysis using the IMDb
        Large Movie Review Dataset.
        """
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Model A — Scratch Transformer")

        st.write(
            """
            A custom Transformer architecture implemented
            and trained from scratch using PyTorch.

            Its architecture includes:

            - Token Embeddings
            - Positional Embeddings
            - Multi-Head Attention
            - Feed-Forward Layers
            - Residual Connections
            - Layer Normalization
            - Classification Head
            """
        )

    with col2:

        st.subheader("Model B — DistilBERT")

        st.write(
            """
            A pretrained DistilBERT model fine-tuned
            for binary sentiment classification.

            Base checkpoint:

            `distilbert-base-uncased`

            The fine-tuned checkpoint is loaded locally
            for interactive inference.
            """
        )

    st.divider()

    st.header("Dataset Information")

    st.write(
        """
        Dataset: IMDb Large Movie Review Dataset

        Task: Binary Sentiment Analysis

        Labels:
        - 0 = Negative
        - 1 = Positive

        Dataset Splits:
        - Training: 19,824 reviews
        - Validation: 4,957 reviews
        - Test: 24,801 reviews
        """
    )

    st.divider()

    st.header("Project Objective")

    st.write(
        """
        The objective is to understand the internal
        architecture of Transformers by implementing
        one model from scratch and comparing it against
        a pretrained Transformer fine-tuned for the
        same NLP task.
        """
    )

    st.info(
        "Both models use the same clean dataset splits "
        "and the same held-out test examples."
    )


# =========================================================
# 13. FOOTER
# =========================================================

st.divider()

st.caption(
    "Transformer Comparison | NLP & Deep Learning Project"
)

st.caption(
    "Scratch Transformer vs. Fine-tuned DistilBERT"
)
