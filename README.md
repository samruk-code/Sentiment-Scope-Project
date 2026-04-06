# SentimentScope — Transformer-Based Sentiment Analysis

> A custom transformer model built from scratch for binary sentiment classification on IMDB movie reviews.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Business Context](#business-context)
- [How This Differs from a Generation Transformer](#how-this-differs-from-a-generation-transformer)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Results](#results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Skills Demonstrated](#skills-demonstrated)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

**SentimentScope** is an end-to-end NLP pipeline that trains a transformer model **from scratch** for binary sentiment classification. Given a movie review, the model predicts whether the sentiment is **positive** or **negative**, achieving over 75% accuracy on held-out test data.

This project was completed as part of the **Udacity AWS AI Scientist Nanodegree** and demonstrates practical skills in deep learning, natural language processing, and transformer architecture design.

---

## Business Context

As a Machine Learning Engineer at **CineScope** — an entertainment company that helps audiences discover movies and shows they love — the goal was to enhance the recommendation engine by understanding user sentiment about content they engage with.

SentimentScope solves this by automatically classifying user reviews as positive or negative. These sentiment signals feed directly into CineScope's personalization algorithms, enabling more accurate recommendations and improving user satisfaction at scale.

---

## How This Differs from a Generation Transformer

This project adapts a transformer from a generation task to a classification task. The key differences:

| Aspect | Generation Model | SentimentScope (Classification) |
|---|---|---|
| Data format | Continuous token stream with sliding window | Individual reviews paired with a label |
| Tokenization | Character-level | Subword (BERT BPE via `bert-base-uncased`) |
| Output | Next-token prediction (token-level) | Single pooled vector → 2 class logits |
| Pooling | Last token hidden state | Mean of all token embeddings |
| Training | Random shuffler over steps | Epoch-based (full passes over dataset) |

---

## Architecture

The model is built entirely in PyTorch from scratch, with only the tokenizer borrowed from `bert-base-uncased`:

```
Input Text
    │
    ▼
[BERT Tokenizer]               ← subword tokenization, padding & truncation (max_length=128)
    │
    ▼
[Token + Position Embeddings]  ← learned embeddings for tokens and positions
    │
    ▼
[Transformer Blocks] x8        ← decoder-style with causal (lower-triangular) masking
    │   ├── Layer Normalization        (pre-norm)
    │   ├── Multi-Head Self-Attention  (8 heads, head_size=32)
    │   ├── Residual Connection
    │   ├── Layer Normalization        (pre-norm)
    │   ├── Feed-Forward Network       (d_embed=256, 4x expansion, GELU)
    │   └── Residual Connection
    │
    ▼
[Mean Pooling]                 ← average all token embeddings → single vector
    │
    ▼
[Classification Head]          ← Linear layer → 2 logits (positive / negative)
    │
    ▼
Sentiment Label
```

**Hyperparameters:**

| Parameter | Value |
|---|---|
| Embedding dimension (`d_embed`) | 256 |
| Transformer blocks | 8 |
| Attention heads | 8 |
| Head size | 32 |
| Max sequence length | 128 |
| Batch size | 32 |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Loss function | CrossEntropyLoss |
| Dropout rate | 0.2 |
| Epochs | 10 |

---

## Dataset

- **Source:** [IMDB Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- **Size:** 50,000 reviews — 25,000 train / 25,000 test
- **Labels:** Binary — `positive` (1) / `negative` (0), balanced classes
- **Split:** Training set → 90% train (22,500) / 10% validation (2,500); test set held out for final evaluation

---

## Results

| Split | Accuracy |
|---|---|
| Validation | Tracked per epoch |
| **Test** | **> 75%** |

Validation accuracy and loss are tracked across all epochs to monitor for overfitting. The model is evaluated once on the held-out test set after training completes.

---

## Project Structure

```
Sentiment-Scope-Project/
│
├── SentimentScope_Project.ipynb   # Main notebook: EDA, architecture, training, evaluation
└── README.md
```

---

## Getting Started

### Prerequisites

```bash
pip install torch transformers pandas matplotlib
```

### Run the Notebook

1. Clone the repository:
   ```bash
   git clone https://github.com/samruk-code/Sentiment-Scope-Project.git
   cd Sentiment-Scope-Project
   ```

2. Download the IMDB dataset and place `aclImdb_v1.tar.gz` in the project root before running.

3. Launch Jupyter:
   ```bash
   jupyter notebook SentimentScope_Project.ipynb
   ```

4. Run all cells top-to-bottom. The notebook will:
   - Load and explore the IMDB dataset with visualizations
   - Tokenize reviews and build PyTorch `DataLoader` objects
   - Define and initialize the custom transformer architecture
   - Train with loss and accuracy tracked per epoch
   - Evaluate final performance on the held-out test set
   - Run sample predictions on new reviews

> **Note:** A GPU is strongly recommended for training. The notebook is compatible with AWS SageMaker, Google Colab, and local environments.

---

## Skills Demonstrated

- **Transformer architecture from scratch** — attention heads, multi-head attention, feed-forward blocks, and learned positional embeddings in PyTorch
- **Adapting transformers for classification** — mean pooling over token embeddings with a linear classification head instead of next-token prediction
- **HuggingFace tokenizer** — subword tokenization with `bert-base-uncased`, including padding, truncation, and attention masks
- **Custom PyTorch Dataset & DataLoader** — batching, shuffling, and preprocessing sequences for training
- **Training loop design** — loss calculation, AdamW optimization, and epoch-based training with per-epoch validation
- **Model evaluation** — accuracy on validation and test sets, overfitting detection across epochs
- **NLP data preprocessing** — exploratory analysis, label distribution, and review length analysis

---

## Acknowledgements

- Project brief provided by **Udacity / AWS AI Scientist Nanodegree**
- Dataset: [IMDB Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) — Maas et al., 2011
- Tokenizer: [bert-base-uncased](https://huggingface.co/bert-base-uncased) by Google via Hugging Face
