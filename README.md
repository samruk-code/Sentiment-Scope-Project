# 🎬 SentimentScope — Transformer-Based Sentiment Analysis

> Sentiment classification on IMDB movie reviews using a custom BERT-based transformer model.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Business Context](#business-context)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Results](#results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Key Concepts & Skills Demonstrated](#key-concepts--skills-demonstrated)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

**SentimentScope** is an end-to-end NLP pipeline that fine-tunes a transformer model for binary sentiment classification. Given a movie review, the model predicts whether the sentiment is **positive** or **negative** with over 75% accuracy on held-out test data.

This project was completed as part of the **AWS AI Scientist Nanodegree** and demonstrates practical skills in deep learning, natural language processing, and transformer architecture design.

---

## Business Context

Imagine you're a Machine Learning Engineer at **CineScope**, an entertainment company building a next-generation recommendation engine. Understanding how users *feel* about content — not just what they watch — is the key to personalizing suggestions at scale.

SentimentScope solves this by automatically classifying user reviews as positive or negative, feeding that signal directly into CineScope's recommendation algorithms to improve user satisfaction and engagement.

---

## Architecture

The model adapts a `bert-base-uncased` transformer backbone for binary classification:

```
Input Text
    │
    ▼
[BERT Tokenizer]  ← subword tokenization, padding & truncation
    │
    ▼
[Transformer Encoder]  ← multi-head self-attention + feed-forward layers
    │
    ▼
[Mean Pooling]  ← condenses all token embeddings → single vector
    │
    ▼
[Classification Head]  ← linear layer → 2 logits (positive / negative)
    │
    ▼
Sentiment Label
```

**Key design decisions vs. a generation model:**

| Aspect | Generation Model | SentimentScope (Classification) |
|---|---|---|
| Data format | Continuous token stream | Review + label pairs |
| Tokenization | Character-level | Subword (BERT BPE) |
| Output | Next-token prediction | Binary class logits |
| Pooling | Last token hidden state | Mean of all token embeddings |
| Training | Sliding window | Epoch-based (full-pass) |

---

## Dataset

- **Source:** [IMDB Large Movie Review Dataset](https://huggingface.co/datasets/imdb) via HuggingFace `datasets`
- **Size:** 50,000 reviews (25k train / 25k test)
- **Labels:** Binary — `positive` / `negative`
- **Split used:** Train → 80% training / 20% validation, with provided test set held out

---

## Results

| Split | Accuracy |
|---|---|
| Training | tracked per epoch |
| Validation | tracked per epoch |
| **Test** | **> 75%** ✅ |

The model checkpoint achieving >75% test accuracy is saved and included in the repository.

---

## Project Structure

```
Sentiment-Scope-Project/
│
├── SentimentScope_Project.ipynb   # Main notebook: EDA, model, training, evaluation
├── model_checkpoint/              # Saved model weights (.pt file)
├── .gitignore
└── README.md
```

---

## Getting Started

### Prerequisites

```bash
pip install torch transformers datasets scikit-learn matplotlib
```

### Run the notebook

1. Clone the repository:
   ```bash
   git clone https://github.com/samruk-code/Sentiment-Scope-Project.git
   cd Sentiment-Scope-Project
   ```

2. Launch Jupyter:
   ```bash
   jupyter notebook SentimentScope_Project.ipynb
   ```

3. Run all cells top-to-bottom. The notebook will:
   - Download and explore the IMDB dataset
   - Tokenize and build PyTorch `DataLoader` objects
   - Initialize and customize the transformer model
   - Train with validation tracking across epochs
   - Evaluate on the held-out test set

> **Note:** A GPU is strongly recommended for training. The notebook is compatible with AWS SageMaker, Google Colab, and local environments.

---

## Key Concepts & Skills Demonstrated

- **Transformer architecture** — understanding attention mechanisms and adapting them for classification
- **HuggingFace ecosystem** — using `bert-base-uncased` tokenizer and pre-trained weights
- **Custom PyTorch Dataset & DataLoader** — batching, shuffling, and padding sequences
- **Training loop design** — loss calculation, optimizer steps, gradient clipping
- **Validation & overfitting detection** — tracking loss and accuracy across epochs
- **Model checkpointing** — saving the best-performing model weights
- **NLP data preprocessing** — tokenization, truncation, attention masks

---

## Acknowledgements

- Project brief provided by **Udacity / AWS AI Scientist Nanodegree**
- Dataset: [IMDB Dataset](https://huggingface.co/datasets/imdb) via HuggingFace
- Model backbone: [bert-base-uncased](https://huggingface.co/bert-base-uncased) by Google
