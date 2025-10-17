# 🧠 Fine-tuning Pretrained Transformers with Financial News Sentiment

This project fine-tunes pretrained transformer models on a **financial news sentiment dataset**, allowing you to analyze the tone of financial news headlines using powerful language models such as **BERT** and **FinBERT**.

---

## 📊 Dataset

We use the [**Twitter Financial News Sentiment**](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) dataset from Hugging Face.

- **Description:** Annotated financial tweets/headlines with sentiment labels (Positive, Negative, Neutral)
- **Source:** Hugging Face Datasets

---

## 🧩 Models

Two pretrained transformer models are supported:

| Model | Description | Hugging Face Link |
|--------|--------------|------------------|
| **BERT Base Uncased** | General-purpose pretrained BERT model | [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) |
| **FinBERT-Tone** | BERT model fine-tuned on financial text for sentiment analysis | [yiyanghkust/finbert-tone](https://huggingface.co/yiyanghkust/finbert-tone) |

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the main script

python main.py



