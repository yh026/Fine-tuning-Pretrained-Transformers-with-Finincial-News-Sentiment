# 🧠 Fine-tuning Pretrained Transformers with Financial News Sentiment

This project fine-tunes pretrained transformer models on a **financial news sentiment dataset**, allowing you to analyze the tone of financial news headlines using powerful language models such as **BERT** and **FinBERT**.

---

## 📊 Dataset

We use the [**Twitter Financial News Sentiment**](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) dataset from Hugging Face.

- **Description:** Annotated financial tweets/headlines with sentiment labels (Bearish, Bullish, Neutral)
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

### 1. Install dependencies (python==3.12.12)
```bash
!pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu126
```

### 2. Run the main script
```bash
python main.py
```

---

## ⚙️ Project Structure
```bash
├── main.py                # Entry point for training / evaluation
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
└── src/
```

---

## Results

- All the plots generated will be stored in results/

