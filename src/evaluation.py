"""
Utility: Evaluation with FinBert
Author: Chen Yuhan
Last Edited: 2025.10.17
"""

from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments,logging,pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.nn import CrossEntropyLoss


def evaluate_finbert(dataset_split, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.to(device)
    model.eval()

    texts = dataset_split['text']
    true_labels = dataset_split['label']

    predictions = []
    all_losses = []
    loss_fct = CrossEntropyLoss()

    # Label mapping
    # Dataset: 0=Bearish, 1=Bullish, 2=Neutral
    # FinBERT: LABEL_0=Neutral, LABEL_1=Positive, LABEL_2=Negative
    # We need to map: Bearish->Negative(2), Bullish->Positive(1), Neutral->Neutral(0)
    
    dataset_to_finbert = {
        0: 2,  # Bearish -> Negative
        1: 1,  # Bullish -> Positive
        2: 0   # Neutral -> Neutral
    }

    finbert_to_dataset = {
        0: 2,  # Neutral -> Neutral
        1: 1,  # Positive -> Bullish
        2: 0   # Negative -> Bearish
    }

    with torch.no_grad():
        for i, text in enumerate(texts):
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get model output
            outputs = model(**inputs)
            logits = outputs.logits

            # Get prediction (FinBERT label)
            pred_finbert = torch.argmax(logits, dim=1).item()

            # Convert to dataset label
            pred_dataset = finbert_to_dataset[pred_finbert]
            predictions.append(pred_dataset)

            # Calculate loss using mapped labels
            true_label_finbert = dataset_to_finbert[true_labels[i]]
            labels = torch.tensor([true_label_finbert]).to(device)
            loss = loss_fct(logits, labels)
            all_losses.append(loss.item())

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    avg_loss = np.mean(all_losses)

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1,
        'predictions': predictions
    }

def evaluate_finbert_pipeline():
    finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

    # nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

    # Load dataset
    ds = load_dataset("zeroshot/twitter-financial-news-sentiment")

    # Evaluate
    print("\nEvaluating on validation set...")
    val_results = evaluate_finbert(ds['validation'], finbert, tokenizer)
    print(f"Validation Loss: {val_results['loss']:.4f}")
    print(f"Validation Accuracy: {val_results['accuracy']:.4f}")
    print(f"Validation F1: {val_results['f1']:.4f}")
