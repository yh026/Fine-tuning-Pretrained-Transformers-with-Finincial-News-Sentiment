# -*- coding: utf-8 -*-
"""
Assignment 3: Fine-tuning Pretrained Transformers
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
import re
import pandas as pd
from matplotlib import pyplot as plt

"""## Download the dataset"""

ds = load_dataset("zeroshot/twitter-financial-news-sentiment")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

"""## Preprocess the data"""

def clean_tweet(text):
    text = re.sub(r"http\S+", "", text)   # remove URLs
    text = re.sub(r"\$\w+", "", text)     # remove tickers like $BYND
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)  # remove mentions
    return text.strip()

def preprocess(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
    
ds = ds.map(lambda x: {"text": clean_tweet(x["text"])})
encoded_dataset = ds.map(preprocess, batched=True)
encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# GPU Usage Monitoring
def print_gpu_usage(tag=""):
    if torch.cuda.is_available():
        print(f"[{tag}] GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"[{tag}] GPU memory reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")

"""## Full Fine-tuning"""

print_gpu_usage("Initial State")

print("Start Full Fine-tuning")

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3,hidden_dropout_prob=0.3,  # add drop out
    attention_probs_dropout_prob=0.3)

training_args = TrainingArguments(
    output_dir="./results_BERT_full_finetune",
    do_eval=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=20,
    load_best_model_at_end=True,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_strategy="epoch",
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1": f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

print("Full fine-tune GPU usage:")
print_gpu_usage("Full")

trainer.evaluate()

full_logs = pd.DataFrame(trainer.state.log_history)
full_logs["setup"] = "Full Fine-Tuning"

del model, trainer
torch.cuda.empty_cache()

"""## LoRA"""

print_gpu_usage("LoRA Initial")

from peft import LoraConfig, get_peft_model

# 1. Load base model
model_lora = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

# 2. Configure LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],  # apply LoRA on attention layers
    lora_dropout=0.3,
    bias="none",
    task_type="SEQ_CLS",
)

# 3. Wrap model with LoRA
model_lora = get_peft_model(model_lora, lora_config)
model_lora.print_trainable_parameters()

# 4. Training
training_args = TrainingArguments(
    output_dir="./results_lora",
    do_eval=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-4,   # higher LR for LoRA
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=20,
    weight_decay=0.01,
    logging_dir="./logs_lora",
    logging_strategy="epoch",
)

trainer_lora = Trainer(
    model=model_lora,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer_lora.train()

print("LoRA fine-tune GPU usage:")
print_gpu_usage("LoRA")

trainer_lora.evaluate()

lora_logs = pd.DataFrame(trainer_lora.state.log_history)
lora_logs["setup"] = "LoRA Fine-Tuning"

"""## Evaluation and Comparison"""

# Combine logs
logs = pd.concat([full_logs, lora_logs])

# Filter only training logs (where eval_loss is NaN)
train_logs = logs[~logs["loss"].isna()]

plt.figure(figsize=(8, 5))
for setup, df in train_logs.groupby("setup"):
    plt.plot(df["epoch"], df["loss"], label=setup)

plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

eval_full = full_logs[full_logs["eval_loss"].notna()][["epoch", "eval_loss", "eval_accuracy", "eval_f1"]]
eval_lora = lora_logs[lora_logs["eval_loss"].notna()][["epoch", "eval_loss", "eval_accuracy", "eval_f1"]]
plt.figure(figsize=(8,5))
plt.plot(eval_full["epoch"][0:20], eval_full["eval_loss"][0:20], label="Full Fine-Tuning", marker="o")
plt.plot(eval_lora["epoch"], eval_lora["eval_loss"], label="LoRA Fine-Tuning", marker="o")

# Find the minimum loss for best performance epoch
min_idx_full = eval_full["eval_loss"][0:20].idxmin()
min_epoch_full = eval_full.loc[min_idx_full, "epoch"]
plt.axvline(x=min_epoch_full, color='blue', linestyle='--', alpha=0.7, label=f'Full Min (Epoch {min_epoch_full:.0f})')

min_idx_lora = eval_lora["eval_loss"].idxmin()
min_epoch_lora = eval_lora.loc[min_idx_lora, "epoch"]
plt.axvline(x=min_epoch_lora, color='orange', linestyle='--', alpha=0.7, label=f'LoRA Min (Epoch {min_epoch_lora:.0f})')

plt.title("Validation Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Eval Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

"""For both full fine-tuning and LoRA fine-tuning method, the minimum evalution loss happens when epoch=6. After epoch > 6, the loss keep increasing, which represents that the model is overfitting"""

# Keep only eval metrics
eval_logs = logs[logs["loss"].isna()]  # only evaluation steps

# Plot Accuracy & F1
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for setup, df in eval_logs.groupby("setup"):
    ax[0].plot(df["epoch"], df["eval_accuracy"], label=setup)
    ax[1].plot(df["epoch"], df["eval_f1"], label=setup)

    epoch_6_data = df[df["epoch"] == 6]

    acc_val = epoch_6_data["eval_accuracy"].values[0]
    f1_val = epoch_6_data["eval_f1"].values[0]

    ax[0].annotate(f'{acc_val:.3f}',
                  xy=(6, acc_val),
                  xytext=(10, 10),
                  textcoords='offset points',
                  bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                  arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    ax[1].annotate(f'{f1_val:.3f}',
                  xy=(6, f1_val),
                  xytext=(10, 10),
                  textcoords='offset points',
                  bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                  arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))


ax[0].set_title("Validation Accuracy per Epoch")
ax[1].set_title("Validation F1 per Epoch")
for a in ax:
    a.set_xlabel("Epoch")
    a.set_ylabel("Score")
    a.legend()
    a.grid(True)
plt.tight_layout()
plt.show()

"""### Evaluation with FinBERT"""

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

# Load dataset
ds = load_dataset("zeroshot/twitter-financial-news-sentiment")

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

def evaluate_finbert(dataset_split, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.to(device)
    model.eval()

    texts = dataset_split['text']
    true_labels = dataset_split['label']

    predictions = []
    all_losses = []
    loss_fct = CrossEntropyLoss()

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

# Evaluate on train set
print("Evaluating on training set...")
train_results = evaluate_finbert(ds['train'], finbert, tokenizer)
print(f"Train Loss: {train_results['loss']:.4f}")
print(f"Train Accuracy: {train_results['accuracy']:.4f}")
print(f"Train F1: {train_results['f1']:.4f}")

print("\nEvaluating on validation set...")
val_results = evaluate_finbert(ds['validation'], finbert, tokenizer)
print(f"Validation Loss: {val_results['loss']:.4f}")
print(f"Validation Accuracy: {val_results['accuracy']:.4f}")
print(f"Validation F1: {val_results['f1']:.4f}")

"""#### After cleaning"""

ds_clean = ds.map(lambda x: {"text": clean_tweet(x["text"])})

train_cl_results = evaluate_finbert(ds_clean['train'], finbert, tokenizer)
print(f"Train Loss: {train_results['loss']:.4f}")

print("\nEvaluating on validation set...")
val_cl_results = evaluate_finbert(ds_clean['validation'], finbert, tokenizer)
print(f"Validation Loss: {val_results['loss']:.4f}")
print(f"Validation Accuracy: {val_results['accuracy']:.4f}")
print(f"Validation F1: {val_results['f1']:.4f}")

