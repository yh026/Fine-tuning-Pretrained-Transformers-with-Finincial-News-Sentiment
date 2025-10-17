"""
Utility: Train BERT base model with Full Fine-tuning
Author: Chen Yuhan
Last Edited: 2025.10.17
"""

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments,logging,pipeline
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd
import torch

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1": f1}
def full_fine_tuning(encoded_dataset,tokenizer):

    print("Starting Full Fine-Tuning")
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
        # num_train_epochs=20,
        num_train_epochs = 2, # for debug
        load_best_model_at_end=True,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Starting Training")
    trainer.train()
    full_logs = pd.DataFrame(trainer.state.log_history)
    full_logs["setup"] = "Full Fine-Tuning"

    # Release GPU memory
    del model, trainer
    torch.cuda.empty_cache()

    return full_logs
