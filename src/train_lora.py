# -*- coding: utf-8 -*-
"""
Utility: LoRA fine-tuning
Author: Chen Yuhan
Last Edited: 2025.10.17
"""


from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments,logging,pipeline
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from peft import LoraConfig, get_peft_model

from src.train_full import compute_metrics

def lora_fine_tuning(encoded_dataset,tokenizer):
    # Load base model
    model_lora = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],  # apply LoRA on attention layers
        lora_dropout=0.3,
        bias="none",
        task_type="SEQ_CLS",
    )

    # Wrap model with LoRA
    model_lora = get_peft_model(model_lora, lora_config)
    model_lora.print_trainable_parameters()

    # Training
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

    lora_logs = pd.DataFrame(trainer_lora.state.log_history)
    lora_logs["setup"] = "LoRA Fine-Tuning"

    

    return lora_logs
