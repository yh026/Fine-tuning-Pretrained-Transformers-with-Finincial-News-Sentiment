# -*- coding: utf-8 -*-
"""
Assignment 3: Fine-tuning Pretrained Transformers
Author: Chen Yuhan
Last Edited: 2025.10.17
"""


import torch

from src.data_preprocessing import load_and_preprocess_data
from src.train_full import full_fine_tuning
from src.train_lora import lora_fine_tuning
from src.visualization import create_comparison_plots
from src.evaluation import evaluate_finbert_pipeline
def print_gpu_usage(tag=""):
    if torch.cuda.is_available():
        print(f"[{tag}] GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"[{tag}] GPU memory reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")

def main():
    encoded_dataset,tokenizer = load_and_preprocess_data()
    
    # Full fine-tuning
    print_gpu_usage("Initial State")
    full_logs = full_fine_tuning(encoded_dataset,tokenizer)
    print("Full fine-tune GPU usage:")
    print_gpu_usage("Full")

    # LoRA fine-tuning
    print_gpu_usage("LoRA Initial")
    lora_logs = lora_fine_tuning(encoded_dataset,tokenizer)
    print("LoRA fine-tune GPU usage:")
    print_gpu_usage("LoRA")

    # Create comparison figures
    create_comparison_plots(full_logs, lora_logs)

    # Evaluation with other Fine-tuned Models
    evaluate_finbert_pipeline()

if __name__ == "__main__":
    main()