# -*- coding: utf-8 -*-
"""
Utility: Data loading and preprocessing
Author: Chen Yuhan
Last Edited: 2025.10.17
"""
import re
from datasets import load_dataset
from transformers import BertTokenizer

def clean_tweet(text):
    """
    Clean tweet text by removing URLs, stock tickers, and mentions.
    
    Args:
        text (str): Raw tweet text
        
    Returns:
        str: Cleaned text
    """
    text = re.sub(r"http\S+", "", text)   # remove URLs
    text = re.sub(r"\$\w+", "", text)     # remove tickers like $BYND
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)  # remove mentions
    return text.strip()

def preprocess(examples, tokenizer, max_length=128):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)
 
def load_and_preprocess_data():
    """
    Load dataset and apply preprocessing.
    
    Args:
        config: Configuration object
        
    Returns:
        DatasetDict: Preprocessed and encoded dataset
    """
    # Load dataset
    print(f"Loading dataset...")
    dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")
    
    # Clean tweets
    print("Cleaning tweet text...")
    dataset = dataset.map(lambda x: {"text": clean_tweet(x["text"])})
    
    # Load tokenizer
    print(f"Loading tokenizer:")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Tokenize dataset
    print("Tokenizing dataset...")
    encoded_dataset = dataset.map(
        lambda x: preprocess(x, tokenizer, 128),
        batched=True
    )
    
    # Set format for PyTorch
    encoded_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )
    
    print(f"✓ Preprocessing complete")
    print(f"  Train samples: {len(encoded_dataset['train'])}")
    print(f"  Validation samples: {len(encoded_dataset['validation'])}")
    
    return encoded_dataset, tokenizer
