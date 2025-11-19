"""
Prepare the GoEmotions dataset for training.
GoEmotions is a dataset of Reddit comments labeled with 27 emotion categories.
We'll format it as text generation: "Emotion: [emotion]\nText: [comment text]"

Dataset: https://github.com/google-research/google-research/tree/master/goemotions
"""

import os
import pandas as pd
import tiktoken
import numpy as np
from datasets import load_dataset

print("Downloading GoEmotions dataset from HuggingFace...")

# Load the dataset from HuggingFace
dataset = load_dataset("google-research-datasets/go_emotions", "simplified")

# Emotion labels
emotions = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

def format_example(example):
    """
    Format a single example as:
    Emotion: [emotion]
    Text: [text]
    <|endoftext|>

    Using <|endoftext|> as it's GPT-2's native special token for document separation.
    """
    # Get the emotion labels (can be multiple)
    emotion_ids = example['labels']
    if len(emotion_ids) == 0:
        emotion_labels = ['neutral']
    else:
        emotion_labels = [emotions[i] for i in emotion_ids]

    emotion_str = ', '.join(emotion_labels)
    text = example['text']

    return f"Emotion: {emotion_str}\nText: {text}\n<|endoftext|>"

print("Formatting training data...")
train_texts = [format_example(ex) for ex in dataset['train']]
train_data = ''.join(train_texts)

print("Formatting validation data...")
val_texts = [format_example(ex) for ex in dataset['validation']]
val_data = ''.join(val_texts)

print(f"Train examples: {len(train_texts)}")
print(f"Validation examples: {len(val_texts)}")
print(f"Total train characters: {len(train_data):,}")
print(f"Total val characters: {len(val_data):,}")

# Encode with tiktoken gpt2 bpe
print("Encoding with GPT-2 BPE tokenizer...")
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# Export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

output_dir = os.path.dirname(__file__)
train_ids.tofile(os.path.join(output_dir, 'train.bin'))
val_ids.tofile(os.path.join(output_dir, 'val.bin'))

print(f"Saved train.bin and val.bin to {output_dir}")
print("\nExample formatted data:")
print(train_texts[0])
print(train_texts[1])
