#!/usr/bin/env python3
"""
Simple training script for DeepAll data using NanoChat
Trainiert ein kleines Modell mit deinen JSONL-Daten
"""

import json
import os
import sys
from pathlib import Path

# Setup paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "deepall" / "nanogpt-pytorch-deepall-v1" / "data jsonl"
CACHE_DIR = Path.home() / ".cache" / "nanochat"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("DeepAll Simple Training")
print("=" * 60)
print()

# Step 1: Load and combine data
print("Step 1: Loading DeepAll data...")
all_data = []
for jsonl_file in DATA_DIR.glob("*.jsonl"):
    print(f"  Reading {jsonl_file.name}...")
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                all_data.append(data)
            except:
                pass

print(f"✓ Loaded {len(all_data)} examples")
print()

# Step 2: Convert to plain text for tokenizer
print("Step 2: Converting to plain text...")
text_file = CACHE_DIR / "deepall_text.txt"
with open(text_file, 'w', encoding='utf-8') as f:
    for item in all_data:
        prompt = item.get('prompt', '')
        completion = item.get('completion', '')
        combined = prompt + completion
        f.write(combined + '\n')

print(f"✓ Wrote {len(all_data)} examples to {text_file}")
print(f"  File size: {text_file.stat().st_size / 1024:.1f} KB")
print()

# Step 3: Create a simple tokenizer from the text
print("Step 3: Creating tokenizer...")
try:
    from nanochat.tokenizer import RustBPETokenizer
    
    # Read all text
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Train tokenizer
    print(f"  Training BPE tokenizer on {len(text):,} characters...")
    tokenizer = RustBPETokenizer.train_from_iterator(
        iter([text]),  # Single document
        vocab_size=8192
    )
    
    # Save tokenizer
    tokenizer_dir = CACHE_DIR / "tokenizer"
    tokenizer_dir.mkdir(exist_ok=True)
    tokenizer.save_to_directory(tokenizer_dir)
    print(f"✓ Tokenizer saved to {tokenizer_dir}")
    
except Exception as e:
    print(f"✗ Error creating tokenizer: {e}")
    print("  Continuing anyway...")

print()

# Step 4: Create training data in NanoChat format
print("Step 4: Creating training data...")
train_file = CACHE_DIR / "deepall_train.txt"
with open(train_file, 'w', encoding='utf-8') as f:
    for item in all_data:
        prompt = item.get('prompt', '')
        completion = item.get('completion', '')
        combined = prompt + completion
        f.write(combined + '\n')

print(f"✓ Training data: {train_file}")
print()

# Step 5: Summary
print("=" * 60)
print("✓ Data preparation complete!")
print("=" * 60)
print()
print("Next steps:")
print()
print("1. Train base model:")
print("   python -m scripts.base_train \\")
print("     --depth=4 \\")
print("     --max_seq_len=256 \\")
print("     --device_batch_size=1 \\")
print("     --total_batch_size=256 \\")
print("     --num_iterations=100 \\")
print("     --device_type=cpu")
print()
print("2. Fine-tune with your data:")
print("   python -m scripts.chat_sft \\")
print("     --device_batch_size=1 \\")
print("     --num_epochs=2 \\")
print("     --device_type=cpu")
print()
print("3. Chat with your model:")
print("   python -m scripts.chat_cli -p 'Was ist DeepAll?' --device-type=cpu")
print()

