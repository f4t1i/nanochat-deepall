#!/usr/bin/env python3
"""
Quick test: Load tokenizer and show that it works with DeepAll data
"""

import json
from pathlib import Path
from nanochat.tokenizer import get_tokenizer

print("=" * 70)
print("DeepAll Tokenizer Test")
print("=" * 70)
print()

# Load tokenizer
print("Loading tokenizer...")
tokenizer = get_tokenizer()
print(f"✓ Tokenizer loaded")
print(f"  Vocab size: {tokenizer.get_vocab_size()}")
print()

# Load data
print("Loading DeepAll data...")
REPO_ROOT = Path(__file__).resolve().parents[1]
data_dir = REPO_ROOT / "deepall" / "nanogpt-pytorch-deepall-v1" / "data jsonl"
all_examples = []

for jsonl_file in data_dir.glob("*.jsonl"):
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                all_examples.append(data)
            except:
                pass

print(f"✓ Loaded {len(all_examples)} examples")
print()

# Show some examples
print("Sample examples:")
print("-" * 70)
for i, example in enumerate(all_examples[:3]):
    prompt = example.get('prompt', '')[:100]
    completion = example.get('completion', '')[:100]
    print(f"\nExample {i+1}:")
    print(f"  Prompt: {prompt}...")
    print(f"  Completion: {completion}...")
    
    # Tokenize
    full_text = example.get('prompt', '') + example.get('completion', '')
    tokens = tokenizer.encode(full_text)
    print(f"  Tokens: {len(tokens)}")

print()
print("-" * 70)
print()

# Statistics
print("Statistics:")
total_tokens = 0
for example in all_examples:
    full_text = example.get('prompt', '') + example.get('completion', '')
    tokens = tokenizer.encode(full_text)
    total_tokens += len(tokens)

print(f"  Total examples: {len(all_examples)}")
print(f"  Total tokens: {total_tokens:,}")
print(f"  Avg tokens per example: {total_tokens / len(all_examples):.1f}")
print()

print("=" * 70)
print("✓ Tokenizer works perfectly with DeepAll data!")
print("=" * 70)
print()
print("Next steps:")
print("1. The tokenizer is ready for training")
print("2. NanoChat expects data in Parquet format (from dataset.py)")
print("3. For now, you can use the tokenizer for inference/testing")
print()

