#!/usr/bin/env python3
"""
Create tokenizer for DeepAll data
"""

import os
from pathlib import Path
from nanochat.tokenizer import RustBPETokenizer
from nanochat.common import get_base_dir

print("=" * 60)
print("Creating Tokenizer for DeepAll Data")
print("=" * 60)
print()

# Get cache directory
base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
text_file = os.path.join(base_dir, "deepall_text.txt")

print(f"Base dir: {base_dir}")
print(f"Text file: {text_file}")
print(f"Tokenizer dir: {tokenizer_dir}")
print()

# Check if text file exists
if not os.path.exists(text_file):
    print(f"❌ Text file not found: {text_file}")
    print("Run train_simple_deepall.py first!")
    exit(1)

print("✓ Text file found")
print()

# Read text
print("Reading text...")
with open(text_file, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"✓ Read {len(text):,} characters")
print()

# Train tokenizer
print("Training tokenizer...")
print("  vocab_size: 8192")
print("  pattern: GPT-4 style")

tokenizer = RustBPETokenizer.train_from_iterator(
    iter([text]),  # Single document
    vocab_size=8192
)

print(f"✓ Tokenizer trained")
print(f"  vocab_size: {tokenizer.get_vocab_size()}")
print()

# Save tokenizer
print("Saving tokenizer...")
tokenizer.save(tokenizer_dir)
print(f"✓ Tokenizer saved to {tokenizer_dir}")
print()

# Create token_bytes.pt (required by base_train.py)
print("Creating token_bytes.pt...")
import torch

vocab_size = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
token_bytes = []
for token_id in range(vocab_size):
    token_str = token_strings[token_id]
    if token_str in special_set:
        token_bytes.append(0)
    else:
        id_bytes = len(token_str.encode("utf-8"))
        token_bytes.append(id_bytes)

token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
print(f"✓ Saved token_bytes to {token_bytes_path}")
print()

# Verify
print("Verifying tokenizer...")
from nanochat.tokenizer import get_tokenizer
tok = get_tokenizer()
print(f"✓ Tokenizer loaded successfully")
print(f"  vocab_size: {tok.get_vocab_size()}")
print()

# Test encoding
test_text = "DeepAll ist ein Framework"
encoded = tok.encode(test_text)
decoded = tok.decode(encoded)
print(f"Test encoding:")
print(f"  Input: {test_text}")
print(f"  Tokens: {len(encoded)}")
print(f"  Decoded: {decoded}")
print(f"  Match: {decoded == test_text}")
print()

print("=" * 60)
print("✓ Tokenizer ready!")
print("=" * 60)

