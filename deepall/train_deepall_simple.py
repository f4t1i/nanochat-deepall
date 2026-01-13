#!/usr/bin/env python3
"""
Simple training script for DeepAll using NanoChat components
Trainiert ein kleines Modell direkt mit deinen JSONL-Daten
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from nanochat.tokenizer import get_tokenizer
from nanochat.gpt import GPT

print("=" * 70)
print("DeepAll Simple Training")
print("=" * 70)
print()

# Setup
device = "cpu"
REPO_ROOT = Path(__file__).resolve().parents[1]
data_dir = REPO_ROOT / "deepall" / "nanogpt-pytorch-deepall-v1" / "data jsonl"
cache_dir = Path.home() / ".cache" / "nanochat"

# Load tokenizer
print("Loading tokenizer...")
tokenizer = get_tokenizer()
print(f"✓ Tokenizer loaded (vocab_size: {tokenizer.get_vocab_size()})")
print()

# Load data
print("Loading DeepAll data...")
all_texts = []
for jsonl_file in data_dir.glob("*.jsonl"):
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                prompt = data.get('prompt', '')
                completion = data.get('completion', '')
                combined = prompt + completion
                all_texts.append(combined)
            except:
                pass

print(f"✓ Loaded {len(all_texts)} examples")
print()

# Tokenize
print("Tokenizing data...")
all_tokens = []
for text in all_texts:
    tokens = tokenizer.encode(text)
    all_tokens.extend(tokens)

print(f"✓ Total tokens: {len(all_tokens):,}")
print()

# Create model
print("Creating model...")
model = GPT(
    vocab_size=tokenizer.get_vocab_size(),
    num_layers=4,
    model_dim=256,
    num_heads=2,
    num_kv_heads=2,
    max_seq_len=256,
)
model = model.to(device)
print(f"✓ Model created")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
print()

# Training loop (very simple)
print("Training...")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
model.train()

num_iterations = 10
seq_len = 256

for iteration in range(num_iterations):
    # Sample random batch
    if len(all_tokens) < seq_len + 1:
        print(f"⚠️  Not enough tokens ({len(all_tokens)}) for seq_len={seq_len}")
        break
    
    idx = torch.randint(0, len(all_tokens) - seq_len, (1,)).item()
    x = torch.tensor(all_tokens[idx:idx+seq_len], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(all_tokens[idx+1:idx+seq_len+1], dtype=torch.long, device=device).unsqueeze(0)
    
    # Forward pass
    logits = model(x)
    loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        y.view(-1)
    )
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (iteration + 1) % 5 == 0:
        print(f"  Iteration {iteration+1}/{num_iterations}, Loss: {loss.item():.4f}")

print()
print("=" * 70)
print("✓ Training complete!")
print("=" * 70)
print()
print("Model is ready for inference!")
print()
print("Next: Test the model with:")
print("  python -c \"from nanochat.tokenizer import get_tokenizer; from nanochat.model import GPT; tok = get_tokenizer(); model = GPT(...); print('Ready!')\"")
print()

