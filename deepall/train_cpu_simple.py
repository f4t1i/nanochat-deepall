#!/usr/bin/env python3
"""
Simple CPU training script for DeepAll using PyTorch's standard Transformer.
This bypasses nanochat's Flash Attention 3 requirement.
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add nanochat to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nanochat.tokenizer import get_tokenizer

print("=" * 70)
print("DeepAll Training with SimpleGPT (CPU)")
print("=" * 70)

# Setup
device = "cpu"
REPO_ROOT = Path(__file__).resolve().parents[1]
data_path = REPO_ROOT / "deepall" / "nanogpt-pytorch-deepall-v1" / "deepall 1-5  fr  nano big1.txt"

# Step 1: Load tokenizer
print("\nStep 1: Loading tokenizer...")
tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()
print(f"✓ Tokenizer loaded (vocab_size: {vocab_size})")

# Step 2: Load and tokenize data
print("\nStep 2: Loading and tokenizing data...")
with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()
    all_tokens = tokenizer.encode(text)
print(f"✓ Loaded {len(all_tokens):,} tokens from {data_path.name}")

# Step 3: Create simple model
print("\nStep 3: Creating simple model...")

class SimpleGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=256, n_layer=4, n_head=2, seq_len=256):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(seq_len, n_embd)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=n_embd,
                nhead=n_head,
                dim_feedforward=n_embd * 4,
                batch_first=True,
                dropout=0.1
            )
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.seq_len = seq_len

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)


model = SimpleGPT(vocab_size=vocab_size)
model = model.to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model created ({num_params:,} parameters)")

# Step 4: Training
print("\nStep 4: Training (500 iterations)...")
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
model.train()

num_iterations = 500
seq_len = 256
batch_size = 4
losses = []

for iteration in range(num_iterations):
    batch_x, batch_y = [], []
    for _ in range(batch_size):
        idx = torch.randint(0, len(all_tokens) - seq_len, (1,)).item()
        batch_x.append(all_tokens[idx:idx + seq_len])
        batch_y.append(all_tokens[idx + 1:idx + seq_len + 1])

    x = torch.tensor(batch_x, dtype=torch.long, device=device)
    y = torch.tensor(batch_y, dtype=torch.long, device=device)

    logits = model(x)
    loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        y.view(-1)
    )

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    losses.append(loss.item())

    if (iteration + 1) % 50 == 0:
        avg_loss = sum(losses[-50:]) / 50
        print(f"  Iteration {iteration + 1}/{num_iterations}, Loss: {avg_loss:.4f}")

# Step 5: Save checkpoint
print("\nStep 5: Saving checkpoint...")
checkpoint_dir = Path.home() / ".cache" / "nanochat" / "deepall_simple_checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

checkpoint = {
    "model_state_dict": model.state_dict(),
    "vocab_size": vocab_size,
    "n_embd": 256,
    "n_layer": 4,
    "n_head": 2,
    "seq_len": 256,
    "final_loss": losses[-1] if losses else None,
    "iterations": num_iterations,
}
checkpoint_path = checkpoint_dir / "model.pt"
torch.save(checkpoint, checkpoint_path)
print(f"✓ Saved to {checkpoint_path}")

# Step 6: Test generation
print("\nStep 6: Testing generation...")
model.eval()
with torch.no_grad():
    prompt = "DeepAll ist"
    prompt_tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    for _ in range(50):
        if input_ids.size(1) >= 256:
            input_ids = input_ids[:, -255:]
        logits = model(input_ids)
        probs = torch.softmax(logits[:, -1, :] / 0.8, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    generated_text = tokenizer.decode(input_ids[0].tolist())
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text[:300]}...")

print()
print("=" * 70)
print("✓ Training complete!")
print(f"Final Loss: {losses[-1]:.4f}" if losses else "No training done")
print(f"Checkpoint: {checkpoint_path}")
print("=" * 70)

