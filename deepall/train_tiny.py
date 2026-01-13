#!/usr/bin/env python3
"""
TinyGPT Training - 1M Parameter Modell für CPU
Trainiert in ~10 Minuten auf CPU
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# ============== TINY GPT MODEL (1M params) ==============
class TinyGPT(nn.Module):
    def __init__(self, vocab_size=50257, n_embd=128, n_head=4, n_layer=4, block_size=256):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight  # weight tying
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x).split(C, dim=2)
        q, k, v = [t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in qkv]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))

# ============== TRAINING ==============
def main():
    print("=" * 60)
    print("🚀 TinyGPT Training (1M Parameter, CPU-optimiert)")
    print("=" * 60)
    
    # Config
    block_size = 256
    batch_size = 8
    max_iters = 200
    lr = 3e-4
    
    # Load data
    data_dir = Path(__file__).parent
    all_text = ""
    for f in data_dir.glob("*.txt"):
        all_text += f.read_text(encoding="utf-8", errors="ignore") + "\n"
    print(f"📚 {len(all_text):,} Zeichen geladen")
    
    # Simple char-level tokenizer (schneller als tiktoken)
    chars = sorted(list(set(all_text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi.get(c, 0) for c in s]
    decode = lambda l: ''.join([itos.get(i, '') for i in l])
    
    data = torch.tensor(encode(all_text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    print(f"📊 Train: {len(train_data):,} | Val: {len(val_data):,} tokens")
    print(f"📝 Vocab: {vocab_size} chars")
    
    # Model
    model = TinyGPT(vocab_size=vocab_size, n_embd=128, n_head=4, n_layer=4, block_size=block_size)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Modell: {n_params:,} Parameter ({n_params/1e6:.2f}M)")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    def get_batch(split):
        d = train_data if split == 'train' else val_data
        ix = torch.randint(len(d) - block_size, (batch_size,))
        x = torch.stack([d[i:i+block_size] for i in ix])
        y = torch.stack([d[i+1:i+block_size+1] for i in ix])
        return x, y
    
    print(f"\n🔥 Training startet ({max_iters} Iterationen)...")
    for i in range(max_iters):
        xb, yb = get_batch('train')
        _, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 20 == 0 or i == max_iters - 1:
            model.eval()
            with torch.no_grad():
                xv, yv = get_batch('val')
                _, val_loss = model(xv, yv)
            model.train()
            print(f"  Step {i:4d} | Train: {loss.item():.4f} | Val: {val_loss.item():.4f}")
    
    # Save
    out_path = data_dir / "TinyGPT_trained.pt"
    torch.save({
        'model': model.state_dict(),
        'vocab': {'stoi': stoi, 'itos': itos},
        'config': {'vocab_size': vocab_size, 'n_embd': 128, 'n_head': 4, 'n_layer': 4}
    }, out_path)
    print(f"\n✅ Gespeichert: {out_path}")
    
    # Generate sample
    print("\n📝 Beispiel-Generierung:")
    model.eval()
    ctx = torch.zeros((1, 1), dtype=torch.long)
    for _ in range(200):
        logits, _ = model(ctx[:, -block_size:])
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        ctx = torch.cat([ctx, next_tok], dim=1)
    print(decode(ctx[0].tolist()))

if __name__ == "__main__":
    main()

