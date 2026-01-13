#!/usr/bin/env python3
"""
DeepMaster Fine-Tuning für Kaggle GPU
Optimiert für schnelles Training mit 8x GPU
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json
import time

# ============== GPT MODEL ==============
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, bias=True):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x).split(self.n_embd, dim=2)
        q, k, v = [t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in qkv]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))

class MLP(nn.Module):
    def __init__(self, n_embd, bias=True):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, bias=True):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd, bias=bias)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, bias)
        self.ln_2 = nn.LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, bias)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, bias=True):
        super().__init__()
        self.block_size = block_size
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd),
            wpe = nn.Embedding(block_size, n_embd),
            h = nn.ModuleList([Block(n_embd, n_head, block_size, bias) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd, bias=bias),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(T, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        logits = self.lm_head(self.transformer.ln_f(x))
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# ============== TRAINING ==============
def main():
    print("=" * 60)
    print("🚀 DeepMaster Fine-Tuning (Kaggle GPU)")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")
    
    # Config
    block_size = 1024
    batch_size = 32 if device == "cuda" else 4
    learning_rate = 3e-4
    max_iters = 1000
    eval_interval = 100
    
    # Load model
    print("\n📂 Lade Modell...")
    ckpt = torch.load('/kaggle/input/deepmaster/DeepMaster_converted.pt', map_location=device, weights_only=False)
    args = ckpt['model_args']
    
    model = GPT(args['vocab_size'], args['n_embd'], args['n_head'], args['n_layer'], args['block_size'], args.get('bias', True))
    model.load_state_dict(ckpt['model'], strict=False)
    model.to(device)
    model.train()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✅ {n_params:,} Parameter ({n_params/1e6:.1f}M)")
    
    # Load training data
    print("\n📚 Lade Trainingsdaten...")
    data_path = Path('/kaggle/input/deepmaster/training_data.txt')
    if not data_path.exists():
        # Fallback: combine all txt files
        all_text = ""
        for f in Path('/kaggle/input/deepmaster').glob('*.txt'):
            all_text += f.read_text(encoding='utf-8', errors='ignore') + "\n"
    else:
        all_text = data_path.read_text(encoding='utf-8', errors='ignore')
    
    print(f"   {len(all_text):,} Zeichen")
    
    # Tokenize with tiktoken
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    data = torch.tensor(enc.encode(all_text), dtype=torch.long)
    
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    print(f"   Train: {len(train_data):,} | Val: {len(val_data):,} tokens")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    def get_batch(split):
        d = train_data if split == 'train' else val_data
        ix = torch.randint(len(d) - block_size, (batch_size,))
        x = torch.stack([d[i:i+block_size] for i in ix])
        y = torch.stack([d[i+1:i+block_size+1] for i in ix])
        return x.to(device), y.to(device)
    
    # Training loop
    print(f"\n🔥 Training ({max_iters} Iterationen)...")
    start_time = time.time()
    
    for iter in range(max_iters):
        if iter % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                xv, yv = get_batch('val')
                _, val_loss = model(xv, yv)
            model.train()
            elapsed = time.time() - start_time
            print(f"  Iter {iter:4d} | Val Loss: {val_loss.item():.4f} | Time: {elapsed:.1f}s")
        
        xb, yb = get_batch('train')
        _, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Save
    out_dir = Path('/kaggle/working')
    out_dir.mkdir(exist_ok=True)
    
    torch.save({
        'model': model.state_dict(),
        'model_args': args,
    }, out_dir / 'DeepMaster_finetuned.pt')
    
    print(f"\n✅ Modell gespeichert: {out_dir / 'DeepMaster_finetuned.pt'}")
    print(f"⏱️  Gesamtzeit: {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    main()

