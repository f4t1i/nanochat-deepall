#!/usr/bin/env python3
"""
Frage das DeepMaster Modell
"""
import torch
import torch.nn.functional as F
import tiktoken
import sys
from pathlib import Path

# GPT Model Definition (muss identisch sein wie beim Training)
import torch.nn as nn
import math

class GPTConfig:
    def __init__(self, **kwargs):
        self.n_layer = kwargs.get('n_layer', 12)
        self.n_head = kwargs.get('n_head', 12)
        self.n_embd = kwargs.get('n_embd', 768)
        self.vocab_size = kwargs.get('vocab_size', 50257)
        self.block_size = kwargs.get('block_size', 1024)
        self.dropout = kwargs.get('dropout', 0.0)
        self.bias = kwargs.get('bias', True)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x).split(self.n_embd, dim=2)
        q, k, v = [t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in qkv]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)

@torch.no_grad()
def generate(model, idx, max_new_tokens=100, temperature=0.8, top_k=40):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -1024:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def main():
    model_path = Path(__file__).parent / "DeepMaster_converted.pt"
    print(f"🧠 Lade Modell: {model_path}")
    
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    config = GPTConfig(**ckpt['model_args'])
    model = GPT(config)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"✅ Modell geladen ({sum(p.numel() for p in model.parameters()):,} Parameter)")
    
    enc = tiktoken.get_encoding("gpt2")
    
    # Prompt
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Was ist DeepFlow?"
    print(f"\n💬 Prompt: {prompt}\n")
    print("=" * 50)
    
    ids = enc.encode(prompt)
    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    
    y = generate(model, x, max_new_tokens=150, temperature=0.7, top_k=50)
    output = enc.decode(y[0].tolist())
    print(output)
    print("=" * 50)

if __name__ == "__main__":
    import io
    import sys

    # Output in Datei schreiben
    with open("/home/deepall/nanochat/deepall/model_antwort.txt", "w") as f:
        old_stdout = sys.stdout
        sys.stdout = f
        main()
        sys.stdout = old_stdout

    # Auch anzeigen
    with open("/home/deepall/nanochat/deepall/model_antwort.txt", "r") as f:
        print(f.read())

