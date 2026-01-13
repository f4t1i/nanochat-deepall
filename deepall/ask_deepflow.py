#!/usr/bin/env python3
"""DeepMaster Inference - exakte nanoGPT Architektur"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

print("=" * 50)
print("🧠 DeepMaster Inference")
print("=" * 50)

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

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(T, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        return self.lm_head(self.transformer.ln_f(x))

print("📂 Lade Modell...")
ckpt = torch.load('deepall/DeepMaster_converted.pt', map_location='cpu', weights_only=False)
args = ckpt['model_args']
print(f"   n_layer={args['n_layer']}, n_head={args['n_head']}, n_embd={args['n_embd']}")

model = GPT(args['vocab_size'], args['n_embd'], args['n_head'], args['n_layer'], args['block_size'], args.get('bias', True))
model.load_state_dict(ckpt['model'], strict=False)
model.eval()
print(f"✅ {sum(p.numel() for p in model.parameters()):,} Parameter geladen")

enc = tiktoken.get_encoding("gpt2")

prompts = [
    "Was ist DeepFlow?",
    "DeepFlow analysiert",
    "M005 DeepFlow",
    "Ursachen-Wirkungs-Ketten",
]

for prompt in prompts:
    print(f"\n💬 Prompt: {prompt}")
    print("-" * 50)

    ids = torch.tensor(enc.encode(prompt)).unsqueeze(0)

    with torch.no_grad():
        for i in range(25):
            logits = model(ids[:, -1024:])
            logits = logits[:, -1, :] / 0.9
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_id], dim=1)

    print(enc.decode(ids[0].tolist()))
    print()

