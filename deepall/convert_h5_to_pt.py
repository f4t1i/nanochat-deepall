#!/usr/bin/env python3
"""
Konvertiert DeepMaster Keras H5-Weights nach PyTorch GPT-2 Format.

Architektur:
- vocab_size: 50257
- block_size: 1024
- n_layer: 12
- n_head: 12
- n_embd: 768
- ~124M Parameter
"""

import h5py
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from dataclasses import dataclass

# === Pfade ===
H5_PATH = Path("/home/deepall/nanochat/deepall/nanogpt-pytorch-deepall-v1/data jsonl/DeepMaster_P100_v11_4_EVO.weights.h5")
OUTPUT_PATH = Path("/home/deepall/nanochat/deepall/DeepMaster_converted.pt")


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim**0.5))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def convert_weights(h5_path: Path, output_path: Path) -> GPT:
    """Konvertiert Keras H5 Weights nach PyTorch."""
    print("=" * 60)
    print("DeepMaster H5 → PyTorch Konvertierung")
    print("=" * 60)
    print(f"\nQuelle: {h5_path}")
    print(f"Ziel:   {output_path}\n")

    config = GPTConfig()
    model = GPT(config)

    with h5py.File(h5_path, "r") as f:
        # 1. Token Embeddings
        wte = f["layers/gpt2_backbone/layers/reversible_embedding/vars/0"][:]
        model.transformer.wte.weight.data = torch.from_numpy(wte).float()
        print(f"✓ Token Embeddings: {wte.shape}")

        # 2. Position Embeddings
        wpe = f["layers/gpt2_backbone/layers/position_embedding/vars/0"][:]
        model.transformer.wpe.weight.data = torch.from_numpy(wpe).float()
        print(f"✓ Position Embeddings: {wpe.shape}")

        # 3. Final LayerNorm
        ln_f_w = f["layers/gpt2_backbone/layers/layer_normalization/vars/0"][:]
        ln_f_b = f["layers/gpt2_backbone/layers/layer_normalization/vars/1"][:]
        model.transformer.ln_f.weight.data = torch.from_numpy(ln_f_w).float()
        model.transformer.ln_f.bias.data = torch.from_numpy(ln_f_b).float()
        print("✓ Final LayerNorm")

        # 4. Transformer Blocks
        for i in range(config.n_layer):
            suffix = "" if i == 0 else f"_{i}"
            prefix = f"layers/gpt2_backbone/layers/transformer_decoder{suffix}"
            block = model.transformer.h[i]

            # Attention LayerNorm (ln_1)
            ln1_w = f[f"{prefix}/_self_attention_layer_norm/vars/0"][:]
            ln1_b = f[f"{prefix}/_self_attention_layer_norm/vars/1"][:]
            block.ln_1.weight.data = torch.from_numpy(ln1_w).float()
            block.ln_1.bias.data = torch.from_numpy(ln1_b).float()

            # Q, K, V weights - Keras: (n_embd, n_head, head_dim) = (768, 12, 64)
            q_w = f[f"{prefix}/_self_attention_layer/query_dense/vars/0"][:]
            q_b = f[f"{prefix}/_self_attention_layer/query_dense/vars/1"][:]
            k_w = f[f"{prefix}/_self_attention_layer/key_dense/vars/0"][:]
            k_b = f[f"{prefix}/_self_attention_layer/key_dense/vars/1"][:]
            v_w = f[f"{prefix}/_self_attention_layer/value_dense/vars/0"][:]
            v_b = f[f"{prefix}/_self_attention_layer/value_dense/vars/1"][:]

            # Reshape: (768, 12, 64) -> (768, 768)
            q_w = q_w.reshape(config.n_embd, config.n_embd)
            k_w = k_w.reshape(config.n_embd, config.n_embd)
            v_w = v_w.reshape(config.n_embd, config.n_embd)
            q_b = q_b.reshape(config.n_embd)
            k_b = k_b.reshape(config.n_embd)
            v_b = v_b.reshape(config.n_embd)

            # Concatenate Q, K, V -> c_attn: (768, 2304)
            c_attn_w = np.concatenate([q_w, k_w, v_w], axis=1)
            c_attn_b = np.concatenate([q_b, k_b, v_b])
            block.attn.c_attn.weight.data = torch.from_numpy(c_attn_w.T).float()
            block.attn.c_attn.bias.data = torch.from_numpy(c_attn_b).float()

            # Output projection (c_proj): (12, 64, 768) -> (768, 768)
            o_w = f[f"{prefix}/_self_attention_layer/output_dense/vars/0"][:]
            o_b = f[f"{prefix}/_self_attention_layer/output_dense/vars/1"][:]
            o_w = o_w.reshape(config.n_embd, config.n_embd)
            block.attn.c_proj.weight.data = torch.from_numpy(o_w.T).float()
            block.attn.c_proj.bias.data = torch.from_numpy(o_b).float()

            # FFN LayerNorm (ln_2)
            ln2_w = f[f"{prefix}/_feedforward_layer_norm/vars/0"][:]
            ln2_b = f[f"{prefix}/_feedforward_layer_norm/vars/1"][:]
            block.ln_2.weight.data = torch.from_numpy(ln2_w).float()
            block.ln_2.bias.data = torch.from_numpy(ln2_b).float()

            # FFN c_fc: (768, 3072)
            fc_w = f[f"{prefix}/_feedforward_intermediate_dense/vars/0"][:]
            fc_b = f[f"{prefix}/_feedforward_intermediate_dense/vars/1"][:]
            block.mlp.c_fc.weight.data = torch.from_numpy(fc_w.T).float()
            block.mlp.c_fc.bias.data = torch.from_numpy(fc_b).float()

            # FFN c_proj: (3072, 768)
            proj_w = f[f"{prefix}/_feedforward_output_dense/vars/0"][:]
            proj_b = f[f"{prefix}/_feedforward_output_dense/vars/1"][:]
            block.mlp.c_proj.weight.data = torch.from_numpy(proj_w.T).float()
            block.mlp.c_proj.bias.data = torch.from_numpy(proj_b).float()

            print(f"✓ Transformer Block {i}")

    # Checkpoint speichern
    checkpoint = {
        "model": model.state_dict(),
        "model_args": {
            "vocab_size": config.vocab_size,
            "block_size": config.block_size,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_embd": config.n_embd,
            "dropout": config.dropout,
            "bias": config.bias,
        },
        "iter_num": 0,
        "best_val_loss": float("inf"),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)

    size_mb = output_path.stat().st_size / 1e6
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\n{'=' * 60}")
    print(f"✓ Konvertierung abgeschlossen!")
    print(f"  Checkpoint: {output_path}")
    print(f"  Größe: {size_mb:.1f} MB")
    print(f"  Parameter: {n_params:,}")
    print(f"{'=' * 60}")

    return model


def test_model(model: GPT):
    """Testet das konvertierte Modell mit einer kurzen Generierung."""
    import tiktoken

    print("\n🧪 Teste Modell-Generierung...")

    enc = tiktoken.get_encoding("gpt2")
    model.eval()

    prompt = "DeepALL ist"
    tokens = enc.encode(prompt)
    x = torch.tensor([tokens], dtype=torch.long)

    with torch.no_grad():
        output = model.generate(x, max_new_tokens=50, temperature=0.8, top_k=40)

    generated = enc.decode(output[0].tolist())
    print(f"\nPrompt: {prompt}")
    print(f"Generiert: {generated[:200]}...")


if __name__ == "__main__":
    model = convert_weights(H5_PATH, OUTPUT_PATH)
    test_model(model)

