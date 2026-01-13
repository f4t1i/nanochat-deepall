#!/usr/bin/env python3
"""
Weitertraining des DeepMaster GPT-2 Modells auf DeepALL-Daten.
"""

import torch
import torch.nn as nn
import tiktoken
from pathlib import Path
import sys
import json
import glob

# Unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).parent))
from convert_h5_to_pt import GPT, GPTConfig

# === Konfiguration ===
CHECKPOINT_PATH = Path("/home/deepall/nanochat/deepall/DeepMaster_converted.pt")
DATA_DIR = Path("/home/deepall/nanochat/deepall/nanogpt-pytorch-deepall-v1")
OUTPUT_PATH = Path("/home/deepall/nanochat/deepall/DeepMaster_finetuned.pt")

# Training Hyperparameter
BATCH_SIZE = 2
SEQ_LEN = 256
LEARNING_RATE = 1e-5  # Niedrig für Fine-tuning
NUM_ITERATIONS = 500
EVAL_INTERVAL = 50
SAVE_INTERVAL = 100


def load_training_data(tokenizer):
    """Lädt ALLE verfügbaren Trainingsdaten aus dem gesamten deepall-Ordner."""
    all_tokens = []
    deepall_root = Path("/home/deepall/nanochat/deepall")

    print(f"🔍 Scanne: {deepall_root}")

    # 1. Alle TXT-Dateien
    txt_files = list(deepall_root.rglob("*.txt"))
    print(f"\n📄 TXT-Dateien: {len(txt_files)}")
    for txt_file in txt_files:
        try:
            with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if len(content) > 50:
                    tokens = tokenizer.encode(content)
                    all_tokens.extend(tokens)
                    print(f"   ✓ {txt_file.name}: {len(tokens):,} Tokens")
        except Exception as e:
            continue

    # 2. Alle CSV-Dateien
    csv_files = list(deepall_root.rglob("*.csv"))
    print(f"\n📊 CSV-Dateien: {len(csv_files)}")
    for csv_file in csv_files:
        try:
            with open(csv_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                # CSV als strukturierten Text formatieren
                text = f"DATABASE_TABLE: {csv_file.name}\nCONTENT:\n{content}\n\n"
                tokens = tokenizer.encode(text)
                all_tokens.extend(tokens)
                print(f"   ✓ {csv_file.name}: {len(tokens):,} Tokens")
        except:
            continue

    # 3. Alle JSON-Dateien
    json_files = list(deepall_root.rglob("*.json"))
    print(f"\n🔧 JSON-Dateien: {len(json_files)}")
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
                text = f"CONFIG_SOURCE: {json_file.name}\nDATA: {json.dumps(data, ensure_ascii=False)}\n\n"
                tokens = tokenizer.encode(text)
                all_tokens.extend(tokens)
        except:
            continue
    print(f"   → JSON gesamt: {len(json_files)} Dateien geladen")

    # 4. Alle JSONL-Dateien
    jsonl_files = list(deepall_root.rglob("*.jsonl"))
    print(f"\n📋 JSONL-Dateien: {len(jsonl_files)}")
    for jsonl_file in jsonl_files:
        try:
            with open(jsonl_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        text = f"Instruction: {data.get('instruction', '')}\n"
                        text += f"Context: {data.get('context', '')}\n"
                        text += f"Response: {data.get('response', '')}\n\n"
                        tokens = tokenizer.encode(text)
                        all_tokens.extend(tokens)
                    except:
                        continue
            print(f"   ✓ {jsonl_file.name}")
        except:
            continue

    # 5. Alle MD-Dateien
    md_files = list(deepall_root.rglob("*.md"))
    print(f"\n📝 Markdown-Dateien: {len(md_files)}")
    for md_file in md_files:
        try:
            with open(md_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if len(content) > 50:
                    text = f"DOCUMENTATION: {md_file.name}\n{content}\n\n"
                    tokens = tokenizer.encode(text)
                    all_tokens.extend(tokens)
                    print(f"   ✓ {md_file.name}: {len(tokens):,} Tokens")
        except:
            continue

    print(f"\n{'='*50}")
    print(f"📊 GESAMT: {len(all_tokens):,} Tokens")
    print(f"{'='*50}")

    return all_tokens


def get_batch(tokens, batch_size, seq_len, device):
    """Erstellt einen Batch aus zufälligen Positionen."""
    ix = torch.randint(len(tokens) - seq_len - 1, (batch_size,))
    x = torch.stack([torch.tensor(tokens[i:i+seq_len]) for i in ix])
    y = torch.stack([torch.tensor(tokens[i+1:i+seq_len+1]) for i in ix])
    return x.to(device), y.to(device)


def train():
    print("=" * 60)
    print("DeepMaster Fine-Tuning")
    print("=" * 60)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️  Device: {device}")
    
    # Tokenizer
    print("\n📝 Lade Tokenizer...")
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Modell laden
    print(f"\n🧠 Lade Modell: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["model_args"])
    model = GPT(config)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   → {n_params:,} Parameter")
    
    # Trainingsdaten
    print("\n📚 Lade Trainingsdaten...")
    tokens = load_training_data(tokenizer)
    print(f"   → Gesamt: {len(tokens):,} Tokens")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_ITERATIONS)
    
    # Training
    print(f"\n🔥 Starte Training ({NUM_ITERATIONS} Iterationen)...")
    print("-" * 60)
    
    model.train()
    losses = []
    
    for iteration in range(NUM_ITERATIONS):
        x, y = get_batch(tokens, BATCH_SIZE, SEQ_LEN, device)
        
        logits, loss = model(x, y)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        losses.append(loss.item())
        
        # Zeige jeden 10. Step
        if (iteration + 1) % 10 == 0:
            avg_loss = sum(losses[-10:]) / min(10, len(losses))
            lr = scheduler.get_last_lr()[0]
            print(f"Iter {iteration+1:4d} | Loss: {avg_loss:.4f} | LR: {lr:.2e}", flush=True)
        
        if (iteration + 1) % SAVE_INTERVAL == 0:
            save_checkpoint(model, config, iteration + 1, losses[-1])
    
    # Finales Speichern
    save_checkpoint(model, config, NUM_ITERATIONS, losses[-1], final=True)
    
    # Test
    test_generation(model, tokenizer, device)


def save_checkpoint(model, config, iteration, loss, final=False):
    """Speichert einen Checkpoint."""
    path = OUTPUT_PATH if final else OUTPUT_PATH.with_suffix(f".iter{iteration}.pt")
    
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
        "iter_num": iteration,
        "loss": loss,
    }
    torch.save(checkpoint, path)
    print(f"💾 Checkpoint gespeichert: {path.name}")


def test_generation(model, tokenizer, device):
    """Testet die Generierung nach dem Training."""
    print("\n" + "=" * 60)
    print("🧪 Test-Generierung")
    print("=" * 60)
    
    model.eval()
    prompts = ["DeepMaster", "FATONI Protokoll:", "Modul M001"]
    
    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        x = torch.tensor([tokens], dtype=torch.long, device=device)
        
        with torch.no_grad():
            output = model.generate(x, max_new_tokens=50, temperature=0.7, top_k=40)
        
        text = tokenizer.decode(output[0].tolist())
        print(f"\n>>> {prompt}")
        print(text[:200])


if __name__ == "__main__":
    train()

