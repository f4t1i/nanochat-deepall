"""
Kaggle Notebook Cells - Kopiere diese Zellen in dein Kaggle Notebook
"""

# ============================================================
# ZELLE 1: Setup & Dependencies
# ============================================================
print("=" * 60)
print("🚀 DeepMaster Fine-Tuning Setup")
print("=" * 60)

# Installiere Dependencies
import subprocess
import sys

print("\n📦 Installiere Dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tiktoken", "torch"])
print("✅ Dependencies installiert")

# Prüfe GPU
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n🖥️  Device: {device}")
if device == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================
# ZELLE 2: Prüfe Dataset
# ============================================================
print("\n" + "=" * 60)
print("📂 Prüfe Dataset")
print("=" * 60)

from pathlib import Path
import os

input_dir = Path('/kaggle/input/deepmaster')
print(f"\n📁 Input Directory: {input_dir}")
print(f"   Existiert: {input_dir.exists()}")

if input_dir.exists():
    print("\n   Dateien:")
    for f in sorted(input_dir.iterdir()):
        size = f.stat().st_size / (1024**2) if f.is_file() else 0
        if f.is_file():
            print(f"   ✅ {f.name} ({size:.1f} MB)")
        else:
            print(f"   📁 {f.name}/")
else:
    print("\n   ❌ FEHLER: Dataset nicht gefunden!")
    print("   Stelle sicher, dass du das Dataset als INPUT hinzugefügt hast!")

# ============================================================
# ZELLE 3: Lade Modell
# ============================================================
print("\n" + "=" * 60)
print("🧠 Lade Modell")
print("=" * 60)

model_path = input_dir / 'DeepMaster_converted.pt'
print(f"\n📂 Modell: {model_path}")
print(f"   Existiert: {model_path.exists()}")

if model_path.exists():
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    print(f"   ✅ Geladen")
    print(f"   Config: {ckpt['model_args']}")
else:
    print(f"   ❌ FEHLER: Modell nicht gefunden!")

# ============================================================
# ZELLE 4: Lade Trainingsdaten
# ============================================================
print("\n" + "=" * 60)
print("📚 Lade Trainingsdaten")
print("=" * 60)

data_path = input_dir / 'training_data.txt'
print(f"\n📄 Daten: {data_path}")
print(f"   Existiert: {data_path.exists()}")

if data_path.exists():
    all_text = data_path.read_text(encoding='utf-8', errors='ignore')
    print(f"   ✅ Geladen")
    print(f"   Größe: {len(all_text):,} Zeichen")
    
    # Tokenize
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(all_text)
    print(f"   Tokens: {len(tokens):,}")
else:
    print(f"   ❌ FEHLER: Trainingsdaten nicht gefunden!")

# ============================================================
# ZELLE 5: Starte Training
# ============================================================
print("\n" + "=" * 60)
print("🔥 Starte Training")
print("=" * 60)

# Führe das Training Script aus
exec(open('/kaggle/input/deepmaster/kaggle_train.py').read())

# ============================================================
# ZELLE 6: Verify Output
# ============================================================
print("\n" + "=" * 60)
print("✅ Verify Output")
print("=" * 60)

output_dir = Path('/kaggle/working')
output_file = output_dir / 'DeepMaster_finetuned.pt'

print(f"\n📂 Output: {output_file}")
print(f"   Existiert: {output_file.exists()}")

if output_file.exists():
    size = output_file.stat().st_size / (1024**2)
    print(f"   ✅ Größe: {size:.1f} MB")
    
    # Lade und prüfe
    ckpt = torch.load(output_file, map_location=device, weights_only=False)
    print(f"   ✅ Kann geladen werden")
    print(f"   Keys: {list(ckpt.keys())}")
    print(f"\n✅ Training erfolgreich!")
else:
    print(f"   ❌ FEHLER: Output nicht gefunden!")

