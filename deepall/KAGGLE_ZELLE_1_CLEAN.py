#!/usr/bin/env python3
# ============================================================================
# KAGGLE NOTEBOOK - ZELLE 1: Repository + Dateien laden
# ============================================================================
# Kopiere DIESEN KOMPLETTEN Code in eine neue Zelle im Kaggle Notebook
# NICHT vermischen mit anderen Zellen!
# ============================================================================

import os
import shutil

# Lösche altes Repository
repo_path = "/kaggle/working/nanochat-deepall"
if os.path.exists(repo_path):
    print("🗑️  Lösche altes Repository...")
    shutil.rmtree(repo_path)

# Clone Repository
print("📥 Clone Repository...")
os.system("cd /kaggle/working && git clone https://github.com/f4t1i/nanochat-deepall.git")

# Installiere Dependencies
print("📦 Installiere Dependencies...")
os.system("pip install -q torch transformers datasets tqdm numpy pandas scikit-learn")

# Prüfe Repository
if os.path.exists(f"{repo_path}/deepall"):
    print("✅ Repository geladen!")
else:
    print("❌ Repository Problem!")
    exit()

# Prüfe Training Data
deepall_path = f"{repo_path}/deepall"
training_file = f"{deepall_path}/training_data.txt"
if os.path.exists(training_file):
    with open(training_file, 'r') as f:
        lines = f.readlines()
    print(f"✅ Training Data vorhanden ({len(lines)} Zeilen)")
else:
    print("⚠️  training_data.txt nicht gefunden")

# Prüfe Training Script
train_script = f"{deepall_path}/kaggle_train.py"
if os.path.exists(train_script):
    print(f"✅ kaggle_train.py vorhanden")
else:
    print("❌ kaggle_train.py NICHT gefunden!")

print("\n✅ Setup fertig!")

