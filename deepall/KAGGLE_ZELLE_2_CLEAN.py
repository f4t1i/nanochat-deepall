#!/usr/bin/env python3
# ============================================================================
# KAGGLE NOTEBOOK - ZELLE 2: Training starten
# ============================================================================
# Kopiere DIESEN KOMPLETTEN Code in eine NEUE Zelle im Kaggle Notebook
# NICHT vermischen mit anderen Zellen!
# ============================================================================

import os
import sys

repo_path = "/kaggle/working/nanochat-deepall"
sys.path.insert(0, repo_path)
os.chdir(f"{repo_path}/deepall")

print("🚀 Starte Training...")
print("=" * 60)

os.system("python kaggle_train.py")

