#!/usr/bin/env python3
# ============================================================================
# KAGGLE NOTEBOOK - ZELLE 3: Ergebnisse prüfen
# ============================================================================
# Kopiere DIESEN KOMPLETTEN Code in eine NEUE Zelle im Kaggle Notebook
# NICHT vermischen mit anderen Zellen!
# ============================================================================

import os

output_path = "/kaggle/working/nanochat-deepall/deepall"
print("📁 Output Dateien:")

for f in os.listdir(output_path):
    if f.endswith(('.pt', '.pth')):
        full_path = os.path.join(output_path, f)
        size = os.path.getsize(full_path) / (1024*1024)
        print(f"  ✅ {f} ({size:.0f} MB)")

print("\n✅ Training fertig!")

