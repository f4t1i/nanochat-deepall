#!/usr/bin/env python3
"""
Verifiziere dass alle Kaggle-Dateien vorhanden und korrekt sind
"""
import os
from pathlib import Path

print("=" * 60)
print("✅ Kaggle Files Verification")
print("=" * 60)

base_dir = Path(__file__).parent
files_to_check = {
    'DeepMaster_converted.pt': 'Basis-Modell (124M Parameter)',
    'training_data.txt': 'Kombinierte Trainingsdaten',
    'kaggle_train.py': 'Training Script für Kaggle',
    'KAGGLE_SETUP.md': 'Detaillierte Anleitung',
    'KAGGLE_CHECKLIST.md': 'Checklist',
    'KAGGLE_QUICK_START.txt': 'Quick Start Guide',
    'KAGGLE_README.md': 'README',
}

print("\n📦 Dateien zum Hochladen:\n")

all_ok = True
for filename, description in files_to_check.items():
    filepath = base_dir / filename
    if filepath.exists():
        size = filepath.stat().st_size
        size_mb = size / (1024 * 1024)
        print(f"  ✅ {filename}")
        print(f"     → {description}")
        print(f"     → Größe: {size_mb:.1f} MB\n")
    else:
        print(f"  ❌ {filename} - NICHT GEFUNDEN!")
        all_ok = False

print("=" * 60)

if all_ok:
    print("✅ ALLE DATEIEN VORHANDEN!")
    print("\n🚀 Bereit für Kaggle Upload:")
    print("   1. Gehe zu: https://www.kaggle.com/settings/datasets")
    print("   2. Erstelle neues Dataset: 'deepmaster'")
    print("   3. Lade diese Dateien hoch:")
    for filename in ['DeepMaster_converted.pt', 'training_data.txt', 'kaggle_train.py']:
        print(f"      - {filename}")
    print("\n   4. Erstelle Notebook mit GPU")
    print("   5. Folge KAGGLE_QUICK_START.txt")
else:
    print("❌ FEHLER: Einige Dateien fehlen!")
    print("   Bitte stelle sicher, dass alle Dateien im deepall/ Ordner sind")

print("=" * 60)

