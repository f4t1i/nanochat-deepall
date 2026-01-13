# 🚀 KAGGLE NOTEBOOK - OHNE DATASET INPUT!

## ✅ Das Dataset wird aus dem Repo gezogen!

Du brauchst **KEIN** separates Dataset als INPUT hinzufügen!
Das Training Data ist bereits im Repository!

---

## 📌 ZELLE 1: Repository laden + Dataset vorbereiten

```python
# Setup und Repository laden
!apt-get update -qq && apt-get install -y git > /dev/null 2>&1
!cd /kaggle/working && git clone https://github.com/f4t1i/nanochat-deepall.git

# Installiere Dependencies
!pip install -q torch transformers datasets tqdm numpy pandas scikit-learn

import os
print("✅ Repository geladen!")

# Prüfe Training Data
repo_path = "/kaggle/working/nanochat-deepall"
training_file = f"{repo_path}/deepall/training_data.txt"

if os.path.exists(training_file):
    with open(training_file, 'r') as f:
        lines = f.readlines()
    print(f"✅ Training Data gefunden: {len(lines)} Zeilen")
    print(f"   Erste Zeile: {lines[0][:100]}...")
else:
    print("❌ Training Data nicht gefunden!")

# Prüfe Modell
model_file = f"{repo_path}/deepall/DeepMaster_converted.pt"
if os.path.exists(model_file):
    size = os.path.getsize(model_file) / (1024*1024)
    print(f"✅ Modell gefunden: {size:.2f} MB")
else:
    print("❌ Modell nicht gefunden!")
```

---

## 📌 ZELLE 2: Training starten

```python
import os
import sys

repo_path = "/kaggle/working/nanochat-deepall"
sys.path.insert(0, repo_path)
os.chdir(f"{repo_path}/deepall")

print("🚀 Starte Training mit Repository Data...")
print("=" * 60)

# Training starten
os.system("python kaggle_train.py")
```

---

## 📌 ZELLE 3: Ergebnisse prüfen

```python
import os

output_path = "/kaggle/working/nanochat-deepall/deepall"
print("📁 Output Dateien:")
for f in os.listdir(output_path):
    if f.endswith(('.pt', '.pth', '.txt', '.log')):
        full_path = os.path.join(output_path, f)
        size = os.path.getsize(full_path) / (1024*1024)
        print(f"  ✅ {f} ({size:.2f} MB)")

model_file = f"{output_path}/DeepMaster_finetuned.pt"
if os.path.exists(model_file):
    print(f"\n✅ Modell erfolgreich trainiert!")
else:
    print(f"\n⚠️  Modell nicht gefunden")
```

---

## 💾 DOWNLOAD

1. Rechts: **"Output"** Tab
2. Wähle die `.pt` Dateien
3. Klicke: **"Download"**

---

## ✅ FERTIG!

**Keine separaten Inputs nötig!**
Alles kommt aus dem Repository! 🎉

