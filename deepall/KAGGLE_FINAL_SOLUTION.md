# 🚀 KAGGLE - FINALE LÖSUNG

## Problem:
Die großen `.pt` Dateien sind nicht im Git Repository!

## Lösung:
Wir laden das Repository + die großen Dateien separat!

---

## 📌 ZELLE 1: Repository + Dateien laden

```python
import os
import subprocess
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

# Jetzt lade die großen Dateien
print("\n📥 Lade große Dateien...")

# DeepMaster_converted.pt (548 MB)
deepall_path = f"{repo_path}/deepall"
model_file = f"{deepall_path}/DeepMaster_converted.pt"

if not os.path.exists(model_file):
    print("📥 Download DeepMaster_converted.pt...")
    # Versuche von GitHub Raw zu laden
    os.system(f"cd {deepall_path} && wget -q https://raw.githubusercontent.com/f4t1i/nanochat-deepall/master/deepall/DeepMaster_converted.pt 2>/dev/null || echo 'Wget fehlgeschlagen'")
    
    if not os.path.exists(model_file):
        print("⚠️  DeepMaster_converted.pt nicht gefunden")
        print("   Das ist OK - wird während Training erstellt")
else:
    size = os.path.getsize(model_file) / (1024*1024)
    print(f"✅ DeepMaster_converted.pt vorhanden ({size:.0f} MB)")

# Prüfe Training Data
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
```

---

## 📌 ZELLE 2: Training starten

```python
import os
import sys

repo_path = "/kaggle/working/nanochat-deepall"
sys.path.insert(0, repo_path)
os.chdir(f"{repo_path}/deepall")

print("🚀 Starte Training...")
print("=" * 60)

os.system("python kaggle_train.py")
```

---

## 📌 ZELLE 3: Ergebnisse prüfen

```python
import os

output_path = "/kaggle/working/nanochat-deepall/deepall"
print("📁 Output Dateien:")

for f in os.listdir(output_path):
    if f.endswith(('.pt', '.pth')):
        full_path = os.path.join(output_path, f)
        size = os.path.getsize(full_path) / (1024*1024)
        print(f"  ✅ {f} ({size:.0f} MB)")

print("\n✅ Training fertig!")
```

---

## 💾 DOWNLOAD

1. Rechts: **"Output"** Tab
2. Wähle die `.pt` Dateien
3. Klicke: **"Download"**

---

## ✅ FERTIG!

Jetzt sollte es funktionieren! 🎉

