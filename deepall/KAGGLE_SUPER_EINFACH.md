# 🚀 KAGGLE - SUPER EINFACH (3 Zellen)

## ⚠️ WICHTIG:
**JEDE Zelle EINZELN kopieren!**
**NICHT vermischen!**
**NICHT mehrere Zellen in eine Zelle kopieren!**

---

## 📌 ZELLE 1: Repository laden

Kopiere DIESEN Code in eine **NEUE ZELLE**:

```python
import os
import shutil

repo_path = "/kaggle/working/nanochat-deepall"
if os.path.exists(repo_path):
    print("🗑️  Lösche altes Repository...")
    shutil.rmtree(repo_path)

print("📥 Clone Repository...")
os.system("cd /kaggle/working && git clone https://github.com/f4t1i/nanochat-deepall.git")

print("📦 Installiere Dependencies...")
os.system("pip install -q torch transformers datasets tqdm numpy pandas scikit-learn")

if os.path.exists(f"{repo_path}/deepall"):
    print("✅ Repository geladen!")
else:
    print("❌ Repository Problem!")
    exit()

deepall_path = f"{repo_path}/deepall"
training_file = f"{deepall_path}/training_data.txt"
if os.path.exists(training_file):
    with open(training_file, 'r') as f:
        lines = f.readlines()
    print(f"✅ Training Data vorhanden ({len(lines)} Zeilen)")

train_script = f"{deepall_path}/kaggle_train.py"
if os.path.exists(train_script):
    print(f"✅ kaggle_train.py vorhanden")

print("\n✅ Setup fertig!")
```

**Ergebnis:** ✅ Setup fertig!

---

## 📌 ZELLE 2: Training starten

Kopiere DIESEN Code in eine **NEUE ZELLE**:

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

**Ergebnis:** 🎯 Training läuft (30 Min)

---

## 📌 ZELLE 3: Ergebnisse prüfen

Kopiere DIESEN Code in eine **NEUE ZELLE**:

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

**Ergebnis:** ✅ Modell trainiert

---

## 💾 DOWNLOAD

1. Rechts: **"Output"** Tab
2. Wähle die `.pt` Dateien
3. Klicke: **"Download"**

---

## ✅ FERTIG!

🎉 Dein Modell ist trainiert!

