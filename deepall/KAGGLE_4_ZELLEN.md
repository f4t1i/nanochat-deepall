# 🚀 KAGGLE NOTEBOOK - 4 SAUBERE ZELLEN

## ⚠️ WICHTIG ZUERST:
1. Erstelle Kaggle Notebook mit **GPU**
2. Wähle dein **"deepmaster" Dataset als INPUT** (rechts: "+ Add input")
3. Kopiere JEDE Zelle EINZELN (nicht vermischen!)

---

## 📌 ZELLE 1: Repository laden

**Kopiere DIESEN Code in eine neue Zelle:**

```python
# Setup und Repository laden
!apt-get update -qq && apt-get install -y git > /dev/null 2>&1
!cd /kaggle/working && git clone https://github.com/f4t1i/nanochat-deepall.git
!pip install -q torch transformers datasets tqdm numpy pandas scikit-learn

import os
print("✅ Repository geladen!")
print("📁 Dateien:")
os.system("ls -la /kaggle/working/nanochat-deepall/deepall/ | head -15")
```

**Ergebnis:** ✅ Repository ist geladen

---

## 📌 ZELLE 2: Prüfe Dataset

**Kopiere DIESEN Code in eine NEUE Zelle:**

```python
import os

input_path = "/kaggle/input"
print("📂 Input Verzeichnis:")
if os.path.exists(input_path):
    for item in os.listdir(input_path):
        print(f"  ✅ {item}")
else:
    print("  ❌ Kein Input - Dataset MUSS hinzugefügt werden!")

repo_path = "/kaggle/working/nanochat-deepall"
training_file = f"{repo_path}/deepall/training_data.txt"
print(f"\n📄 Training Datei existiert: {os.path.exists(training_file)}")

model_file = f"{repo_path}/deepall/DeepMaster_converted.pt"
print(f"🤖 Modell vorhanden: {os.path.exists(model_file)}")

train_script = f"{repo_path}/deepall/kaggle_train.py"
print(f"📜 Training Script vorhanden: {os.path.exists(train_script)}")
```

**Ergebnis:** ✅ Alle Dateien sind vorhanden

---

## 📌 ZELLE 3: Training starten

**Kopiere DIESEN Code in eine NEUE Zelle:**

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

## 📌 ZELLE 4: Ergebnisse prüfen

**Kopiere DIESEN Code in eine NEUE Zelle:**

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

**Ergebnis:** ✅ Modell ist trainiert

---

## 💾 DOWNLOAD

1. Rechts: **"Output"** Tab
2. Wähle die `.pt` Dateien
3. Klicke: **"Download"**

---

## ✅ FERTIG!

Dein Modell ist trainiert und bereit zum Download! 🎉

