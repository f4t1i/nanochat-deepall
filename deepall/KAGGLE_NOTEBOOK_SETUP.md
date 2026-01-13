# 🚀 Kaggle Notebook Setup - Komplette Anleitung

## 📋 Schritt 1: Notebook erstellen

1. Gehe zu: https://www.kaggle.com/code
2. Klicke: **"New Notebook"**
3. Wähle: **GPU** (rechts oben)
4. Speichern

---

## 📥 Schritt 2: Dataset hinzufügen (WICHTIG!)

1. Rechts: **"Input"** Tab
2. Klicke: **"+ Add input"**
3. Suche: **"deepmaster"** (dein Dataset)
4. Wähle es aus
5. Klicke: **"Add"**

**⚠️ OHNE DIESEN SCHRITT FUNKTIONIERT NICHTS!**

---

## 💻 Schritt 3: Zelle 1 - Repository laden

Kopiere diesen Code in die **ERSTE ZELLE**:

```bash
# Setup und Repository laden
!apt-get update -qq && apt-get install -y git > /dev/null 2>&1

# Clone Repository
!cd /kaggle/working && git clone https://github.com/f4t1i/nanochat-deepall.git

# Installiere Dependencies
!pip install -q torch transformers datasets tqdm numpy pandas scikit-learn

# Prüfe Setup
import os
print("✅ Repository geladen!")
print("📁 Dateien:")
os.system("ls -la /kaggle/working/nanochat-deepall/deepall/ | head -15")
```

---

## 🔍 Schritt 4: Zelle 2 - Prüfe Dataset

```python
import os

# Prüfe Input
input_path = "/kaggle/input"
print("📂 Input Verzeichnis:")
if os.path.exists(input_path):
    for item in os.listdir(input_path):
        print(f"  ✅ {item}")
        dataset_path = os.path.join(input_path, item)
        if os.path.isdir(dataset_path):
            files = os.listdir(dataset_path)
            print(f"     Dateien: {len(files)}")
            for f in files[:5]:
                print(f"       - {f}")
else:
    print("  ❌ Kein Input Verzeichnis!")

# Prüfe Repository
repo_path = "/kaggle/working/nanochat-deepall"
print(f"\n📁 Repository: {repo_path}")
print(f"  Existiert: {os.path.exists(repo_path)}")

# Prüfe Training Dateien
training_file = f"{repo_path}/deepall/training_data.txt"
print(f"\n📄 Training Datei: {training_file}")
print(f"  Existiert: {os.path.exists(training_file)}")
if os.path.exists(training_file):
    with open(training_file, 'r') as f:
        lines = f.readlines()
    print(f"  Zeilen: {len(lines)}")
    print(f"  Erste Zeile: {lines[0][:100]}...")
```

---

## 🎯 Schritt 5: Zelle 3 - Training starten

```python
import os
import sys

# Setze Pfade
repo_path = "/kaggle/working/nanochat-deepall"
sys.path.insert(0, repo_path)

# Wechsle in Verzeichnis
os.chdir(f"{repo_path}/deepall")

# Starte Training
print("🚀 Starte Training...")
print("=" * 60)

os.system("python kaggle_train.py")
```

---

## 📊 Schritt 6: Zelle 4 - Ergebnisse prüfen

```python
import os

# Prüfe Output
output_path = "/kaggle/working/nanochat-deepall/deepall"
print("📁 Output Dateien:")
for f in os.listdir(output_path):
    if f.endswith(('.pt', '.pth', '.txt', '.log')):
        full_path = os.path.join(output_path, f)
        size = os.path.getsize(full_path) / (1024*1024)  # MB
        print(f"  ✅ {f} ({size:.2f} MB)")

# Prüfe Modell
model_file = f"{output_path}/DeepMaster_finetuned.pt"
if os.path.exists(model_file):
    print(f"\n✅ Modell erfolgreich trainiert!")
    print(f"   Größe: {os.path.getsize(model_file) / (1024*1024):.2f} MB")
else:
    print(f"\n⚠️  Modell nicht gefunden")
```

---

## 💾 Schritt 7: Download

1. Rechts: **"Output"** Tab
2. Wähle die Dateien
3. Klicke: **"Download"**

---

## 🔧 Troubleshooting

### Problem: "Dataset nicht gefunden"
**Lösung**: Prüfe Schritt 2 - Dataset MUSS als INPUT hinzugefügt sein!

### Problem: "Module nicht gefunden"
**Lösung**: Führe Zelle 1 nochmal aus

### Problem: "Out of Memory"
**Lösung**: Reduziere `batch_size` in `kaggle_train.py`

### Problem: "GPU nicht verfügbar"
**Lösung**: Prüfe ob GPU aktiviert ist (rechts oben im Notebook)

---

## ✅ Fertig!

Dein Modell wird trainiert und du kannst es downloaden! 🎉

