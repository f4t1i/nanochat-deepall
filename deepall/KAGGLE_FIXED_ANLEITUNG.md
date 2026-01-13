# 🚀 Kaggle Anleitung - FIXED VERSION

## ⚠️ WICHTIG: Dataset als INPUT hinzufügen!

Das ist der häufigste Fehler! Wenn du das nicht machst, funktioniert nichts!

---

## **SCHRITT 1: Dataset hochladen (5 Min)**

### 1.1 Gehe zu Datasets
```
https://www.kaggle.com/settings/datasets
```

### 1.2 Klick "+ New Dataset"

### 1.3 Upload 3 Dateien
```
✅ DeepMaster_converted.pt    (500 MB)
✅ training_data.txt           (5 MB)
✅ kaggle_train.py             (5 KB)
```

**Wo findest du die Dateien?**
- GitHub: https://github.com/f4t1i/nanochat-deepall/tree/kaggle-deepall/deepall
- Oder lokal: `/home/deepall/nanochat/deepall/`

### 1.4 Dataset-Details
- **Name**: `deepmaster`
- **Beschreibung**: "DeepMaster Fine-Tuning Dataset"
- Klick "Create"

### 1.5 Warte auf Upload
- Status sollte "Uploaded" sein

---

## **SCHRITT 2: Notebook erstellen (2 Min)**

### 2.1 Gehe zu Code
```
https://www.kaggle.com/code
```

### 2.2 Klick "+ New Notebook"

### 2.3 Wähle Einstellungen
- **Language**: Python
- **Accelerator**: GPU (T4 oder P100)

### 2.4 Klick "Create"

---

## **SCHRITT 3: Dataset hinzufügen (WICHTIG!)**

### 3.1 Rechts im Notebook
- Klick "Input" (rechts oben)
- Klick "+ Add input"
- Suche: "deepmaster"
- Wähle dein Dataset
- Klick "Add"

### 3.2 Überprüfe
- Rechts sollte dein Dataset angezeigt werden
- Path: `/kaggle/input/deepmaster/`

**WENN DAS NICHT FUNKTIONIERT:**
- Gehe zurück zu Schritt 1
- Stelle sicher, dass Dataset hochgeladen ist
- Versuche erneut

---

## **SCHRITT 4: Code eingeben (5 Min)**

### 4.1 Zelle 1: Setup
```python
import subprocess
import sys
import torch

print("📦 Installiere Dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tiktoken", "torch"])
print("✅ Dependencies installiert")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Device: {device}")
```

### 4.2 Zelle 2: Prüfe Dataset
```python
from pathlib import Path

input_dir = Path('/kaggle/input/deepmaster')
print(f"📁 Input Directory: {input_dir}")
print(f"   Existiert: {input_dir.exists()}")

if input_dir.exists():
    print("\n   Dateien:")
    for f in sorted(input_dir.iterdir()):
        if f.is_file():
            size = f.stat().st_size / (1024**2)
            print(f"   ✅ {f.name} ({size:.1f} MB)")
else:
    print("   ❌ FEHLER: Dataset nicht gefunden!")
    print("   Hast du das Dataset als INPUT hinzugefügt?")
```

### 4.3 Zelle 3: Training
```python
# Führe Training Script aus
exec(open('/kaggle/input/deepmaster/kaggle_train.py').read())
```

### 4.4 Zelle 4: Verify
```python
import torch
from pathlib import Path

output_file = Path('/kaggle/working/DeepMaster_finetuned.pt')
print(f"📂 Output: {output_file}")
print(f"   Existiert: {output_file.exists()}")

if output_file.exists():
    size = output_file.stat().st_size / (1024**2)
    print(f"   ✅ Größe: {size:.1f} MB")
    ckpt = torch.load(output_file, map_location='cpu', weights_only=False)
    print(f"   ✅ Kann geladen werden")
    print(f"\n✅ Training erfolgreich!")
else:
    print(f"   ❌ FEHLER: Output nicht gefunden!")
```

---

## **SCHRITT 5: Ausführen (30 Min)**

### 5.1 Klick "Run All"
- Oben: "Run All" oder Ctrl+Shift+Enter

### 5.2 Warte auf Completion
- Zelle 1: Dependencies (~2 Min)
- Zelle 2: Dataset Check (~1 Min)
- Zelle 3: Training (~25 Min)
- Zelle 4: Verify (~1 Min)

### 5.3 Überwache Logs
- Loss sollte sinken
- Keine Errors!

---

## **SCHRITT 6: Download (2 Min)**

### 6.1 Nach Training
- Gehe zu "Output" Tab (rechts)
- Sollte `DeepMaster_finetuned.pt` angezeigt werden

### 6.2 Download
- Klick auf die Datei
- Klick "Download"
- Speichere in: `/home/deepall/nanochat/deepall/`

---

## **SCHRITT 7: Lokal testen (1 Min)**

```bash
cd /home/deepall/nanochat
python deepall/ask_deepflow.py
```

---

## 🆘 **TROUBLESHOOTING**

### Problem: "Dataset nicht gefunden"
**Lösung:**
1. Gehe zu Schritt 3
2. Prüfe ob Dataset als INPUT hinzugefügt
3. Path sollte: `/kaggle/input/deepmaster/`

### Problem: "CUDA OOM"
**Lösung:**
1. Ändere in kaggle_train.py: `batch_size: 32 → 16`
2. Oder wechsle zu T4 GPU

### Problem: "tiktoken error"
**Lösung:**
```python
!pip install --upgrade tiktoken
```

### Problem: "Timeout"
**Lösung:**
1. Ändere in kaggle_train.py: `max_iters: 1000 → 500`
2. Oder nutze P100 GPU

---

## ✅ **CHECKLISTE**

- [ ] Dataset hochgeladen
- [ ] Notebook erstellt
- [ ] GPU aktiviert
- [ ] **Dataset als INPUT hinzugefügt** ← WICHTIG!
- [ ] Code eingegeben
- [ ] "Run All" geklickt
- [ ] Training abgewartet
- [ ] Modell heruntergeladen
- [ ] Lokal getestet

---

**Status**: 🚀 BEREIT!

**Nächster Schritt**: Gehe zu https://www.kaggle.com/settings/datasets

