# ✅ Kaggle Fine-Tuning Checklist

## 📦 Dateien zum Hochladen

- [x] `deepall/DeepMaster_converted.pt` (124M - das Basis-Modell)
- [x] `deepall/training_data.txt` (kombinierte Trainingsdaten)
- [x] `deepall/kaggle_train.py` (Training Script)
- [x] `deepall/KAGGLE_SETUP.md` (Anleitung)

**Größen:**
```
DeepMaster_converted.pt: ~500 MB
training_data.txt: ~5-10 MB
kaggle_train.py: ~5 KB
```

## 🚀 Kaggle Workflow

### 1. Dataset erstellen
```
https://www.kaggle.com/settings/datasets
→ New Dataset (Private)
→ Upload die 3 Dateien oben
→ Name: "deepmaster"
```

### 2. Notebook erstellen
```
https://www.kaggle.com/code
→ New Notebook
→ Python + GPU (T4 oder P100)
```

### 3. Notebook-Code
```python
# Zelle 1: Setup
!pip install tiktoken torch -q

# Zelle 2: Training
%run /kaggle/input/deepmaster/kaggle_train.py

# Zelle 3: Verify
import torch
ckpt = torch.load('/kaggle/working/DeepMaster_finetuned.pt', weights_only=False)
print("✅ Training erfolgreich!")
```

### 4. Ausführen
- Klick "Run All"
- Warte ~30 Minuten
- Download Output: `DeepMaster_finetuned.pt`

## 📊 Erwartete Metriken

| Metrik | Wert |
|--------|------|
| Training Time | ~30 Min (GPU) |
| Start Loss | ~4.3 |
| Final Loss | ~3.5-4.0 |
| Parameter | 124M (unverändert) |
| Output Size | ~500 MB |

## 🔍 Nach dem Training

1. Download `DeepMaster_finetuned.pt` von Kaggle
2. Speichere in `deepall/DeepMaster_finetuned.pt`
3. Teste lokal:
   ```bash
   python deepall/ask_deepflow.py
   ```
4. Modell sollte jetzt bessere DeepALL-Antworten geben

## ⚠️ Häufige Fehler

| Fehler | Lösung |
|--------|--------|
| CUDA out of memory | batch_size: 32 → 16 |
| Dataset not found | Input hinzufügen (rechts im Notebook) |
| tiktoken error | `!pip install tiktoken` |
| File not found | Pfade prüfen: `/kaggle/input/deepmaster/` |

## 💡 Tipps

- **GPU wählen**: T4 (kostenlos) oder P100 (schneller)
- **Timeout**: Notebook läuft max 9 Stunden
- **Speicher**: Output wird automatisch gespeichert
- **Iterationen**: 1000 ist gut, 500 ist schneller

## 🎯 Nächste Schritte nach Training

1. Modell lokal testen
2. Weitere Fine-Tuning Runden (optional)
3. In Production deployen
4. Feedback sammeln

