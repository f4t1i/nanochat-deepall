# 🚀 DeepMaster Fine-Tuning auf Kaggle

## Schritt 1: Dataset hochladen

1. Gehe zu https://www.kaggle.com/settings/account
2. Erstelle einen neuen **Private Dataset**:
   - Name: `deepmaster`
   - Beschreibung: "DeepMaster Training Data"

3. Lade folgende Dateien hoch:
   ```
   deepall/DeepMaster_converted.pt          (124M)
   deepall/training_data.txt                (alle .txt Dateien kombiniert)
   deepall/kaggle_train.py                  (Training Script)
   ```

## Schritt 2: Kaggle Notebook erstellen

1. Gehe zu https://www.kaggle.com/code
2. Klicke "New Notebook"
3. Wähle **Python** + **GPU** (T4 oder P100)
4. Füge folgende Zellen ein:

### Zelle 1: Setup
```python
!pip install tiktoken torch -q
import os
os.chdir('/kaggle/working')
```

### Zelle 2: Training starten
```python
%run /kaggle/input/deepmaster/kaggle_train.py
```

### Zelle 3: Modell testen
```python
import torch
import tiktoken

# Lade trainiertes Modell
ckpt = torch.load('/kaggle/working/DeepMaster_finetuned.pt', weights_only=False)
print("✅ Modell trainiert!")
print(f"   Parameter: {sum(p.numel() for p in ckpt['model'].values()):,}")
```

## Schritt 3: Notebook ausführen

1. Klicke "Run All"
2. Warte ~30 Minuten (mit GPU)
3. Modell wird in `/kaggle/working/DeepMaster_finetuned.pt` gespeichert

## Schritt 4: Modell herunterladen

1. Nach Training: Klicke "Output" Tab
2. Download: `DeepMaster_finetuned.pt`
3. Speichere in `deepall/DeepMaster_finetuned.pt`

## Schritt 5: Lokal testen

```bash
python deepall/ask_deepflow.py
```

---

## 📊 Erwartete Ergebnisse

- **Training Time**: ~30 Min (mit GPU)
- **Final Loss**: ~3.5-4.0 (von 4.3 Start)
- **Modell-Größe**: 124M Parameter (unverändert)
- **Output**: Bessere DeepALL-spezifische Antworten

## 🔧 Troubleshooting

**Problem**: "CUDA out of memory"
- Lösung: Reduziere `batch_size` von 32 auf 16 in `kaggle_train.py`

**Problem**: "Dataset nicht gefunden"
- Lösung: Stelle sicher, dass Dataset als Input hinzugefügt ist (rechts im Notebook)

**Problem**: "tiktoken nicht installiert"
- Lösung: Füge `!pip install tiktoken` in erste Zelle ein

