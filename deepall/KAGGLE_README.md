# 🚀 DeepMaster Fine-Tuning auf Kaggle

## 📋 Übersicht

Dieses Paket enthält alles, was du brauchst, um das **DeepMaster GPT-2 Modell** auf Kaggle mit GPU zu trainieren.

**Was ist enthalten:**
- ✅ `DeepMaster_converted.pt` - 124M Parameter GPT-2 Basis-Modell
- ✅ `training_data.txt` - Kombinierte DeepALL Trainingsdaten (~1.2M Tokens)
- ✅ `kaggle_train.py` - Optimiertes Training-Script für GPU
- ✅ `KAGGLE_SETUP.md` - Detaillierte Anleitung
- ✅ `KAGGLE_QUICK_START.txt` - Schnelle Schritt-für-Schritt Anleitung

## 🎯 Ziel

Das Modell von einem **generischen GPT-2** zu einem **DeepALL-spezialisierten Modell** trainieren, das:
- DeepFlow (M005) versteht
- Ursachen-Wirkungs-Ketten analysiert
- DeepALL-Konzepte korrekt reproduziert

## ⏱️ Zeitaufwand

| Schritt | Zeit |
|---------|------|
| Dataset hochladen | 5 Min |
| Notebook erstellen | 2 Min |
| Code eingeben | 5 Min |
| Training (GPU) | 30 Min |
| Download | 2 Min |
| **TOTAL** | **~45 Min** |

## 🚀 Quick Start

1. **Lese**: `KAGGLE_QUICK_START.txt`
2. **Folge**: Den 6 Schritten
3. **Warte**: ~30 Minuten
4. **Download**: Trainiertes Modell
5. **Teste**: `python deepall/ask_deepflow.py`

## 📊 Technische Details

**Modell-Architektur:**
```
GPT-2 (124M Parameter)
- 12 Transformer Blöcke
- 12 Attention Heads
- 768 Hidden Dimension
- 1024 Context Length
```

**Training-Konfiguration:**
```
Batch Size: 32 (GPU) / 4 (CPU)
Learning Rate: 3e-4
Iterations: 1000
Eval Interval: 100
Optimizer: AdamW
```

**Daten:**
```
Trainingsdaten: ~1.2M Tokens
Train/Val Split: 90/10
Tokenizer: GPT-2 (tiktoken)
```

## 🔧 Troubleshooting

**Q: "CUDA out of memory"**
A: Reduziere `batch_size` von 32 auf 16 in `kaggle_train.py`

**Q: "Dataset not found"**
A: Stelle sicher, dass du das Dataset als Input hinzugefügt hast (rechts im Notebook)

**Q: "Notebook timeout"**
A: Reduziere `max_iters` von 1000 auf 500

**Q: "tiktoken not found"**
A: Füge `!pip install tiktoken` in erste Zelle ein

## 📈 Erwartete Ergebnisse

Nach dem Training sollte das Modell:
- ✅ Bessere Perplexity auf DeepALL-Daten haben
- ✅ DeepFlow-spezifische Fragen besser beantworten
- ✅ Weniger generischen Text produzieren
- ✅ Ursachen-Wirkungs-Ketten erkennen

**Beispiel vorher:**
```
Prompt: "Was ist DeepFlow?"
Output: "Why is deep deepflow a critical problem in the energy..."
```

**Beispiel nachher (erwartet):**
```
Prompt: "Was ist DeepFlow?"
Output: "DeepFlow ist ein Modul (M005) das Muster in 
Entscheidungsprozessen analysiert und Ursachen-Wirkungs-Ketten abbildet..."
```

## 🎓 Nächste Schritte

1. **Training abschließen** auf Kaggle
2. **Modell testen** lokal
3. **Weitere Runden** (optional) mit mehr Daten
4. **In Production** deployen
5. **Feedback** sammeln und iterieren

## 📞 Support

Falls Probleme auftreten:
1. Prüfe `KAGGLE_SETUP.md` für detaillierte Anleitung
2. Prüfe Kaggle Notebook Logs
3. Versuche mit kleinerer `batch_size`
4. Reduziere `max_iters` für schnelleres Testen

---

**Status**: ✅ Bereit für Kaggle
**Letzte Aktualisierung**: 2026-01-13
**Modell**: DeepMaster (GPT-2 124M)

