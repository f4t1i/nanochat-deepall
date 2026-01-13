# DeepAll – NanoChat Integration

Dieses Verzeichnis bündelt alle Dateien und Skripte, mit denen dein DeepAll-Wissen in NanoChat-Modelle überführt wird.

## Struktur

- `nanogpt-pytorch-deepall-v1/`
  - Rohdaten, Analysen und kombinierte Textdateien
  - `data jsonl/` – alle JSONL-Dateien mit `prompt` + `completion`
  - `deepall 1-5  fr  nano big1.txt` – große kombinierte Textdatei
- `convert_deepall_to_parquet.py`
  - Liest alle `*.jsonl` aus `nanogpt-pytorch-deepall-v1/data jsonl/`
  - Schreibt `base_data/shard_00000.parquet` (NanoChat-kompatibel)
- `create_tokenizer_deepall.py`
  - (Hilfsskript für Tokenizer-Experimente)
- `quick_test_deepall.py`
  - Lädt den NanoChat-Tokenizer und testet ihn auf deinen JSONL-Daten
- `train_deepall_hf.py`
  - HuggingFace-GPT2-Train auf deinen DeepAll-Texten
- `train_simple_deepall.py`
  - Vereinfachte Pipeline:
  - liest JSONL → schreibt `~/.cache/nanochat/deepall_text.txt`
  - optional: trainiert einfachen BPE-Tokenizer
- `train_deepall_simple.py`
  - Einfaches NanoChat-Modell direkt auf JSONL-Texten (Toy-Training)
- `train_deepall_nanochat.py`
  - Training eines simplen GPT-artigen Modells auf `deepall 1-5  fr  nano big1.txt`
- `logs/`
  - Logfiles von DeepAll-Runs (z. B. `training_output.log`)

Alle Skripte verwenden jetzt Pfade relativ zum Repo-Root, d. h. sie funktionieren, solange du sie aus dem Projektordner `/home/deepall/nanochat` heraus startest.

## Wichtige Kommandos

### 1. JSONL → Parquet für NanoChat

- Vorbereitung für NanoChat-`base_train`:

```bash
python deepall/convert_deepall_to_parquet.py
```

Ergebnis: `base_data/shard_00000.parquet`.

### 2. Einfache Datenaufbereitung + kleines Base-Training

- Kombiniert JSONL und bereitet Text vor:

```bash
python deepall/train_simple_deepall.py
```

- Komplettes, kleines CPU-Training mit NanoChat (inkl. `base_train`):

```bash
./train_deepall_final.sh
```

### 3. Vollständiges CPU-Pipeline-Skript (JSONL → Tokenizer → Base → SFT)

```bash
./train_deepall_cpu.sh
```

Dieses Skript:
- setzt das Python-Env via `uv` auf
- kombiniert alle JSONL-Dateien
- trainiert den NanoChat-Tokenizer
- trainiert ein kleines Basismodell
- führt SFT auf deinen DeepAll-Daten durch
- generiert anschließend einen NanoChat-Report

### 4. HuggingFace-Training

```bash
python deepall/train_deepall_hf.py
```

Trainiert ein kleines GPT2-ähnliches Modell (HF-Transformers) direkt auf deinen DeepAll-Texten.

### 5. NanoChat-Training auf großer TXT-Datei

```bash
python deepall/train_deepall_nanochat.py
```

Verwendet `deepall 1-5  fr  nano big1.txt` und trainiert ein simples GPT-artiges Modell (reines PyTorch, ohne NanoChat-Dataloader).

## Hinweis

- Die DeepAll-Pfade sind nicht mehr hart auf `/home/deepall/...` verdrahtet, sondern werden relativ zu diesem Repo aufgelöst.
- Wenn du das Repo verschiebst, funktioniert alles weiter, solange die Struktur unter `deepall/` gleich bleibt.

