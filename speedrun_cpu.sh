#!/bin/bash

# CPU-optimized training script for nanochat
# This trains a TINY model that can actually finish on CPU in reasonable time
# Expected runtime: ~2-4 hours on 32-core CPU

echo "=========================================="
echo "NanoChat CPU Training (Tiny Model)"
echo "=========================================="
echo ""
echo "⚠️  CPU Training ist SEHR langsam!"
echo "⚠️  Wir trainieren ein SEHR kleines Modell (d4, ~20M params)"
echo "⚠️  Erwartete Zeit: 2-4 Stunden"
echo ""
read -p "Fortfahren? (ja/nein): " confirm
if [ "$confirm" != "ja" ]; then
    echo "Abgebrochen."
    exit 1
fi

# Setup
export OMP_NUM_THREADS=32  # Nutze alle CPU Kerne
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

echo ""
echo "Schritt 1: Python Environment Setup..."
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv

# Install CPU version (no GPU dependencies)
echo "Installiere Dependencies (CPU-only)..."
uv pip install --python .venv/bin/python torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
uv sync

source .venv/bin/activate

# Verify CPU-only PyTorch
echo ""
echo "Prüfe PyTorch Installation..."
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CPU cores: {torch.get_num_threads()}')"

# -----------------------------------------------------------------------------
# Report setup
echo ""
echo "Schritt 2: Report Setup..."
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer (minimal dataset)

echo ""
echo "Schritt 3: Tokenizer Training..."
echo "Lade minimalen Datensatz (2 shards = ~500MB)..."
python -m nanochat.dataset -n 2

# Train tokenizer on minimal data
echo "Trainiere Tokenizer (kleines Vokabular für schnelleres Training)..."
python -m scripts.tok_train --max_chars=500000000 --vocab_size=16384

# Evaluate tokenizer
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (TINY model for CPU)

echo ""
echo "Schritt 4: Base Model Training (d4, ~20M params)..."
echo "⚠️  Dies dauert 2-4 Stunden auf CPU..."

# Download a bit more data in background
python -m nanochat.dataset -n 10 &
DATASET_DOWNLOAD_PID=$!

echo "Warte auf Datensatz-Download..."
wait $DATASET_DOWNLOAD_PID

# Train TINY model
# d4 = 4 layers, ~20M parameters
# Very small batches and context to fit in CPU memory
# Short training (100 iterations) just to test the pipeline
python -m scripts.base_train \
    --depth=4 \
    --aspect_ratio=64 \
    --max_seq_len=256 \
    --device_batch_size=1 \
    --total_batch_size=256 \
    --num_iterations=100 \
    --eval_every=25 \
    --eval_tokens=10000 \
    --core_metric_every=-1 \
    --sample_every=50 \
    --device_type=cpu

echo ""
echo "✓ Base Training abgeschlossen!"

# Evaluate (skip CORE benchmark, it's too slow on CPU)
echo ""
echo "Schritt 5: Evaluation..."
python -m scripts.base_loss \
    --device_batch_size=1 \
    --split_tokens=10000 \
    --device_type=cpu

# Skip base_eval (CORE benchmark) - too slow on CPU
echo "⚠️  Überspringe CORE Benchmark (zu langsam auf CPU)"

# -----------------------------------------------------------------------------
# Midtraining (minimal)

echo ""
echo "Schritt 6: Midtraining..."
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

python -m scripts.mid_train \
    --device_batch_size=1 \
    --total_batch_size=256 \
    --num_iterations=50 \
    --eval_every=25 \
    --device_type=cpu

# Skip chat_eval - too slow
echo "⚠️  Überspringe Chat Evaluation (zu langsam auf CPU)"

# -----------------------------------------------------------------------------
# SFT (minimal)

echo ""
echo "Schritt 7: Supervised Fine-tuning..."
python -m scripts.chat_sft \
    --device_batch_size=1 \
    --target_examples_per_step=4 \
    --num_epochs=1 \
    --eval_every=50 \
    --eval_metrics_every=-1 \
    --device_type=cpu

# -----------------------------------------------------------------------------
# Generate report
echo ""
echo "Schritt 8: Generiere Report..."
python -m nanochat.report generate

echo ""
echo "=========================================="
echo "✓ CPU Training abgeschlossen!"
echo "=========================================="
echo ""
echo "Dein Modell ist fertig (sehr klein, aber funktional)!"
echo ""
echo "Teste es:"
echo "  python -m scripts.chat_cli -p 'Hallo, wer bist du?' --device-type=cpu"
echo ""
echo "Oder starte Web UI:"
echo "  python -m scripts.chat_web --device-type=cpu"
echo ""
echo "⚠️  Hinweis: Das Modell ist SEHR klein (d4, ~20M params)"
echo "⚠️  Für bessere Qualität: GPU verwenden oder länger trainieren"
echo "=========================================="

