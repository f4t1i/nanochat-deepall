#!/bin/bash

# DeepAll Custom Training Script (CPU)
# Trainiert ein NanoChat Modell mit deinen DeepAll Daten
# Runtime: ~1-2 Stunden auf 32-core CPU

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/deepall/nanogpt-pytorch-deepall-v1/data jsonl"

echo "=========================================="
echo "DeepAll NanoChat Training (CPU)"
echo "=========================================="
echo ""
echo "📊 Daten: $DATA_DIR/"
echo "📈 Format: JSONL (prompt + completion)"
echo "⏱️  Erwartete Zeit: 1-2 Stunden"
echo ""

# Setup
export OMP_NUM_THREADS=32
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# Check if data exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Daten nicht gefunden: $DATA_DIR"
    exit 1
fi

echo "✓ Daten gefunden"
echo ""

# Python venv setup
echo "Schritt 1: Python Environment..."
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv

# Install CPU PyTorch
echo "Installiere PyTorch (CPU)..."
uv pip install --python .venv/bin/python torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
uv sync

source .venv/bin/activate

# Verify
python -c "import torch; print(f'✓ PyTorch {torch.__version__} (CPU mode)')"

# Reset report
echo ""
echo "Schritt 2: Report Setup..."
python -m nanochat.report reset

# Prepare training data
echo ""
echo "Schritt 3: Vorbereitung Trainingsdaten..."

# Combine all JSONL files into one
COMBINED_DATA="$NANOCHAT_BASE_DIR/deepall_combined.jsonl"
cat "$DATA_DIR"/*.jsonl > "$COMBINED_DATA"

echo "✓ Kombinierte Daten: $COMBINED_DATA"
wc -l "$COMBINED_DATA"

# Convert JSONL to plain text for tokenizer
# Format: prompt + completion on one line
PLAIN_TEXT="$NANOCHAT_BASE_DIR/deepall_text.txt"
python << 'EOF'
import json
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file, 'r', encoding='utf-8') as f_in:
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            try:
                data = json.loads(line)
                prompt = data.get('prompt', '')
                completion = data.get('completion', '')
                combined = prompt + completion
                f_out.write(combined + '\n')
            except:
                pass

print(f"✓ Konvertiert zu: {output_file}")
EOF
python -c "
import json
import sys

input_file = '$COMBINED_DATA'
output_file = '$PLAIN_TEXT'

with open(input_file, 'r', encoding='utf-8') as f_in:
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            try:
                data = json.loads(line)
                prompt = data.get('prompt', '')
                completion = data.get('completion', '')
                combined = prompt + completion
                f_out.write(combined + '\n')
            except:
                pass

print(f'✓ Konvertiert zu: {output_file}')
"

# Train tokenizer on your data
echo ""
echo "Schritt 4: Tokenizer Training..."
python -m scripts.tok_train \
    --input_file "$PLAIN_TEXT" \
    --vocab_size=8192

python -m scripts.tok_eval

# Base model training (tiny, but with your data)
echo ""
echo "Schritt 5: Base Model Training..."
echo "⏱️  Dies dauert 1-2 Stunden..."

python -m scripts.base_train \
    --depth=4 \
    --aspect_ratio=64 \
    --max_seq_len=256 \
    --device_batch_size=1 \
    --total_batch_size=256 \
    --num_iterations=200 \
    --eval_every=50 \
    --eval_tokens=5000 \
    --sample_every=100 \
    --device_type=cpu

echo "✓ Base Training abgeschlossen!"

# Evaluation
echo ""
echo "Schritt 6: Evaluation..."
python -m scripts.base_loss \
    --device_batch_size=1 \
    --split_tokens=5000 \
    --device_type=cpu

# SFT mit deinen Daten
echo ""
echo "Schritt 7: Supervised Fine-tuning mit DeepAll Daten..."

# Convert JSONL to SFT format
SFT_DATA="$NANOCHAT_BASE_DIR/deepall_sft.jsonl"
python -c "
import json

input_file = '$COMBINED_DATA'
output_file = '$SFT_DATA'

with open(input_file, 'r', encoding='utf-8') as f_in:
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            try:
                data = json.loads(line)
                # Convert to SFT format
                sft_item = {
                    'messages': [
                        {'role': 'user', 'content': data.get('prompt', '')},
                        {'role': 'assistant', 'content': data.get('completion', '')}
                    ]
                }
                f_out.write(json.dumps(sft_item) + '\n')
            except:
                pass

print(f'✓ SFT Daten: {output_file}')
"

python -m scripts.chat_sft \
    --device_batch_size=1 \
    --target_examples_per_step=2 \
    --num_epochs=2 \
    --eval_every=25 \
    --device_type=cpu

# Generate report
echo ""
echo "Schritt 8: Generiere Report..."
python -m nanochat.report generate

echo ""
echo "=========================================="
echo "✓ Training abgeschlossen!"
echo "=========================================="
echo ""
echo "Dein DeepAll-Modell ist fertig!"
echo ""
echo "Teste es:"
echo "  python -m scripts.chat_cli -p 'Was ist DeepAll?' --device-type=cpu"
echo ""
echo "Oder Web UI:"
echo "  python -m scripts.chat_web --device-type=cpu"
echo ""
echo "=========================================="

