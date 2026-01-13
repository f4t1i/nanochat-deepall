#!/bin/bash

# Final DeepAll Training Script
# Trainiert mit deinen Daten auf CPU

set -e  # Exit on error

echo "=========================================="
echo "DeepAll NanoChat Training (Final)"
echo "=========================================="
echo ""

# Activate venv
source .venv/bin/activate

# Setup
export OMP_NUM_THREADS=32
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

echo "✓ Environment ready"
echo ""

# Step 1: Prepare data
echo "Step 1: Preparing data..."
python deepall/train_simple_deepall.py
echo ""

# Step 2: Train base model (very small, just to test)
echo "Step 2: Training base model (d4, ~20M params)..."
echo "⏱️  This will take ~30-60 minutes on CPU..."
echo ""

python -m scripts.base_train \
    --depth=4 \
    --aspect_ratio=64 \
    --max_seq_len=256 \
    --device_batch_size=1 \
    --total_batch_size=256 \
    --num_iterations=50 \
    --eval_every=25 \
    --device_type=cpu

echo ""
echo "✓ Base training complete!"
echo ""

# Step 3: Chat with the model
echo "=========================================="
echo "✓ Training complete!"
echo "=========================================="
echo ""
echo "Your model is ready! Test it:"
echo ""
echo "  python -m scripts.chat_cli -p 'Was ist DeepAll?' --device-type=cpu"
echo ""
echo "Or start the web UI:"
echo "  python -m scripts.chat_web --device-type=cpu"
echo ""

