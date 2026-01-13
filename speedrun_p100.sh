#!/bin/bash

# P100-optimized training script for nanochat
# This is adapted from speedrun.sh but tuned for P100 (16GB, no BF16 support)
# Expected runtime: ~8-12 hours on single P100 (vs 4 hours on 8xH100)

# Key differences from speedrun.sh:
# - Uses FP16 instead of BF16 (P100 doesn't support BF16)
# - Smaller model (d12 instead of d20) to fit in 16GB VRAM
# - Smaller batch sizes
# - Single GPU (no DDP)

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup (optional)
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Report setup
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer (same as speedrun.sh)

# Download first 8 shards for tokenizer training
python -m nanochat.dataset -n 8

# Download more shards in background (we need fewer for d12 model)
# d12 model is ~150M params, Chinchilla ratio 20x means 3B tokens
# At 4.8 chars/token, that's ~14.4B chars, or ~58 shards
# Round up to 80 for safety
python -m nanochat.dataset -n 80 &
DATASET_DOWNLOAD_PID=$!

# Train tokenizer
python -m scripts.tok_train --max_chars=2000000000 --vocab_size=65536
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining) - P100 optimized

echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# P100 specific settings:
# - depth=12 (instead of 20) -> ~150M params instead of 561M
# - device_batch_size=8 (instead of 32) -> fits in 16GB
# - total_batch_size=131072 (instead of 524288) -> 4x smaller
# - No torchrun (single GPU)

python -m scripts.base_train \
    --depth=12 \
    --device_batch_size=8 \
    --total_batch_size=131072 \
    --target_param_data_ratio=20 \
    --run=$WANDB_RUN

# Evaluate
python -m scripts.base_loss
python -m scripts.base_eval

# -----------------------------------------------------------------------------
# Midtraining

curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

python -m scripts.mid_train --run=$WANDB_RUN
python -m scripts.chat_eval -- -i mid

# -----------------------------------------------------------------------------
# Supervised Finetuning

python -m scripts.chat_sft --run=$WANDB_RUN
python -m scripts.chat_eval -- -i sft

# -----------------------------------------------------------------------------
# Generate report
python -m nanochat.report generate

echo "Training complete! You can now chat with your model:"
echo "  python -m scripts.chat_cli -p 'Why is the sky blue?'"
echo "  python -m scripts.chat_web"

