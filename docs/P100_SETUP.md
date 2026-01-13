# NanoChat on P100 GPU (16GB)

This guide explains how to train NanoChat on a P100 GPU with 16GB VRAM.

## Key Differences from H100 Setup

The P100 is a Pascal architecture GPU (compute capability 6.0) with some limitations:
- **No BF16 support** - We use FP16 instead
- **No Tensor Cores** - Training will be slower
- **16GB VRAM** - We need smaller batch sizes and a smaller model

## Automatic Mixed Precision Detection

The codebase now **automatically detects** your GPU capabilities and selects the best dtype:
- **Ampere/Hopper (A100, H100)**: BF16
- **Volta/Turing (V100, T4)**: BF16 (limited support)
- **Pascal (P100)**: FP16
- **CPU/MPS**: FP32

You don't need to manually specify `--dtype` anymore - it will auto-detect!

## Quick Start

### Option 1: Use the P100-optimized script (Recommended)

```bash
bash speedrun_p100.sh
```

This script:
- Trains a **d12 model** (~150M params) instead of d20 (561M)
- Uses **smaller batch sizes** to fit in 16GB
- Automatically uses **FP16** mixed precision
- Takes ~8-12 hours on a single P100

### Option 2: Manual training with custom settings

```bash
# Setup environment
uv sync --extra gpu
source .venv/bin/activate

# Train tokenizer
python -m nanochat.dataset -n 8
python -m scripts.tok_train --max_chars=2000000000 --vocab_size=65536

# Download more data in background
python -m nanochat.dataset -n 80 &

# Train base model (d12, smaller batches)
python -m scripts.base_train \
    --depth=12 \
    --device_batch_size=8 \
    --total_batch_size=131072 \
    --target_param_data_ratio=20

# The dtype will be auto-detected as float16 on P100
```

## Model Size Recommendations

| Model | Params | VRAM (FP16) | Batch Size | Training Time |
|-------|--------|-------------|------------|---------------|
| d8    | ~70M   | ~8GB        | 16         | ~4 hours      |
| d12   | ~150M  | ~12GB       | 8          | ~8 hours      |
| d16   | ~270M  | ~14GB       | 4          | ~16 hours     |
| d20   | ~560M  | ~18GB       | ❌ Too big | ❌            |

**Recommendation**: Use **d12** for the best balance of quality and training time.

## Troubleshooting

### Out of Memory (OOM)
If you get OOM errors:
1. Reduce `--device_batch_size` (try 4 or 2)
2. Reduce `--total_batch_size` (try 65536)
3. Use a smaller model (d8 instead of d12)

### Slow Training
P100 is ~3-4x slower than H100 for this workload:
- No Tensor Cores
- Lower memory bandwidth
- Older architecture

This is expected. Be patient!

### Dtype Errors
If you see errors about bfloat16:
- Make sure you're using the latest code (with autodetect_dtype)
- You can manually override with `--dtype=float16`

## Performance Expectations

On a single P100:
- **Tokenizer training**: ~10 minutes
- **Base model (d12)**: ~8 hours
- **Midtraining**: ~30 minutes
- **SFT**: ~20 minutes
- **Total**: ~9-10 hours

Compare to 8×H100:
- **Total**: ~4 hours

So you're looking at **~2.5x slower** on a single P100, which is actually pretty good!

## What Changed in the Code

All training and inference scripts now:
1. Import `autodetect_dtype` from `nanochat.common`
2. Call `autodetect_dtype(device_type)` to get the best dtype
3. Use that dtype for `torch.amp.autocast`

The detection logic is in `nanochat/common.py`:
- Checks GPU compute capability
- Returns `torch.float16` for P100 (compute capability 6.0)
- Returns `torch.bfloat16` for newer GPUs (7.0+)

## Next Steps

After training completes:
```bash
# Chat with your model
python -m scripts.chat_cli -p "Why is the sky blue?"

# Or launch web UI
python -m scripts.chat_web
```

Enjoy your $10 ChatGPT clone! 🚀

