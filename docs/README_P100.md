# NanoChat on P100 - Quick Start Guide

## 🎯 What This Is

This is a modified version of Andrej Karpathy's NanoChat that **automatically detects your GPU** and uses the right precision:
- **P100** → FP16 (float16)
- **V100/A100/H100** → BF16 (bfloat16)
- **CPU** → FP32 (float32)

No manual configuration needed!

## 🚀 Quick Start (3 Steps)

### Step 1: Verify Your Setup
```bash
bash verify_p100_setup.sh
```

This checks:
- ✓ GPU is detected
- ✓ VRAM is sufficient
- ✓ Dependencies can be installed

### Step 2: Train Your Model
```bash
bash speedrun_p100.sh
```

This will:
- Install dependencies
- Train tokenizer (~10 min)
- Train d12 model (~8 hours on P100)
- Run midtraining and SFT
- Generate a report

### Step 3: Chat with Your Model
```bash
source .venv/bin/activate
python -m scripts.chat_cli -p "Why is the sky blue?"
```

Or launch the web UI:
```bash
python -m scripts.chat_web
```

## 📊 What to Expect

| Stage | Time (P100) | Output |
|-------|-------------|--------|
| Tokenizer | ~10 min | 65K vocab tokenizer |
| Base Training | ~8 hours | d12 model (150M params) |
| Midtraining | ~30 min | Personality/tools |
| SFT | ~20 min | Chat-tuned model |
| **Total** | **~9 hours** | **Ready to chat!** |

Compare to 8×H100: ~4 hours for d20 (561M params)

## 💾 Resource Requirements

- **GPU**: P100 with 16GB VRAM
- **Disk**: ~30GB free space
- **RAM**: 16GB+ recommended
- **Time**: ~9 hours
- **Cost**: ~$10-15 (depending on cloud provider)

## 🔧 What Changed

All training scripts now **auto-detect** the best dtype:

```python
# Old (hardcoded)
autocast_ctx = torch.amp.autocast(dtype=torch.bfloat16)

# New (auto-detect)
dtype = autodetect_dtype(device_type)  # Returns float16 on P100
autocast_ctx = torch.amp.autocast(dtype=dtype)
```

See `CHANGES_FOR_P100.md` for full technical details.

## 🎛️ Advanced Usage

### Train a Smaller Model (Faster)
```bash
python -m scripts.base_train \
    --depth=8 \
    --device_batch_size=16 \
    --total_batch_size=131072
```

This trains d8 (~70M params) in ~4 hours.

### Train a Larger Model (Better Quality)
```bash
python -m scripts.base_train \
    --depth=16 \
    --device_batch_size=4 \
    --total_batch_size=131072
```

This trains d16 (~270M params) in ~16 hours.

### Manual Dtype Override
```bash
python -m scripts.base_train --dtype=float16  # Force FP16
python -m scripts.base_train --dtype=bfloat16  # Force BF16
python -m scripts.base_train --dtype=auto  # Auto-detect (default)
```

## 🐛 Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
python -m scripts.base_train --device_batch_size=4

# Or use smaller model
python -m scripts.base_train --depth=8
```

### Slow Training
P100 is ~2.5x slower than H100. This is normal:
- No Tensor Cores
- Older architecture
- Lower memory bandwidth

Be patient! ☕

### "Module not found" Errors
```bash
# Make sure venv is activated
source .venv/bin/activate

# Reinstall dependencies
uv sync --extra gpu
```

## 📚 Documentation

- `P100_SETUP.md` - Detailed setup guide
- `CHANGES_FOR_P100.md` - Technical implementation details
- `SUMMARY.md` - High-level overview
- `README_P100.md` - This file

## 🎓 The Andrej Karpathy Way

This implementation follows AK's philosophy:
1. **Minimal** - Only essential changes
2. **Hackable** - Easy to understand and modify
3. **Practical** - Solves a real problem
4. **Educational** - Well-documented

The entire change is ~100 lines of code across 10 files.

## 🤝 Contributing

Found a bug? Have an improvement?
1. Test it on P100
2. Make sure it's backward compatible
3. Document it clearly
4. Submit a PR

## 📝 License

Same as NanoChat (MIT)

## 🙏 Credits

- **Andrej Karpathy** - Original NanoChat
- **This modification** - P100 support via auto-detection

---

**Ready to build your $10 ChatGPT clone?**

```bash
bash speedrun_p100.sh
```

Let's go! 🚀

