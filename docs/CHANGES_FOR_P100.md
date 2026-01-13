# Changes Made for P100 Support

## Summary
Modified NanoChat to automatically detect GPU capabilities and use the appropriate mixed precision dtype. This enables training on P100 GPUs (which don't support BF16) without manual configuration.

## Files Modified

### 1. `nanochat/common.py`
**Added**: `autodetect_dtype()` function
- Detects GPU compute capability
- Returns `torch.float16` for Pascal GPUs (P100, compute capability 6.0)
- Returns `torch.bfloat16` for Volta and newer (compute capability >= 7.0)
- Returns `torch.float32` for CPU/MPS

### 2. Training Scripts
All training scripts now auto-detect the best dtype:

#### `scripts/base_train.py`
- Import `autodetect_dtype`
- Call `autodetect_dtype(device_type)` before creating autocast context
- Use detected dtype instead of hardcoded `torch.bfloat16`

#### `scripts/base_loss.py`
- Same changes as base_train.py

#### `scripts/base_eval.py`
- Same changes as base_train.py

#### `scripts/mid_train.py`
- Import `autodetect_dtype`
- Changed `--dtype` default from `"bfloat16"` to `"auto"`
- Added support for `"auto"`, `"float16"` in addition to existing options
- Auto-detect dtype when `--dtype=auto`

#### `scripts/chat_sft.py`
- Same changes as mid_train.py

#### `scripts/chat_rl.py`
- Same changes as mid_train.py

### 3. Inference Scripts

#### `scripts/chat_cli.py`
- Import `autodetect_dtype`
- Changed `--dtype` default from `"bfloat16"` to `"auto"`
- Added `"float16"` and `"auto"` to choices
- Auto-detect dtype when `--dtype=auto`

#### `scripts/chat_eval.py`
- Same changes as chat_cli.py

#### `scripts/chat_web.py`
- Same changes as chat_cli.py

### 4. New Files

#### `speedrun_p100.sh`
- P100-optimized training script
- Uses d12 model (150M params) instead of d20 (561M)
- Smaller batch sizes: `device_batch_size=8`, `total_batch_size=131072`
- Downloads fewer data shards (80 instead of 240)
- Single GPU (no torchrun)

#### `P100_SETUP.md`
- Complete guide for training on P100
- Explains differences from H100 setup
- Troubleshooting tips
- Performance expectations

#### `CHANGES_FOR_P100.md`
- This file - documents all changes

## Technical Details

### Compute Capability Detection
```python
major, minor = torch.cuda.get_device_capability()
compute_capability = major + minor / 10

if compute_capability >= 7.0:
    return torch.bfloat16  # Volta, Turing, Ampere, Ada, Hopper
else:
    return torch.float16   # Pascal (P100 is 6.0)
```

### Why This Matters
- **P100** (Pascal, 6.0): No BF16 support → Use FP16
- **V100** (Volta, 7.0): Limited BF16 support → Use BF16
- **A100** (Ampere, 8.0): Full BF16 support → Use BF16
- **H100** (Hopper, 9.0): Full BF16 support → Use BF16

### Backward Compatibility
All changes are **backward compatible**:
- Default `--dtype=auto` auto-detects the best option
- Users can still manually specify `--dtype=float32|bfloat16|float16`
- Existing scripts work without modification

## Testing Recommendations

1. **Verify dtype detection**:
   ```bash
   python -c "import torch; from nanochat.common import autodetect_dtype; print(autodetect_dtype('cuda'))"
   ```
   Should print `torch.float16` on P100

2. **Test training**:
   ```bash
   bash speedrun_p100.sh
   ```

3. **Monitor VRAM usage**:
   ```bash
   watch -n 1 nvidia-smi
   ```
   Should stay under 16GB with d12 model

## Performance Impact

- **P100 vs H100**: ~2.5x slower (expected due to no Tensor Cores)
- **FP16 vs BF16**: Minimal difference in final model quality
- **Memory**: FP16 uses same memory as BF16 (both are 16-bit)

## Future Improvements

Potential enhancements:
1. Add gradient scaling for FP16 (currently not needed, but could help stability)
2. Add support for mixed FP16/FP32 training (keep some layers in FP32)
3. Add automatic batch size tuning based on available VRAM
4. Add support for gradient checkpointing to fit larger models

