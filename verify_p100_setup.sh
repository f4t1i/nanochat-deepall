#!/bin/bash

# Verification script for P100 setup
# Run this to check if your system is ready for training

echo "=========================================="
echo "NanoChat P100 Setup Verification"
echo "=========================================="
echo ""

# Check if we're on a GPU server
echo "1. Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
    echo "✓ NVIDIA GPU detected"
else
    echo "✗ nvidia-smi not found. Are you on a GPU server?"
    exit 1
fi
echo ""

# Check GPU compute capability
echo "2. Checking GPU compute capability..."
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
echo "Compute capability: $COMPUTE_CAP"

if [[ "$COMPUTE_CAP" == "6.0" ]]; then
    echo "✓ P100 detected (compute 6.0) - will use FP16"
elif [[ "$COMPUTE_CAP" > "7.0" ]]; then
    echo "✓ Modern GPU detected (compute $COMPUTE_CAP) - will use BF16"
else
    echo "⚠ Older GPU detected (compute $COMPUTE_CAP) - will use FP16"
fi
echo ""

# Check VRAM
echo "3. Checking GPU memory..."
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
echo "Total VRAM: ${VRAM} MB"

if [ "$VRAM" -lt 12000 ]; then
    echo "⚠ Warning: Less than 12GB VRAM. Consider using d8 model instead of d12"
elif [ "$VRAM" -lt 16000 ]; then
    echo "✓ 12-16GB VRAM - d12 model recommended"
elif [ "$VRAM" -lt 20000 ]; then
    echo "✓ 16-20GB VRAM - d12 or d16 model recommended"
else
    echo "✓ 20GB+ VRAM - d20 model possible"
fi
echo ""

# Check for uv
echo "4. Checking for uv package manager..."
if command -v uv &> /dev/null; then
    echo "✓ uv is installed"
else
    echo "✗ uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "✓ uv installed"
fi
echo ""

# Check for virtual environment
echo "5. Checking Python virtual environment..."
if [ -d ".venv" ]; then
    echo "✓ Virtual environment exists"
else
    echo "⚠ Virtual environment not found. Creating..."
    uv venv
    echo "✓ Virtual environment created"
fi
echo ""

# Check if dependencies are installed
echo "6. Checking Python dependencies..."
if [ -d ".venv" ]; then
    source .venv/bin/activate
    if python -c "import torch" 2>/dev/null; then
        TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
        echo "✓ PyTorch installed: $TORCH_VERSION"
        
        # Check CUDA availability
        CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
        if [ "$CUDA_AVAILABLE" = "True" ]; then
            echo "✓ PyTorch can access CUDA"
        else
            echo "✗ PyTorch cannot access CUDA. Check your installation."
        fi
    else
        echo "⚠ PyTorch not installed. Run: uv sync --extra gpu"
    fi
    deactivate
else
    echo "⚠ Cannot check dependencies without virtual environment"
fi
echo ""

# Check disk space
echo "7. Checking disk space..."
CACHE_DIR="$HOME/.cache/nanochat"
AVAILABLE_SPACE=$(df -BG "$HOME" | tail -1 | awk '{print $4}' | sed 's/G//')
echo "Available space: ${AVAILABLE_SPACE}GB"

if [ "$AVAILABLE_SPACE" -lt 30 ]; then
    echo "⚠ Warning: Less than 30GB free. Training requires ~25GB for data and checkpoints"
else
    echo "✓ Sufficient disk space"
fi
echo ""

# Summary
echo "=========================================="
echo "Summary"
echo "=========================================="
echo ""

if [[ "$COMPUTE_CAP" == "6.0" ]] && [ "$VRAM" -ge 12000 ]; then
    echo "✓ Your P100 is ready for training!"
    echo ""
    echo "Recommended command:"
    echo "  bash speedrun_p100.sh"
    echo ""
    echo "This will train a d12 model (~150M params) in ~8-12 hours"
elif [[ "$COMPUTE_CAP" > "7.0" ]]; then
    echo "✓ Your GPU is ready for training!"
    echo ""
    echo "You have a modern GPU. You can use the standard speedrun:"
    echo "  bash speedrun.sh"
    echo ""
    echo "Or the P100 script for a smaller/faster model:"
    echo "  bash speedrun_p100.sh"
else
    echo "⚠ Your system may have issues. Please review the warnings above."
fi

echo ""
echo "For more information, see P100_SETUP.md"
echo "=========================================="

