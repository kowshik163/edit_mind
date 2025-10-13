#!/bin/bash

# Autonomous Video Editor Installation Script
# This script handles the complex dependency installation process

set -e  # Exit on any error

echo "🚀 Starting Autonomous Video Editor Installation..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1-2)
echo "📋 Detected Python version: $python_version"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
    echo "❌ Error: Python 3.8 or higher is required"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip, setuptools, wheel first
echo "⬆️ Upgrading build tools..."
pip install --upgrade pip setuptools wheel

# Install core dependencies first (minimal set)
echo "📚 Installing core dependencies..."
pip install numpy>=1.24.0 
pip install pillow>=10.0.0
pip install requests>=2.31.0
pip install tqdm>=4.65.0
pip install pyyaml>=6.0.1
pip install omegaconf>=2.3.0
pip install rich>=13.4.0

# Install PyTorch separately (most common source of issues)
echo "🔥 Installing PyTorch..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if [[ $(uname -m) == "arm64" ]]; then
        # Apple Silicon
        pip install torch torchvision torchaudio
    else
        # Intel Mac
        pip install torch torchvision torchaudio
    fi
else
    # Linux - detect CUDA
    if command -v nvidia-smi &> /dev/null; then
        echo "🎮 CUDA detected, installing PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        echo "💻 No CUDA detected, installing CPU-only PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
fi

# Install OpenCV
echo "📷 Installing OpenCV..."
pip install opencv-python>=4.8.0

# Install remaining ML dependencies
echo "🧠 Installing ML dependencies..."
pip install transformers>=4.30.0
pip install accelerate>=0.20.0
pip install datasets>=2.12.0

# Install video processing
echo "🎬 Installing video processing libraries..."
pip install moviepy>=1.0.3
pip install ffmpeg-python>=0.2.0
pip install imageio>=2.31.0
pip install imageio-ffmpeg>=0.4.8

# Install audio processing
echo "🎵 Installing audio processing libraries..."
pip install librosa>=0.10.0
pip install soundfile>=0.12.1

# Install scientific computing
echo "🔬 Installing scientific computing libraries..."
pip install scipy>=1.11.0
pip install scikit-learn>=1.3.0
pip install pandas>=2.0.0

# Install training dependencies
echo "🏋️ Installing training dependencies..."
pip install deepspeed>=0.9.0 || echo "⚠️ DeepSpeed installation failed (optional)"
pip install wandb>=0.15.0
pip install tensorboard>=2.13.0

# Install package in development mode
echo "📦 Installing autonomous video editor package..."
pip install -e .

echo "✅ Installation completed successfully!"
echo ""
echo "🎯 Usage:"
echo "  source .venv/bin/activate  # Activate environment"
echo "  python -m src.core.cli --help  # Show help"
echo "  python scripts/simple_demo.py  # Run demo"
echo ""
echo "🔧 Troubleshooting:"
echo "  - If you encounter issues, check the logs above"
echo "  - For GPU support, ensure CUDA drivers are installed"
echo "  - For Apple Silicon Macs, some packages may need Rosetta 2"