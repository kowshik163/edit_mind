#!/bin/bash

# Setup script for Autonomous Video Editor
echo "🚀 Setting up Autonomous Video Editor..."

# Create Python virtual environment
echo "📦 Creating virtual environment..."
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/{webvid10m,audioset,cc3m,amv_dataset,tiktok_edits,movie_trailers,sports_highlights}
mkdir -p experiments/logs
mkdir -p outputs/{videos,timelines,effects}

# Download model checkpoints (placeholder)
echo "🧠 Setting up model checkpoints..."
mkdir -p checkpoints
# wget https://example.com/pretrained_model.pt -O checkpoints/pretrained_model.pt

# Setup pre-commit hooks
echo "🔧 Setting up development tools..."
pre-commit install

# Make scripts executable
chmod +x scripts/*.py

echo "✅ Setup completed!"
echo ""
echo "🎬 To get started:"
echo "  source venv/bin/activate"
echo "  python -m src.core.cli info"
echo ""
echo "🏃 To start training:"
echo "  python scripts/train.py --phase pretraining"
echo ""
echo "✂️ To edit a video:"
echo "  python scripts/edit_video.py --video input.mp4 --prompt 'Create an AMV edit'"
