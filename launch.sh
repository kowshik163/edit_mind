#!/bin/bash
# Autonomous Video Editor - Main Launch Script

echo "🎬 Autonomous Video Editor - Starting..."

# Set Python path
export PYTHONPATH="/Users/gkowshikreddy/Downloads/auto_editor_prototype/src:$PYTHONPATH"

# Check if we're in the right directory
if [ ! -f "src/core/hybrid_ai.py" ]; then
    echo "❌ Please run this script from the project root directory"
    exit 1
fi

# Available commands:
case "$1" in
    "train")
        echo "🚀 Starting training pipeline..."
        python scripts/train.py --config configs/config.yaml
        ;;
    "edit")
        echo "✂️  Starting video editing..."
        python -m src.core.cli edit --input "$2" --output "$3"
        ;;
    "info")
        echo "📊 System information..."
        python -m src.core.cli info
        ;;
    "test")
        echo "🧪 Running system tests..."
        python scripts/test_implementations.py
        ;;
    "setup-datasets")
        echo "📥 Setting up datasets..."
        python scripts/setup_datasets.py
        ;;
    *)
        echo "🎬 Autonomous Video Editor"
        echo ""
        echo "Usage: ./launch.sh [command] [args...]"
        echo ""
        echo "Available commands:"
        echo "  train           - Start training pipeline"
        echo "  edit <in> <out> - Edit video file"
        echo "  info            - Show system info"
        echo "  test            - Run tests"
        echo "  setup-datasets  - Download datasets"
        echo ""
        echo "Examples:"
        echo "  ./launch.sh info"
        echo "  ./launch.sh edit input.mp4 output.mp4"
        echo "  ./launch.sh train"
        ;;
esac
