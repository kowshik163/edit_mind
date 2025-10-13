"""
Inference Script for Autonomous Video Editor
Deploy trained model for video editing inference
"""

import os
import sys
import torch
import argparse
import logging
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inference.autonomous_editor import AutonomousVideoEditor
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def main():
    """Run inference with trained model"""
    parser = argparse.ArgumentParser(description="Autonomous Video Editor Inference")
    parser.add_argument("--input", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, help="Output video path (auto-generated if not provided)")
    parser.add_argument("--style", type=str, default="cinematic", 
                       choices=["cinematic", "phonk", "tiktok", "amv", "professional"],
                       help="Editing style to apply")
    parser.add_argument("--config", type=str, default="configs/main_config.yaml", help="Config file")
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint path")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load config
    config = OmegaConf.load(args.config) if Path(args.config).exists() else OmegaConf.create({})
    
    # Initialize editor
    editor = AutonomousVideoEditor(config)
    
    # Load checkpoint if provided
    if args.checkpoint and Path(args.checkpoint).exists():
        editor.load_checkpoint(args.checkpoint)
    
    # Generate output path if not provided
    if not args.output:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_edited_{args.style}.mp4")
    
    logger.info(f"🎬 Processing: {args.input}")
    logger.info(f"📤 Output: {args.output}")
    logger.info(f"🎨 Style: {args.style}")
    
    try:
        # Run editing
        result = editor.edit_video(
            input_path=args.input,
            output_path=args.output,
            style=args.style
        )
        
        if result:
            logger.info(f"✅ Video edited successfully: {args.output}")
        else:
            logger.error("❌ Video editing failed")
            
    except Exception as e:
        logger.error(f"❌ Error during inference: {e}")


if __name__ == "__main__":
    main()