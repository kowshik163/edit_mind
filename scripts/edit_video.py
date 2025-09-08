#!/usr/bin/env python3
"""
Inference script for autonomous video editing
Usage: python scripts/edit_video.py --video input.mp4 --prompt "Create an AMV style edit with beat sync"
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.hybrid_ai import HybridVideoAI
from utils.config_loader import load_config
from utils.setup_logging import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Autonomous Video Editor")
    parser.add_argument(
        "--video", 
        type=str, 
        required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        required=True,
        help="Editing prompt/instruction"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Output video path (auto-generated if not specified)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="checkpoints/best_model.pt",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/main_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--style", 
        type=str, 
        choices=["amv", "cinematic", "tiktok", "trailer", "sports", "phonk"],
        default=None,
        help="Editing style preset"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup logging
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🎬 Autonomous Video Editor - Inference Mode")
    logger.info(f"Input: {args.video}")
    logger.info(f"Prompt: {args.prompt}")
    
    # Check if input file exists
    if not os.path.exists(args.video):
        logger.error(f"❌ Input video not found: {args.video}")
        return
    
    # Load configuration
    config = load_config(args.config)
    
    # Load trained model
    logger.info(f"🧠 Loading model from: {args.model}")
    try:
        model = HybridVideoAI.from_checkpoint(args.model)
        model.eval()
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return
    
    # Prepare editing prompt
    editing_prompt = args.prompt
    if args.style:
        editing_prompt = f"Style: {args.style}. {editing_prompt}"
    
    # Generate output path if not specified
    if args.output is None:
        input_path = Path(args.video)
        args.output = str(input_path.parent / f"{input_path.stem}_edited{input_path.suffix}")
    
    logger.info(f"🎯 Output will be saved to: {args.output}")
    
    # Run autonomous editing
    try:
        logger.info("🔄 Starting autonomous editing...")
        
        output_path = model.autonomous_edit(
            video_path=args.video,
            prompt=editing_prompt
        )
        
        # Move to specified output path if different
        if output_path != args.output:
            import shutil
            shutil.move(output_path, args.output)
            output_path = args.output
        
        logger.info(f"✅ Editing completed successfully!")
        logger.info(f"📁 Output saved to: {output_path}")
        
        # Print some basic info about the output
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            logger.info(f"📊 Output file size: {file_size:.2f} MB")
        
    except Exception as e:
        logger.error(f"❌ Editing failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return
    
    logger.info("🎉 Autonomous video editing completed!")


if __name__ == "__main__":
    main()
