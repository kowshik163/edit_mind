#!/usr/bin/env python3
"""
Training script for the Autonomous Video Editor
Usage: python scripts/train.py --config configs/main_config.yaml
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.hybrid_ai import HybridVideoAI
from training.trainer import MultiModalTrainer
from utils.config_loader import load_config
from utils.setup_logging import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Train Autonomous Video Editor")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/main_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--phase", 
        type=str, 
        choices=["all", "pretraining", "distillation", "finetuning", "rlhf", "autonomous"],
        default="all",
        help="Training phase to run"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting Autonomous Video Editor Training")
    logger.info(f"Config: {args.config}")
    logger.info(f"Phase: {args.phase}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize model
    logger.info("🧠 Initializing Hybrid AI Model...")
    if args.resume:
        logger.info(f"📂 Resuming from checkpoint: {args.resume}")
        model = HybridVideoAI.from_checkpoint(args.resume)
    else:
        model = HybridVideoAI(config)
    
    # Initialize trainer
    logger.info("🏃 Initializing Trainer...")
    trainer = MultiModalTrainer(config, model)
    
    # Run training based on phase
    if args.phase == "all":
        trainer.train_all_phases()
    elif args.phase == "pretraining":
        trainer.phase1_fusion_pretraining()
    elif args.phase == "distillation":
        trainer.phase2_distillation()
    elif args.phase == "finetuning":
        trainer.phase3_editing_finetuning()
    elif args.phase == "rlhf":
        trainer.phase4_self_improvement()
    elif args.phase == "autonomous":
        trainer.phase5_autonomous_integration()
    
    logger.info("✅ Training completed successfully!")


if __name__ == "__main__":
    main()
