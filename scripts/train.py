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
    
    try:
        logger.info("🚀 Starting Autonomous Video Editor Training")
        logger.info(f"Config: {args.config}")
        logger.info(f"Phase: {args.phase}")
        
        # Validate config file exists
        if not Path(args.config).exists():
            logger.error(f"❌ Config file not found: {args.config}")
            sys.exit(1)
        
        # Load configuration
        logger.info("📋 Loading configuration...")
        config = load_config(args.config)
        
        # Setup data directories
        data_dir = Path(config.get('data_dir', 'data'))
        output_dir = Path(config.get('output_dir', 'output'))
        checkpoints_dir = Path(config.get('checkpoints_dir', 'checkpoints'))
        
        # Create directories if they don't exist
        for directory in [data_dir, output_dir, checkpoints_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Directory ready: {directory}")
        
        # Initialize model
        logger.info("🧠 Initializing Hybrid AI Model...")
        if args.resume and Path(args.resume).exists():
            logger.info(f"📂 Resuming from checkpoint: {args.resume}")
            model = HybridVideoAI.from_checkpoint(args.resume)
        else:
            if args.resume:
                logger.warning(f"⚠️  Checkpoint file not found: {args.resume}, starting fresh")
            model = HybridVideoAI(config)
            logger.info("✅ Model initialized successfully")
        
        # Initialize trainer
        logger.info("🏃 Initializing Trainer...")
        trainer = MultiModalTrainer(config, model)
        logger.info("✅ Trainer initialized successfully")
        
        # Prepare sample data if none exists
        if not (data_dir / 'train_index.json').exists():
            logger.info("📊 No training data found, creating sample dataset...")
            from utils.data_loader import create_sample_dataset
            create_sample_dataset(str(data_dir), num_samples=100)
            logger.info("✅ Sample dataset created")
        
        # Run training based on phase
        logger.info(f"🎯 Starting training phase: {args.phase}")
        
        if args.phase == "all":
            logger.info("🔄 Running all training phases...")
            trainer.train_all_phases()
        elif args.phase == "pretraining":
            logger.info("🔬 Phase 1: Fusion Pretraining...")
            trainer.phase1_fusion_pretraining()
        elif args.phase == "distillation":
            logger.info("🎓 Phase 2: Knowledge Distillation...")
            trainer.phase2_distillation()
        elif args.phase == "finetuning":
            logger.info("🎬 Phase 3: Editing Fine-tuning...")
            trainer.phase3_editing_finetuning()
        elif args.phase == "rlhf":
            logger.info("🧠 Phase 4: RLHF Self-Improvement...")
            trainer.phase4_self_improvement()
        elif args.phase == "autonomous":
            logger.info("🤖 Phase 5: Autonomous Integration...")
            trainer.phase5_autonomous_integration()
        
        logger.info("✅ Training completed successfully!")
        logger.info(f"📊 Model checkpoints saved in: {checkpoints_dir}")
        logger.info(f"📋 Training logs available in: output/logs/")
        
    except KeyboardInterrupt:
        logger.info("⏹️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Training failed with error: {str(e)}")
        if args.debug:
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
