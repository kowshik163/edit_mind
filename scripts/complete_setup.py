"""
Complete Setup Pipeline for Autonomous Video Editor

This script runs the full pipeline as described:
1. Environment Setup - Install dependencies, set logging & config parsing
2. Model Download & Setup - Pull teacher models (Wan2.2, LTXB, etc.) from HuggingFace
3. Dataset Setup - Download stock videos + template packs (Mixkit, CapCut, etc.)
4. Synthetic Data Generation - Use teacher models to generate synthetic edited videos
5. Distillation Dataset Assembly - Merge template-based + teacher-generated edits
6. Training Loop - Train student model with distillation + perceptual loss
7. Evaluation - Test on unseen clips/templates, save metrics
8. Deployment Prep - Export trained model weights and inference script
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.training_orchestrator import TrainingOrchestrator
from omegaconf import OmegaConf


def setup_logging():
    """Setup comprehensive logging"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"complete_setup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def main():
    """
    Complete setup and training pipeline
    """
    parser = argparse.ArgumentParser(description="Complete Autonomous Video Editor Setup")
    parser.add_argument("--config", type=str, default="configs/main_config.yaml", help="Config file path")
    parser.add_argument("--force-download", action="store_true", help="Force re-download all resources")
    parser.add_argument("--quick", action="store_true", help="Quick setup with minimal data")
    parser.add_argument("--skip-synthetic", action="store_true", help="Skip synthetic data generation")
    parser.add_argument("--num-synthetic-samples", type=int, default=1000, help="Number of synthetic samples to generate")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    logger.info("=" * 80)
    logger.info("🚀 AUTONOMOUS VIDEO EDITOR - COMPLETE SETUP PIPELINE")
    logger.info("=" * 80)
    logger.info("This will:")
    logger.info("1. 🔧 Setup environment and dependencies")
    logger.info("2. 🤖 Download teacher models (Wan2.2, Mochi, LTX-Video, etc.)")
    logger.info("3. 📦 Download template datasets (Mixkit, CapCut, Motion Array, etc.)")
    logger.info("4. 📹 Download stock footage (Pixabay, Pexels, Videvo)")
    logger.info("5. 🎬 Generate synthetic training data using teacher models")
    logger.info("6. 📊 Assemble distillation dataset (templates + synthetic)")
    logger.info("7. 🎯 Train student model with knowledge distillation")
    logger.info("8. 📈 Evaluate on test data and save metrics")
    logger.info("9. 🚀 Prepare deployment-ready model")
    logger.info("=" * 80)
    
    try:
        # Load configuration
        config_path = Path(args.config)
        if config_path.exists():
            config = OmegaConf.load(config_path)
            logger.info(f"📄 Loaded config from: {config_path}")
        else:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            config = None
        
        # Quick mode adjustments
        if args.quick:
            logger.info("⚡ Quick mode enabled - using minimal data for testing")
            if config:
                config.training.num_epochs = 1
                config.training.batch_size = 2
                config.data.max_samples_per_dataset = 50
            args.num_synthetic_samples = 100
        
        # Synthetic data configuration
        if config and not args.skip_synthetic:
            if not hasattr(config, 'synthetic_data'):
                config.synthetic_data = OmegaConf.create({})
            config.synthetic_data.num_samples = args.num_synthetic_samples
        
        # Initialize orchestrator
        logger.info("🎭 Initializing training orchestrator...")
        orchestrator = TrainingOrchestrator(config)
        
        # Run complete pipeline
        logger.info("🚀 Starting complete setup and training pipeline...")
        results = orchestrator.full_setup_and_train(args.force_download)
        
        # Report results
        logger.info("=" * 80)
        logger.info("🏁 PIPELINE COMPLETION REPORT")
        logger.info("=" * 80)
        
        if results["status"] == "success":
            logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info(f"📊 Models ready: {results.get('models_ready', False)}")
            logger.info(f"📊 Datasets ready: {results.get('datasets_ready', False)}")
            logger.info("🎉 Your autonomous video editor is now trained and ready!")
            
            # Generate final summary
            logger.info("📋 FINAL SETUP SUMMARY:")
            logger.info("- Teacher models downloaded and validated")
            logger.info("- Template datasets (Mixkit, CapCut, etc.) downloaded")
            logger.info("- Stock footage downloaded for training")
            logger.info("- Synthetic training data generated using teacher models")
            logger.info("- Student model trained via knowledge distillation")
            logger.info("- Model ready for deployment")
            
            # Show next steps
            logger.info("🚀 NEXT STEPS:")
            logger.info("1. Test your model: python -m src.inference.autonomous_editor --input your_video.mp4")
            logger.info("2. Fine-tune further: python -m scripts.train --resume-from-checkpoint")
            logger.info("3. Deploy: Use the inference script in production")
            
        else:
            logger.error("❌ PIPELINE FAILED")
            logger.error(f"Failed at stage: {results.get('stage', 'unknown')}")
            if "error" in results:
                logger.error(f"Error details: {results['error']}")
            
            # Provide troubleshooting guidance
            logger.info("🔧 TROUBLESHOOTING:")
            logger.info("1. Check your internet connection for downloads")
            logger.info("2. Ensure sufficient disk space (>10GB recommended)")
            logger.info("3. Verify CUDA installation if using GPU")
            logger.info("4. Try --quick mode for testing with minimal data")
            logger.info("5. Check logs for specific error details")
        
        logger.info("=" * 80)
        
        return 0 if results["status"] == "success" else 1
        
    except KeyboardInterrupt:
        logger.warning("⚠️ Setup interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error during setup: {e}")
        logger.error("Full traceback:", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
