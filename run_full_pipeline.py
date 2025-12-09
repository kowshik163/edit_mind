#!/usr/bin/env python3
"""
Complete Autonomous Video Editor - One-Click Training and Setup
"""

import sys
import os
import logging
import argparse
from pathlib import Path
from datetime import datetime

₹
# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def setup_logging(level=logging.INFO):
    """Setup comprehensive logging"""
    
    # Create logs directory
    logs_dir = Path("output/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"autonomous_editor_{timestamp}.log"
    
    # Setup logging configuration
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"📝 Logging to: {log_file}")
    return logger


def check_system_requirements(logger):
    """Check system requirements and dependencies"""
    
    logger.info("🔍 Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("❌ Python 3.8+ required")
        return False
    
    logger.info(f"✅ Python {sys.version.split()[0]}")
    
    # Check critical imports
    missing_packages = []
    
    try:
        import torch
        logger.info(f"✅ PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            logger.info(f"✅ CUDA available - {torch.cuda.device_count()} GPUs")
        else:
            logger.info("ℹ️  CUDA not available - using CPU")
            
    except ImportError:
        missing_packages.append("torch")
    
    try:
        import transformers
        logger.info(f"✅ Transformers {transformers.__version__}")
    except ImportError:
        missing_packages.append("transformers")
    
    try:
        import trl
        logger.info(f"✅ TRL {trl.__version__}")
    except ImportError:
        missing_packages.append("trl")
    
    try:
        import cv2
        logger.info(f"✅ OpenCV {cv2.__version__}")
    except ImportError:
        missing_packages.append("opencv-python")
    
    try:
        import librosa
        logger.info(f"✅ Librosa {librosa.__version__}")
    except ImportError:
        missing_packages.append("librosa")
    
    if missing_packages:
        logger.error(f"❌ Missing packages: {', '.join(missing_packages)}")
        logger.info("Install with: pip install -r requirements.txt")
        return False
    
    logger.info("✅ All dependencies available")
    return True


def install_dependencies(logger, force=False):
    """Install required dependencies"""
    
    if not force:
        logger.info("⚠️  Use --install-deps to automatically install dependencies")
        return False
    
    logger.info("📦 Installing dependencies...")
    
    import subprocess
    
    try:
        # Install requirements
        cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Dependencies installed successfully")
            return True
        else:
            logger.error(f"❌ Failed to install dependencies: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error installing dependencies: {e}")
        return False


def run_complete_pipeline(config_file=None, quick_mode=False, force_download=False, logger=None):
    """Run the complete autonomous video editor training pipeline"""
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("🚀 AUTONOMOUS VIDEO EDITOR - COMPLETE PIPELINE")
    logger.info("=" * 60)
    
    try:
        # Import main components
        from training.training_orchestrator import TrainingOrchestrator
        from omegaconf import OmegaConf
        
        # Load configuration
        if config_file and Path(config_file).exists():
            logger.info(f"📋 Loading config from {config_file}")
            config = OmegaConf.load(config_file)
        else:
            logger.info("📋 Using default configuration")
            config = None
        
        # Create orchestrator
        logger.info("🎯 Initializing training orchestrator...")
        orchestrator = TrainingOrchestrator(config)
        
        # Quick mode adjustments
        if quick_mode:
            logger.info("⚡ Quick mode enabled - minimal training for testing")
            orchestrator.config.training.num_epochs = 1
            orchestrator.config.training.batch_size = 2
            orchestrator.config.data.max_samples_per_dataset = 50
            orchestrator.config.training.eval_steps = 10
            orchestrator.config.training.save_steps = 20
        
        # Run full pipeline
        logger.info("🎬 Starting complete training pipeline...")
        results = orchestrator.full_setup_and_train(force_download=force_download)
        
        # Report results
        logger.info("🏁 PIPELINE COMPLETE")
        logger.info("=" * 40)
        
        if results["status"] == "success":
            logger.info("🎉 SUCCESS! Autonomous video editor is ready!")
            logger.info(f"✅ Models ready: {results['models_ready']}")
            logger.info(f"✅ Datasets ready: {results['datasets_ready']}")
            
            if "results" in results:
                training_results = results["results"]
                logger.info("📊 Training Results:")
                for phase, result in training_results.items():
                    logger.info(f"  {phase}: {result.get('status', 'completed')}")
            
            logger.info("\n🚀 You can now use the autonomous video editor!")
            logger.info("   Run: python scripts/edit_video.py --input your_video.mp4")
            
            return True
            
        else:
            logger.error(f"❌ FAILED at stage: {results.get('stage', 'unknown')}")
            if "error" in results:
                logger.error(f"Error: {results['error']}")
            
            logger.info("\n🔧 Troubleshooting suggestions:")
            logger.info("  - Check system requirements")
            logger.info("  - Verify internet connection for downloads")
            logger.info("  - Check available disk space")
            logger.info("  - Try --quick mode for testing")
            
            return False
    
    except Exception as e:
        logger.error(f"❌ Pipeline failed with exception: {e}")
        logger.exception("Full traceback:")
        return False


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="Autonomous Video Editor - Complete Setup and Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test run with minimal training
    python run_full_pipeline.py --quick
    
    # Full training with custom config
    python run_full_pipeline.py --config configs/main_config.yaml
    
    # Force re-download all datasets and models
    python run_full_pipeline.py --force-download
    
    # Install dependencies and run
    python run_full_pipeline.py --install-deps --quick
        """
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/main_config.yaml",
        help="Configuration file path (default: configs/main_config.yaml)"
    )
    
    parser.add_argument(
        "--quick", 
        action="store_true",
        help="Quick mode - minimal training for testing (recommended for first run)"
    )
    
    parser.add_argument(
        "--force-download", 
        action="store_true",
        help="Force re-download of all models and datasets"
    )
    
    parser.add_argument(
        "--install-deps", 
        action="store_true",
        help="Automatically install required dependencies"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--check-only", 
        action="store_true",
        help="Only check system requirements, don't run training"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(log_level)
    
    logger.info("🎬 AUTONOMOUS VIDEO EDITOR SETUP")
    logger.info("=" * 50)
    
    # Check system requirements
    if not check_system_requirements(logger):
        if args.install_deps:
            if install_dependencies(logger, force=True):
                logger.info("✅ Dependencies installed, please restart the script")
            return 1
        else:
            logger.error("❌ System requirements not met")
            return 1
    
    # If check-only mode, exit here
    if args.check_only:
        logger.info("✅ System check complete")
        return 0
    
    # Run the complete pipeline
    success = run_complete_pipeline(
        config_file=args.config,
        quick_mode=args.quick,
        force_download=args.force_download,
        logger=logger
    )
    
    if success:
        logger.info("\n🎉 AUTONOMOUS VIDEO EDITOR IS READY!")
        logger.info("Thank you for using the Autonomous Video Editor!")
        return 0
    else:
        logger.error("\n❌ Setup failed. Check logs for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())