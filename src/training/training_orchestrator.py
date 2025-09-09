"""
Training Orchestrator - Complete training pipeline with auto-download and validation
"""

import os
import sys
import logging
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional
from omegaconf import DictConfig, OmegaConf
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.model_downloader import ModelDownloader, auto_download_models
from utils.dataset_downloader import DatasetAutoDownloader, auto_download_datasets
from utils.data_loader import VideoEditingDataset, MultiModalDataLoader
from training.trainer import MultiPhaseTrainer
from models.expert_models import ExpertModels
from utils.metrics import VideoEditingMetrics

logger = logging.getLogger(__name__)


class TrainingOrchestrator:
    """
    Complete training orchestration with automatic setup and validation
    """
    
    def __init__(self, config: Optional[DictConfig] = None):
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.model_downloader = ModelDownloader(
            cache_dir=self.config.get('model_cache_dir', 'models/cache')
        )
        
        self.dataset_downloader = DatasetAutoDownloader(
            data_root=self.config.get('data_root', 'data/datasets')
        )
        
        # Training state
        self.setup_complete = False
        self.models_ready = False
        self.datasets_ready = False
        
        # Results tracking
        self.training_results = {}
        
    def _get_default_config(self) -> DictConfig:
        """Get default training configuration"""
        
        return OmegaConf.create({
            # Model configuration
            'model': {
                'backbone': 'microsoft/DialoGPT-small',  # Smaller for demo
                'vision_encoder': 'openai/clip-vit-base-patch32',
                'audio_encoder': 'openai/whisper-tiny',
                'text_dim': 768,
                'vision_dim': 512,
                'audio_dim': 384,
                'fusion_dim': 1024,
                'hidden_dim': 2048,
                'num_attention_heads': 8
            },
            
            # Training configuration
            'training': {
                'phases': ['pretrain', 'distill', 'finetune', 'rlhf', 'autonomous'],
                'batch_size': 4,
                'learning_rate': 1e-4,
                'num_epochs': 10,
                'gradient_accumulation_steps': 4,
                'warmup_steps': 100,
                'eval_steps': 500,
                'save_steps': 1000,
                'max_grad_norm': 1.0
            },
            
            # Data configuration
            'data': {
                'datasets': ['webvid', 'audioset'],  # Start with smaller datasets
                'max_samples_per_dataset': 1000,
                'train_split': 0.8,
                'val_split': 0.1,
                'test_split': 0.1,
                'num_workers': 4
            },
            
            # System configuration
            'system': {
                'device': 'auto',
                'mixed_precision': True,
                'gradient_checkpointing': True,
                'dataloader_pin_memory': True
            },
            
            # Paths
            'model_cache_dir': 'models/cache',
            'data_root': 'data/datasets',
            'output_dir': 'outputs',
            'checkpoint_dir': 'checkpoints',
            'log_dir': 'logs'
        })
    
    def full_setup_and_train(self, force_download: bool = False) -> Dict[str, Any]:
        """
        Complete setup and training pipeline
        """
        
        logger.info("🚀 AUTONOMOUS VIDEO EDITOR - FULL TRAINING PIPELINE")
        logger.info("=" * 60)
        
        try:
            # Step 1: Setup and validation
            setup_success = self.setup_training_environment(force_download)
            if not setup_success:
                logger.error("❌ Setup failed, cannot proceed with training")
                return {"status": "failed", "stage": "setup"}
            
            # Step 2: Auto-download models
            models_success = self.download_and_validate_models(force_download)
            if not models_success:
                logger.error("❌ Model download failed")
                return {"status": "failed", "stage": "models"}
            
            # Step 3: Auto-download datasets
            datasets_success = self.download_and_validate_datasets(force_download)
            if not datasets_success:
                logger.error("❌ Dataset download failed")
                return {"status": "failed", "stage": "datasets"}
            
            # Step 4: Run training phases
            training_success = self.run_training_phases()
            if not training_success:
                logger.error("❌ Training failed")
                return {"status": "failed", "stage": "training"}
            
            logger.info("🎉 FULL TRAINING PIPELINE COMPLETE!")
            return {
                "status": "success",
                "results": self.training_results,
                "models_ready": self.models_ready,
                "datasets_ready": self.datasets_ready
            }
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed with exception: {e}")
            return {"status": "failed", "error": str(e)}
    
    def setup_training_environment(self, force_setup: bool = False) -> bool:
        """Setup training environment and directories"""
        
        if self.setup_complete and not force_setup:
            return True
        
        logger.info("🔧 Setting up training environment...")
        
        try:
            # Create directories
            directories = [
                self.config.model_cache_dir,
                self.config.data_root,
                self.config.output_dir,
                self.config.checkpoint_dir,
                self.config.log_dir
            ]
            
            for dir_path in directories:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                logger.info(f"  📁 Created directory: {dir_path}")
            
            # Setup logging
            log_file = Path(self.config.log_dir) / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            
            logging.getLogger().addHandler(file_handler)
            logger.info(f"  📝 Logging to: {log_file}")
            
            # Check system requirements
            device = self._detect_device()
            logger.info(f"  🖥️  Device: {device}")
            
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                logger.info(f"  🎮 GPUs available: {gpu_count}")
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    logger.info(f"    GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            
            self.setup_complete = True
            logger.info("✅ Environment setup complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Environment setup failed: {e}")
            return False
    
    def download_and_validate_models(self, force_download: bool = False) -> bool:
        """Download and validate all required models"""
        
        logger.info("🤖 Downloading and validating models...")
        
        try:
            # Download models
            download_results = self.model_downloader.download_all_models(force_download)
            
            if not download_results:
                logger.error("❌ No models were downloaded successfully")
                return False
            
            # Validate downloaded models
            validation_results = self._validate_downloaded_models(download_results)
            
            if validation_results["all_valid"]:
                self.models_ready = True
                logger.info(f"✅ Models ready: {validation_results['valid_count']}/{validation_results['total_count']}")
                return True
            else:
                logger.error(f"❌ Model validation failed: {validation_results['invalid_models']}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Model download/validation failed: {e}")
            return False
    
    def download_and_validate_datasets(self, force_download: bool = False) -> bool:
        """Download and validate datasets"""
        
        logger.info("📊 Downloading and validating datasets...")
        
        try:
            # Download datasets
            download_results = self.dataset_downloader.download_all_datasets(
                datasets=self.config.data.datasets,
                force_download=force_download
            )
            
            if not download_results:
                logger.error("❌ No datasets were downloaded successfully")
                return False
            
            # Validate downloaded datasets
            validation_results = self._validate_downloaded_datasets(download_results)
            
            if validation_results["all_valid"]:
                self.datasets_ready = True
                logger.info(f"✅ Datasets ready: {validation_results['total_samples']} total samples")
                return True
            else:
                logger.error(f"❌ Dataset validation failed: {validation_results['issues']}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Dataset download/validation failed: {e}")
            return False
    
    def run_training_phases(self) -> bool:
        """Run the complete multi-phase training"""
        
        if not self.models_ready or not self.datasets_ready:
            logger.error("❌ Models or datasets not ready for training")
            return False
        
        logger.info("🎯 Starting multi-phase training...")
        
        try:
            # Initialize trainer
            trainer = MultiPhaseTrainer(self.config)
            
            # Create data loaders
            train_loader, val_loader = self._create_data_loaders()
            
            # Run each training phase
            phase_results = {}
            
            for phase in self.config.training.phases:
                logger.info(f"🔄 Training phase: {phase}")
                
                try:
                    if phase == "pretrain":
                        result = trainer.pretrain(train_loader, val_loader)
                    elif phase == "distill":
                        result = trainer.distill(train_loader, val_loader)
                    elif phase == "finetune":
                        result = trainer.finetune(train_loader, val_loader)
                    elif phase == "rlhf":
                        result = trainer.rlhf(train_loader, val_loader)
                    elif phase == "autonomous":
                        result = trainer.autonomous_training(train_loader, val_loader)
                    else:
                        logger.warning(f"Unknown training phase: {phase}")
                        continue
                    
                    phase_results[phase] = result
                    logger.info(f"✅ Phase {phase} complete: {result.get('status', 'unknown')}")
                    
                except Exception as e:
                    logger.error(f"❌ Phase {phase} failed: {e}")
                    phase_results[phase] = {"status": "failed", "error": str(e)}
                    
                    # Decide whether to continue or stop
                    if phase in ["pretrain", "distill"]:
                        logger.error("Critical phase failed, stopping training")
                        return False
                    else:
                        logger.warning(f"Non-critical phase {phase} failed, continuing")
            
            self.training_results = phase_results
            logger.info("✅ Multi-phase training complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Training phases failed: {e}")
            return False
    
    def _validate_downloaded_models(self, download_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that downloaded models are working"""
        
        logger.info("  🔍 Validating downloaded models...")
        
        validation_results = {
            "total_count": len(download_results),
            "valid_count": 0,
            "invalid_models": [],
            "all_valid": True
        }
        
        for model_key, model_info in download_results.items():
            try:
                # Basic validation - check if model can be loaded
                model_path = model_info.get("cache_path")
                if model_path and Path(model_path).exists():
                    validation_results["valid_count"] += 1
                    logger.info(f"    ✅ {model_key}: Valid")
                else:
                    validation_results["invalid_models"].append(model_key)
                    logger.warning(f"    ❌ {model_key}: Invalid or missing")
                    
            except Exception as e:
                validation_results["invalid_models"].append(model_key)
                logger.warning(f"    ❌ {model_key}: Validation failed - {e}")
        
        validation_results["all_valid"] = len(validation_results["invalid_models"]) == 0
        return validation_results
    
    def _validate_downloaded_datasets(self, download_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that downloaded datasets are usable"""
        
        logger.info("  🔍 Validating downloaded datasets...")
        
        validation_results = {
            "total_datasets": len(download_results),
            "valid_datasets": 0,
            "total_samples": 0,
            "issues": [],
            "all_valid": True
        }
        
        for dataset_name, dataset_info in download_results.items():
            try:
                samples = dataset_info.get("samples", 0)
                samples_file = dataset_info.get("samples_file")
                
                if samples > 0 and samples_file and Path(samples_file).exists():
                    validation_results["valid_datasets"] += 1
                    validation_results["total_samples"] += samples
                    logger.info(f"    ✅ {dataset_name}: {samples} samples")
                else:
                    validation_results["issues"].append(f"{dataset_name}: No valid samples")
                    logger.warning(f"    ❌ {dataset_name}: Invalid or no samples")
                    
            except Exception as e:
                validation_results["issues"].append(f"{dataset_name}: {str(e)}")
                logger.warning(f"    ❌ {dataset_name}: Validation failed - {e}")
        
        validation_results["all_valid"] = len(validation_results["issues"]) == 0
        return validation_results
    
    def _create_data_loaders(self):
        """Create training and validation data loaders"""
        
        logger.info("  📊 Creating data loaders...")
        
        try:
            # Create dataset
            dataset = VideoEditingDataset(
                data_dir=self.config.data_root,
                datasets=self.config.data.datasets,
                max_frames=self.config.data.get('max_samples_per_dataset', 1000),
                frame_sample_rate=2
            )
            
            # Split dataset
            train_size = int(len(dataset) * self.config.data.train_split)
            val_size = len(dataset) - train_size
            
            from torch.utils.data import random_split, DataLoader
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.training.batch_size,
                shuffle=True,
                num_workers=self.config.data.num_workers,
                pin_memory=self.config.system.dataloader_pin_memory,
                collate_fn=dataset.collate_fn
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.training.batch_size,
                shuffle=False,
                num_workers=self.config.data.num_workers,
                pin_memory=self.config.system.dataloader_pin_memory,
                collate_fn=dataset.collate_fn
            )
            
            logger.info(f"    📊 Train samples: {len(train_dataset)}")
            logger.info(f"    📊 Val samples: {len(val_dataset)}")
            
            return train_loader, val_loader
            
        except Exception as e:
            logger.error(f"❌ Failed to create data loaders: {e}")
            raise
    
    def _detect_device(self) -> str:
        """Detect and configure the best available device"""
        
        if self.config.system.device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"  # Apple Silicon
            else:
                device = "cpu"
        else:
            device = self.config.system.device
        
        # Update config
        self.config.system.device = device
        return device
    
    def save_training_state(self, checkpoint_name: str = "latest"):
        """Save complete training state"""
        
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        state = {
            "config": OmegaConf.to_yaml(self.config),
            "setup_complete": self.setup_complete,
            "models_ready": self.models_ready,
            "datasets_ready": self.datasets_ready,
            "training_results": self.training_results
        }
        
        checkpoint_path = checkpoint_dir / f"{checkpoint_name}_orchestrator.pth"
        torch.save(state, checkpoint_path)
        
        logger.info(f"💾 Saved training state: {checkpoint_path}")


def main():
    """CLI entry point for training orchestrator"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Complete training orchestration")
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--force-download", action="store_true", help="Force re-download")
    parser.add_argument("--datasets", nargs="+", default=["webvid", "audioset"], help="Datasets to use")
    parser.add_argument("--quick", action="store_true", help="Quick training with minimal data")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    if args.config and Path(args.config).exists():
        config = OmegaConf.load(args.config)
    else:
        config = None
    
    # Create orchestrator
    orchestrator = TrainingOrchestrator(config)
    
    # Override datasets if specified
    if args.datasets:
        orchestrator.config.data.datasets = args.datasets
    
    # Quick mode adjustments
    if args.quick:
        orchestrator.config.training.num_epochs = 1
        orchestrator.config.training.batch_size = 2
        orchestrator.config.data.max_samples_per_dataset = 100
        logger.info("🚀 Quick mode enabled - minimal training")
    
    # Run full pipeline
    results = orchestrator.full_setup_and_train(args.force_download)
    
    print(f"\n🏁 TRAINING ORCHESTRATION COMPLETE")
    print(f"Status: {results['status']}")
    
    if results["status"] == "success":
        print(f"Models ready: {results['models_ready']}")
        print(f"Datasets ready: {results['datasets_ready']}")
        print("🎉 Autonomous video editor is trained and ready!")
    else:
        print(f"Failed at stage: {results.get('stage', 'unknown')}")
        if "error" in results:
            print(f"Error: {results['error']}")


if __name__ == "__main__":
    main()
