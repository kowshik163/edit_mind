#!/usr/bin/env python3
"""
Autonomous Video Editor - Main Application
Complete autonomous video editing system with auto-download, training, and inference
"""

import os
import sys
import argparse
import logging
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Core imports
from src.core.hybrid_ai import HybridVideoAI
from src.utils.model_downloader import ModelDownloader
from src.utils.dataset_downloader import DatasetAutoDownloader
from src.training.training_orchestrator import TrainingOrchestrator
from src.generation.effect_generator import AdvancedEffectGenerator
from src.inference.autonomous_editor import AutonomousVideoEditor

# Enhanced Configuration - Now loads from YAML config
try:
    from omegaconf import DictConfig, OmegaConf
    HAS_OMEGACONF = True
except ImportError:
    HAS_OMEGACONF = False

DEFAULT_CONFIG = {
    'model': {
        # Use advanced teacher models by default
        'backbone': 'meta-llama/Llama-2-7b-hf',  # Upgraded from DialoGPT-small
        'vision_encoder': 'google/siglip-large-patch16-384',  # Upgraded from CLIP
        'audio_encoder': 'openai/whisper-large-v3',  # Upgraded from whisper-tiny
        'text_dim': 1024,  # Increased for larger models
        'vision_dim': 1024,  # Increased for SiGLIP
        'audio_dim': 1024,  # Increased for Whisper-large
        'fusion_dim': 2048,  # Increased fusion capacity
        'hidden_dim': 4096,  # Increased hidden dimensions
        'num_attention_heads': 16  # More attention heads for better performance
    },
    # Add teachers configuration for distillation
    'teachers': {
        'text_model': 'meta-llama/Llama-2-7b-hf',  # Use available model instead of Llama-4-70b
        'vision_encoder': 'google/siglip-large-patch16-384',
        'audio_models': ['openai/whisper-large-v3'],
        'object_detection': 'facebook/detr-resnet-50',  # Use DETR as RT-DETR alternative
        'segmentation': 'facebook/sam-vit-base',  # Use SAM-base as HQ-SAM alternative
        'code_generation': 'codellama/CodeLlama-7b-Python-hf'
    },
    'training': {
        'auto_setup': True,
        'output_dir': 'outputs/',
        'batch_size': 4,
        'learning_rate': 1e-4,
        'num_epochs': 10,
        'save_steps': 1000,
        'eval_steps': 500,
        'warmup_steps': 100,
        'gradient_accumulation_steps': 4,
        'dataloader_num_workers': 4,
        'bf16': True,
        'phases': {
            'pretraining': {'enabled': True, 'epochs': 3},
            'distillation': {'enabled': True, 'epochs': 2},  
            'fine_tuning': {'enabled': True, 'epochs': 3, 'lora_r': 16, 'lora_alpha': 32},
            'rlhf': {'enabled': True, 'epochs': 2, 'beta': 0.1, 'ref_model_path': None},
            'autonomous': {'enabled': True, 'epochs': 2}
        }
    },
    'datasets': {
        'auto_download': True,
        'webvid': {'enabled': True, 'samples': 10000},
        'audioset': {'enabled': True, 'samples': 5000}, 
        'activitynet': {'enabled': True, 'samples': 5000},
        'tvsum': {'enabled': True},
        'summe': {'enabled': True}
    },
    'model_cache_dir': 'models/cache',
    'data_dir': 'data/',
    'effects': {
        'enable_advanced': True,
        'gpu_acceleration': True,
        'quality': 'high'
    }
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('autonomous_editor.log')
    ]
)
logger = logging.getLogger(__name__)


class AutonomousVideoEditorApp:
    """Main application class for the autonomous video editor"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None):
        """
        Initialize the autonomous video editor application
        
        Args:
            config: Optional configuration dictionary
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config, config_path)
        self.model_downloader = None
        self.dataset_downloader = None
        self.training_orchestrator = None
        self.effect_generator = None
        self.autonomous_editor = None
        self.hybrid_ai = None
        
        logger.info("🎬 Autonomous Video Editor Application Initialized")
        logger.info(f"Using models: Text={self.config.get('teachers', {}).get('text_model', 'default')}")
        logger.info(f"Vision={self.config.get('teachers', {}).get('vision_encoder', 'default')}")
        logger.info(f"Audio={self.config.get('teachers', {}).get('audio_models', ['default'])[0]}")
    
    def _load_config(self, config: Optional[Dict[str, Any]], config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file or use provided/default config"""
        
        if config is not None:
            logger.info("Using provided configuration")
            return config
        
        # Try to load from YAML config file
        yaml_config_path = config_path or 'configs/main_config.yaml'
        
        if HAS_OMEGACONF and Path(yaml_config_path).exists():
            try:
                logger.info(f"Loading configuration from {yaml_config_path}")
                yaml_config = OmegaConf.load(yaml_config_path)
                
                # Convert to regular dict and merge with defaults
                yaml_dict = OmegaConf.to_container(yaml_config, resolve=True)
                
                # Merge with default config (YAML takes precedence)
                merged_config = self._deep_merge_config(DEFAULT_CONFIG.copy(), yaml_dict)
                
                logger.info("✅ Successfully loaded YAML configuration")
                return merged_config
                
            except Exception as e:
                logger.warning(f"Failed to load YAML config: {e}, using default")
                
        else:
            if not HAS_OMEGACONF:
                logger.warning("OmegaConf not available, install with: pip install omegaconf")
            if not Path(yaml_config_path).exists():
                logger.warning(f"Config file not found: {yaml_config_path}")
                
        logger.info("Using default configuration")
        return DEFAULT_CONFIG
    
    def _deep_merge_config(self, base_config: Dict, yaml_config: Dict) -> Dict:
        """Deep merge YAML config into base config"""
        
        for key, value in yaml_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                base_config[key] = self._deep_merge_config(base_config[key], value)
            else:
                base_config[key] = value
        
        return base_config
        
    def setup(self):
        """Setup all components"""
        logger.info("Setting up Autonomous Video Editor...")
        
        # Create directories
        os.makedirs(self.config['model_cache_dir'], exist_ok=True)
        os.makedirs(self.config['data_dir'], exist_ok=True)
        os.makedirs(self.config['training']['output_dir'], exist_ok=True)
        
        # Initialize components
        self.model_downloader = ModelDownloader(
            cache_dir=self.config['model_cache_dir']
        )
        
        self.dataset_downloader = DatasetAutoDownloader(
            data_dir=self.config['data_dir']
        )
        
        self.training_orchestrator = TrainingOrchestrator(
            config=self.config,
            model_downloader=self.model_downloader,
            dataset_downloader=self.dataset_downloader
        )
        
        self.effect_generator = AdvancedEffectGenerator(
            quality=self.config['effects']['quality'],
            gpu_acceleration=self.config['effects']['gpu_acceleration']
        )
        
        logger.info("Setup complete!")
        
    def download_models(self):
        """Download all required models"""
        if not self.model_downloader:
            raise RuntimeError("Model downloader not initialized. Call setup() first.")
            
        logger.info("Downloading required models...")
        
        # Download core models
        models_to_download = [
            (self.config['model']['backbone'], 'language'),
            (self.config['model']['vision_encoder'], 'vision'),
            (self.config['model']['audio_encoder'], 'audio')
        ]
        
        for model_name, model_type in models_to_download:
            try:
                logger.info(f"Downloading {model_type} model: {model_name}")
                self.model_downloader.download_model(model_name, model_type=model_type)
            except Exception as e:
                logger.error(f"Failed to download {model_name}: {e}")
                
        logger.info("Model downloads complete!")
        
    def download_datasets(self):
        """Download training datasets"""
        if not self.dataset_downloader:
            raise RuntimeError("Dataset downloader not initialized. Call setup() first.")
            
        if not self.config['datasets']['auto_download']:
            logger.info("Auto-download disabled, skipping dataset download")
            return
            
        logger.info("Downloading training datasets...")
        
        # Download datasets based on config
        if self.config['datasets']['webvid']['enabled']:
            self.dataset_downloader.download_webvid(
                num_samples=self.config['datasets']['webvid']['samples']
            )
            
        if self.config['datasets']['audioset']['enabled']:
            self.dataset_downloader.download_audioset(
                num_samples=self.config['datasets']['audioset']['samples']
            )
            
        if self.config['datasets']['activitynet']['enabled']:
            self.dataset_downloader.download_activitynet(
                num_samples=self.config['datasets']['activitynet']['samples']
            )
            
        if self.config['datasets']['tvsum']['enabled']:
            self.dataset_downloader.download_tvsum()
            
        if self.config['datasets']['summe']['enabled']:
            self.dataset_downloader.download_summe()
            
        logger.info("Dataset downloads complete!")
        
    def train_model(self):
        """Train the autonomous video editing model"""
        if not self.training_orchestrator:
            raise RuntimeError("Training orchestrator not initialized. Call setup() first.")
            
        logger.info("Starting autonomous video editor training...")
        
        try:
            # Run full training pipeline
            self.training_orchestrator.run_full_training()
            logger.info("Training completed successfully!")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
            
    def load_model(self, model_path: Optional[str] = None):
        """Load trained model for inference"""
        logger.info("Loading trained model...")
        
        try:
            # Initialize hybrid AI system
            self.hybrid_ai = HybridVideoAI(self.config)
            
            # Load trained weights if provided
            if model_path and os.path.exists(model_path):
                logger.info(f"Loading weights from {model_path}")
                checkpoint = torch.load(model_path, map_location='cpu')
                self.hybrid_ai.load_state_dict(checkpoint['model_state_dict'])
            else:
                logger.info("Using default pre-trained weights")
                
            # Initialize autonomous editor
            self.autonomous_editor = AutonomousVideoEditor(
                ai_model=self.hybrid_ai,
                effect_generator=self.effect_generator,
                config=self.config
            )
            
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
            
    def edit_video(self, video_path: str, prompt: str, output_path: str):
        """Edit a video using the autonomous editor"""
        if not self.autonomous_editor:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        logger.info(f"Editing video: {video_path}")
        logger.info(f"Prompt: {prompt}")
        
        try:
            # Perform autonomous editing
            result = self.autonomous_editor.edit_video(
                video_path=video_path,
                editing_prompt=prompt,
                output_path=output_path
            )
            
            logger.info(f"Video edited successfully: {output_path}")
            return result
            
        except Exception as e:
            logger.error(f"Video editing failed: {e}")
            raise
            
    def run_demo(self):
        """Run a demo with sample effects"""
        logger.info("Running demo mode...")
        
        # Create a sample video with effects
        import numpy as np
        import cv2
        
        # Create sample frames
        frames = []
        for i in range(60):  # 2 seconds at 30fps
            frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            # Add some pattern
            cv2.circle(frame, (640 + int(200 * np.sin(i * 0.1)), 360), 50, (255, 255, 255), -1)
            frames.append(frame)
            
        # Apply various effects
        effects_to_demo = [
            'fade_in', 'zoom_in', 'color_grade_cinematic', 
            'vintage_film', 'cyberpunk', 'dramatic_shadows'
        ]
        
        for effect_name in effects_to_demo:
            logger.info(f"Applying effect: {effect_name}")
            processed_frames = []
            
            for frame in frames:
                try:
                    processed_frame = self.effect_generator.apply_effect(frame, effect_name)
                    processed_frames.append(processed_frame)
                except Exception as e:
                    logger.warning(f"Effect {effect_name} failed: {e}")
                    processed_frames.append(frame)
                    
            # Save demo video
            output_path = f'demo_{effect_name}.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 30.0, (1280, 720))
            
            for frame in processed_frames:
                out.write(frame)
            out.release()
            
            logger.info(f"Demo video saved: {output_path}")
            
        logger.info("Demo complete!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Autonomous Video Editor')
    parser.add_argument('--setup', action='store_true', 
                       help='Setup the application (create directories, etc.)')
    parser.add_argument('--download-models', action='store_true',
                       help='Download required AI models')
    parser.add_argument('--download-data', action='store_true', 
                       help='Download training datasets')
    parser.add_argument('--train', action='store_true',
                       help='Train the autonomous video editing model')
    parser.add_argument('--load-model', type=str, default=None,
                       help='Path to trained model checkpoint')
    parser.add_argument('--edit-video', type=str, default=None,
                       help='Path to video file to edit')
    parser.add_argument('--prompt', type=str, default="Make this video more cinematic",
                       help='Editing prompt for the AI')
    parser.add_argument('--output', type=str, default='edited_video.mp4',
                       help='Output path for edited video')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo mode with sample effects')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to custom configuration file')
    
    args = parser.parse_args()
    
    # Load custom config if provided
    config = DEFAULT_CONFIG
    if args.config and os.path.exists(args.config):
        import json
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
        # Merge configs (simple merge)
        config.update(custom_config)
    
    # Initialize application
    app = AutonomousVideoEditorApp(config)
    
    try:
        if args.setup:
            app.setup()
            
        if args.download_models:
            app.setup()
            app.download_models()
            
        if args.download_data:
            app.setup()
            app.download_datasets()
            
        if args.train:
            app.setup()
            app.download_models()
            app.download_datasets()
            app.train_model()
            
        if args.demo:
            app.setup()
            app.run_demo()
            
        if args.edit_video:
            app.setup()
            app.load_model(args.load_model)
            app.edit_video(args.edit_video, args.prompt, args.output)
            
        if not any([args.setup, args.download_models, args.download_data, 
                   args.train, args.demo, args.edit_video]):
            print("No action specified. Use --help for available options.")
            
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
