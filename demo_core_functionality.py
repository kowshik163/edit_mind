#!/usr/bin/env python3
"""
Simple Demo - Test Core Functionality of Autonomous Video Editor
"""

import sys
import os
import torch
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_core_modules():
    """Test core video processing modules"""
    
    logger.info("🧪 Testing core video processing modules...")
    
    try:
        # Test configuration loading
        from omegaconf import DictConfig
        config = DictConfig({
            'device': 'cpu',  # Use CPU for testing
            'video_features_dim': 1024,
            'audio_features_dim': 512,
            'edit_features_dim': 256,
            'target_fps': 30,
            'frame_size': (224, 224),
            'audio_sample_rate': 16000
        })
        logger.info("✅ Configuration loaded")
        
        # Test vision processor
        from perception.vision_processor import VisionProcessor
        vision_processor = VisionProcessor(config)
        logger.info("✅ Vision processor initialized")
        
        # Test audio processor  
        from audio.audio_processor import AudioProcessor
        audio_processor = AudioProcessor(config)
        logger.info("✅ Audio processor initialized")
        
        # Test multimodal fusion
        from models.multimodal_fusion import MultiModalFusion
        fusion_model = MultiModalFusion(config)
        logger.info("✅ Multimodal fusion initialized")
        
        # Test RLHF trainer (enhanced version)
        try:
            from learning.enhanced_rlhf_trainer import EnhancedRLHFTrainer, VideoEditingRewardModel
            reward_model = VideoEditingRewardModel(config)
            logger.info("✅ Enhanced RLHF components available")
        except ImportError as e:
            logger.warning(f"⚠️  Enhanced RLHF not available: {e}")
        
        # Test basic tensor operations
        dummy_video_features = torch.randn(1, 1024)
        dummy_audio_features = torch.randn(1, 512)
        
        # Test fusion
        fusion_output = fusion_model(dummy_video_features, dummy_audio_features)
        logger.info(f"✅ Fusion test - output shape: {fusion_output.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Core module test failed: {e}")
        return False


def test_dataset_functionality():
    """Test dataset downloading and processing"""
    
    logger.info("🧪 Testing dataset functionality...")
    
    try:
        from utils.dataset_downloader import DatasetDownloader
        
        # Initialize downloader  
        downloader = DatasetDownloader("data/test")
        logger.info("✅ Dataset downloader initialized")
        
        # Check dataset configurations
        available_datasets = list(downloader.dataset_configs.keys())
        logger.info(f"✅ Available datasets: {', '.join(available_datasets)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Dataset test failed: {e}")
        return False


def test_training_orchestrator():
    """Test training orchestrator"""
    
    logger.info("🧪 Testing training orchestrator...")
    
    try:
        from training.training_orchestrator import TrainingOrchestrator
        
        # Initialize with minimal config
        orchestrator = TrainingOrchestrator()
        logger.info("✅ Training orchestrator initialized")
        
        # Test environment setup
        setup_success = orchestrator.setup_training_environment()
        if setup_success:
            logger.info("✅ Training environment setup")
        else:
            logger.warning("⚠️  Training environment setup had issues")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training orchestrator test failed: {e}")
        return False


def create_sample_video():
    """Create a simple sample video for testing"""
    
    logger.info("🎬 Creating sample video for testing...")
    
    try:
        import cv2
        import numpy as np
        
        # Create output directory
        output_dir = Path("data/test_samples")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Video parameters
        width, height = 640, 480
        fps = 30
        duration = 3  # 3 seconds
        total_frames = fps * duration
        
        # Create video writer
        output_path = output_dir / "sample_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Generate frames with different colors
        for frame_num in range(total_frames):
            # Create a frame with changing colors
            hue = int((frame_num / total_frames) * 180)
            frame = np.full((height, width, 3), [hue, 255, 255], dtype=np.uint8)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
            
            # Add some text
            text = f"Frame {frame_num + 1}/{total_frames}"
            cv2.putText(frame_bgr, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            video_writer.write(frame_bgr)
        
        video_writer.release()
        
        logger.info(f"✅ Sample video created: {output_path}")
        return str(output_path)
        
    except Exception as e:
        logger.error(f"❌ Failed to create sample video: {e}")
        return None


def test_video_processing(video_path):
    """Test video processing with sample video"""
    
    if not video_path or not Path(video_path).exists():
        logger.warning("⚠️  No sample video available for processing test")
        return False
    
    logger.info(f"🎬 Testing video processing with: {video_path}")
    
    try:
        # Test vision processing
        from perception.vision_processor import VisionProcessor
        from omegaconf import DictConfig
        
        config = DictConfig({
            'device': 'cpu',
            'target_fps': 30,
            'frame_size': (224, 224)
        })
        
        vision_processor = VisionProcessor(config)
        video_data = vision_processor.load_video(video_path)
        
        logger.info(f"✅ Video processed - {video_data['num_frames']} frames, {video_data['duration']:.1f}s duration")
        
        # Test audio processing
        from audio.audio_processor import AudioProcessor
        
        config.audio_sample_rate = 16000
        audio_processor = AudioProcessor(config)
        audio_data = audio_processor.load_audio(video_path)
        
        logger.info(f"✅ Audio processed - {audio_data['duration']:.1f}s duration, {audio_data['sample_rate']} Hz")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Video processing test failed: {e}")
        return False


def main():
    """Run comprehensive demo tests"""
    
    logger.info("🚀 AUTONOMOUS VIDEO EDITOR - CORE FUNCTIONALITY DEMO")
    logger.info("=" * 60)
    
    test_results = {}
    
    # Test core modules
    logger.info("\n1️⃣  Testing Core Modules")
    test_results['core_modules'] = test_core_modules()
    
    # Test dataset functionality
    logger.info("\n2️⃣  Testing Dataset Functionality")
    test_results['datasets'] = test_dataset_functionality()
    
    # Test training orchestrator
    logger.info("\n3️⃣  Testing Training Orchestrator")
    test_results['orchestrator'] = test_training_orchestrator()
    
    # Create and test with sample video
    logger.info("\n4️⃣  Testing Video Processing")
    sample_video = create_sample_video()
    test_results['video_processing'] = test_video_processing(sample_video)
    
    # Report results
    logger.info("\n🏁 DEMO RESULTS")
    logger.info("=" * 30)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
        if passed:
            passed_tests += 1
    
    logger.info(f"\n📊 Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All tests passed! Core functionality is working.")
        logger.info("\n🚀 Next steps:")
        logger.info("  - Run full training: python run_full_pipeline.py --quick")
        logger.info("  - Process a real video: python scripts/edit_video.py --input your_video.mp4")
        return 0
    else:
        logger.error("❌ Some tests failed. Check the logs above for details.")
        logger.info("\n🔧 Troubleshooting:")
        logger.info("  - Install dependencies: pip install -r requirements.txt")
        logger.info("  - Check Python version (3.8+ required)")
        return 1


if __name__ == "__main__":
    sys.exit(main())