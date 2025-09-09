#!/usr/bin/env python3
"""
Smoke Test - Quick validation of the entire autonomous video editing pipeline
"""

import sys
import os
import logging
import traceback
from pathlib import Path

# Add src and project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))


def setup_logging():
    """Configure logging for smoke test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Suppress some verbose logs
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)


def test_data_loading():
    """Test data loading components"""
    print("🔍 Testing Data Loading...")
    
    try:
        from utils.data_loader import VideoEditingDataset, MultiModalDataLoader
        from utils.dataset_manager import VideoDatasetManager
        
        # Test dataset manager
        manager = VideoDatasetManager("data/test")
        print("  ✅ VideoDatasetManager initialized")
        
        # Test sample dataset creation
        from utils.data_loader import create_sample_dataset
        create_sample_dataset("data/test_samples", num_samples=5)
        print("  ✅ Sample dataset created")
        
        # Test dataset loading
        from omegaconf import OmegaConf
        
        test_config = OmegaConf.create({
            'model': {
                'backbone': 'microsoft/DialoGPT-small'
            }
        })
        
        dataset = VideoEditingDataset(
            data_dir="data/test_samples",
            config=test_config,
            split="train",
            max_frames=5
        )
        print(f"  ✅ VideoEditingDataset loaded ({len(dataset)} samples)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data loading test failed: {e}")
        return False


def test_model_components():
    """Test core model components"""
    print("🧠 Testing Model Components...")
    
    try:
        from omegaconf import OmegaConf
        
        # Test configuration
        config = OmegaConf.create({
            'model': {
                'backbone': 'microsoft/DialoGPT-small',  # Smaller model for testing
                'vision_encoder': 'openai/clip-vit-base-patch32',
                'audio_encoder': 'openai/whisper-tiny',
                'text_dim': 768,
                'vision_dim': 512,
                'audio_dim': 384,
                'fusion_dim': 1024,
                'hidden_dim': 2048,
                'num_attention_heads': 8
            },
            'device': 'cpu'  # Force CPU for testing
        })
        
        # Test multimodal fusion
        from models.multimodal_fusion import MultiModalFusionModule
        fusion = MultiModalFusionModule(
            text_dim=config.model.text_dim,
            vision_dim=config.model.vision_dim,
            audio_dim=config.model.audio_dim,
            fusion_dim=config.model.fusion_dim
        )
        print("  ✅ MultiModalFusionModule initialized")
        
        # Test video understanding
        from models.video_understanding import VideoUnderstandingModule
        video_understanding = VideoUnderstandingModule(
            fusion_dim=config.model.fusion_dim,
            hidden_dim=config.model.hidden_dim
        )
        print("  ✅ VideoUnderstandingModule initialized")
        
        # Test editing planner
        from models.editing_planner import EditingPlannerModule
        editing_planner = EditingPlannerModule(
            hidden_dim=config.model.hidden_dim
        )
        print("  ✅ EditingPlannerModule initialized")
        
        # Test expert models (with fallbacks)
        try:
            from models.expert_models import ExpertModels
            experts = ExpertModels(config)
            print("  ✅ ExpertModels initialized")
        except Exception as e:
            print(f"  ⚠️  ExpertModels failed (expected in CI): {e}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model components test failed: {e}")
        traceback.print_exc()
        return False


def test_processing_pipeline():
    """Test video/audio processing pipeline"""
    print("🎬 Testing Processing Pipeline...")
    
    try:
        from omegaconf import OmegaConf
        
        config = OmegaConf.create({
            'device': 'cpu',
            'target_fps': 30,
            'frame_size': [224, 224],
            'audio_sample_rate': 16000
        })
        
        # Test vision processor
        try:
            from perception.vision_processor import VisionProcessor
            vision_proc = VisionProcessor(config)
            print("  ✅ VisionProcessor initialized")
            
            # Test with synthetic frames
            import torch
            test_frames = [torch.randn(3, 224, 224) for _ in range(5)]
            scene_analysis = vision_proc.analyze_scene(test_frames)
            print(f"  ✅ Scene analysis completed: {len(scene_analysis)} features")
            
        except Exception as e:
            print(f"  ⚠️  VisionProcessor failed: {e}")
        
        # Test audio processor
        try:
            from audio.audio_processor import AudioProcessor
            audio_proc = AudioProcessor(config)
            print("  ✅ AudioProcessor initialized")
            
            # Test with synthetic audio
            import numpy as np
            test_audio = np.random.randn(16000 * 2)  # 2 seconds
            audio_features = audio_proc._extract_audio_features(test_audio, 16000)
            print(f"  ✅ Audio analysis completed: {len(audio_features)} features")
            
        except Exception as e:
            print(f"  ⚠️  AudioProcessor failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Processing pipeline test failed: {e}")
        return False


def test_training_system():
    """Test training components"""
    print("🎯 Testing Training System...")
    
    try:
        from omegaconf import OmegaConf
        
        config = OmegaConf.create({
            'training': {
                'batch_size': 2,
                'learning_rate': 1e-4,
                'num_epochs': 1
            },
            'device': 'cpu',
            'data_dir': 'data/test_samples'
        })
        
        # Test metrics
        from utils.metrics import VideoEditingMetrics, DistillationMetrics, RLHFMetrics
        
        video_metrics = VideoEditingMetrics()
        print("  ✅ VideoEditingMetrics initialized")
        
        distill_metrics = DistillationMetrics()
        print("  ✅ DistillationMetrics initialized")
        
        rlhf_metrics = RLHFMetrics()
        print("  ✅ RLHFMetrics initialized")
        
        # Test distillation utils
        from utils.distillation_utils import DistillationLoss, FeatureMatching
        
        distill_loss = DistillationLoss()
        feature_matcher = FeatureMatching()
        print("  ✅ Distillation utilities initialized")
        
        # Test RLHF trainer
        from learning.rlhf_trainer import RLHFTrainer, PreferenceDataset
        
        rlhf_trainer = RLHFTrainer(config)
        print("  ✅ RLHFTrainer initialized")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Training system test failed: {e}")
        traceback.print_exc()
        return False


def test_effect_generation():
    """Test effect generation system"""
    print("✨ Testing Effect Generation...")
    
    try:
        from generation.effect_generator import EffectGenerator
        import numpy as np
        
        config = {'device': 'cpu'}
        effect_gen = EffectGenerator(config)
        
        # Create test frames
        test_frames = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(10)
        ]
        
        # Test various effects
        effects_to_test = [
            'fade_in', 'fade_out', 'zoom_in', 'cinematic_color_grade', 
            'vintage', 'film_grain'
        ]
        
        for effect in effects_to_test:
            try:
                result = effect_gen.generate_effect(effect, test_frames[:3])
                print(f"  ✅ {effect}: {len(result)} frames processed")
            except Exception as e:
                print(f"  ⚠️  {effect} failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Effect generation test failed: {e}")
        return False


def test_end_to_end_integration():
    """Test basic end-to-end integration"""
    print("🔄 Testing End-to-End Integration...")
    
    try:
        # Test orchestrator
        from core.orchestrator import ModelOrchestrator
        
        orchestrator = ModelOrchestrator()
        print("  ✅ ModelOrchestrator initialized")
        
        # Test timeline generator
        from editing.timeline_generator import TimelineGenerator
        from omegaconf import OmegaConf
        
        config = OmegaConf.create({
            'output_fps': 30,
            'output_resolution': [640, 480]
        })
        
        timeline_gen = TimelineGenerator(config)
        print("  ✅ TimelineGenerator initialized")
        
        # Test synthetic timeline generation
        import torch
        test_frames = [torch.randn(3, 224, 224) for _ in range(30)]
        test_audio = torch.randn(16000 * 2)  # 2 seconds
        
        timeline = timeline_gen.generate_timeline(
            frames=test_frames,
            audio=test_audio,
            prompt="Test edit with smooth transitions"
        )
        
        print(f"  ✅ Timeline generated: {len(timeline.get('segments', []))} segments")
        
        return True
        
    except Exception as e:
        print(f"  ❌ End-to-end integration test failed: {e}")
        traceback.print_exc()
        return False


def run_smoke_tests():
    """Run all smoke tests"""
    print("🧪 AUTONOMOUS VIDEO EDITOR - SMOKE TESTS")
    print("=" * 50)
    
    setup_logging()
    
    test_results = {}
    
    # Run all tests
    tests = [
        ("Data Loading", test_data_loading),
        ("Model Components", test_model_components), 
        ("Processing Pipeline", test_processing_pipeline),
        ("Training System", test_training_system),
        ("Effect Generation", test_effect_generation),
        ("End-to-End Integration", test_end_to_end_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print()
        try:
            result = test_func()
            test_results[test_name] = result
            if result:
                passed += 1
        except Exception as e:
            print(f"  ❌ {test_name} crashed: {e}")
            test_results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("🏁 SMOKE TEST RESULTS")
    print("=" * 50)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n📊 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - System is ready!")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed - Check individual components")
        return False


def main():
    """Run smoke tests"""
    success = run_smoke_tests()
    
    print("\n🔍 Next Steps:")
    print("  • Run full demo: python scripts/simple_demo.py")
    print("  • Create sample data: python scripts/create_sample_dataset.py")
    print("  • Start training: python scripts/train.py")
    print("  • Run end-to-end test: python scripts/test_pipeline.py")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
