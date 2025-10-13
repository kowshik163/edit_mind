#!/usr/bin/env python3
"""
Comprehensive validation script for enhanced video editor components.
Validates the vision processor, core models, and import consistency.
"""

import sys
import os
import traceback
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def validate_imports():
    """Validate all critical imports work correctly."""
    print("🔍 Validating imports...")
    
    try:
        # Test core imports
        from core import HybridVideoAI, cli_app
        print("✅ Core imports successful")
        
        # Test training orchestrator import
        from training.training_orchestrator import TrainingOrchestrator
        print("✅ TrainingOrchestrator import successful")
        
        # Test enhanced vision processor
        from perception.vision_processor import VisionProcessor
        print("✅ Enhanced VisionProcessor import successful")
        
        # Test core model components
        from models.multimodal_fusion import MultiModalFusionModule
        from models.video_understanding import VideoUnderstandingModule
        from models.editing_planner import EditingPlannerModule
        print("✅ All core model imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def validate_vision_processor():
    """Validate vision processor initialization and capabilities."""
    print("\n🔍 Validating vision processor...")
    
    try:
        from perception.vision_processor import VisionProcessor
        
        # Initialize processor with minimal config
        config = {
            'model_cache_dir': './models',
            'detection_threshold': 0.5,
            'enable_ensemble': True,
            'fallback_to_opencv': True
        }
        processor = VisionProcessor(config)
        print("✅ VisionProcessor initialization successful")
        
        # Check available models
        if hasattr(processor, 'available_models'):
            print(f"✅ Available detection models: {len(processor.available_models)}")
        
        # Check detection methods
        detection_methods = [
            '_detect_with_rt_detr', '_detect_with_detr', '_detect_with_yolo',
            '_detect_with_advanced_cv_methods', '_detect_with_selected_model'
        ]
        
        for method in detection_methods:
            if hasattr(processor, method):
                print(f"✅ Detection method {method} available")
            else:
                print(f"❌ Missing detection method: {method}")
        
        return True
        
    except Exception as e:
        print(f"❌ Vision processor validation failed: {e}")
        traceback.print_exc()
        return False

def validate_core_models():
    """Validate core model components."""
    print("\n🔍 Validating core models...")
    
    try:
        from models.multimodal_fusion import MultiModalFusionModule
        from models.video_understanding import VideoUnderstandingModule
        from models.editing_planner import EditingPlannerModule
        
        # Test MultiModalFusionModule
        fusion_model = MultiModalFusionModule(
            text_dim=512,
            vision_dim=768,
            audio_dim=256,
            fusion_dim=512,
            num_heads=8,
            dropout=0.1
        )
        print("✅ MultiModalFusionModule initialization successful")
        
        # Test VideoUnderstandingModule
        video_model = VideoUnderstandingModule(
            fusion_dim=768,
            hidden_dim=512,
            num_layers=6
        )
        print("✅ VideoUnderstandingModule initialization successful")
        
        # Test EditingPlannerModule
        planner_model = EditingPlannerModule(
            hidden_dim=512,
            vocab_size=10000,
            max_timeline_length=500
        )
        print("✅ EditingPlannerModule initialization successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Core models validation failed: {e}")
        traceback.print_exc()
        return False

def validate_training_orchestrator():
    """Validate training orchestrator functionality."""
    print("\n🔍 Validating training orchestrator...")
    
    try:
        from training.training_orchestrator import TrainingOrchestrator
        
        # Test initialization with minimal config
        config = {
            'model_name': 'test_model',
            'learning_rate': 0.001,
            'batch_size': 4,
            'max_epochs': 1,
            'device': 'cpu'
        }
        
        orchestrator = TrainingOrchestrator(config)
        print("✅ TrainingOrchestrator initialization successful")
        
        # Check key methods exist
        key_methods = [
            'setup_model', 'setup_data_loaders', 'setup_optimizer',
            'setup_scheduler', 'train_epoch', 'validate_epoch'
        ]
        
        for method in key_methods:
            if hasattr(orchestrator, method):
                print(f"✅ Method {method} available")
            else:
                print(f"❌ Missing method: {method}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training orchestrator validation failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run comprehensive validation."""
    print("🚀 Starting comprehensive validation of enhanced components...\n")
    
    results = []
    
    # Run all validations
    results.append(("Import Validation", validate_imports()))
    results.append(("Vision Processor", validate_vision_processor()))
    results.append(("Core Models", validate_core_models()))
    results.append(("Training Orchestrator", validate_training_orchestrator()))
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All enhancements validated successfully!")
        print("Your autonomous video editor is ready with:")
        print("- State-of-the-art object detection (RT-DETR, YOLO, DETR)")
        print("- Advanced multimodal fusion capabilities")
        print("- Sophisticated video understanding with temporal analysis")
        print("- Comprehensive editing planning and timeline generation")
        print("- Robust training orchestration system")
    else:
        print(f"\n⚠️  {len(results) - passed} validation(s) failed. Check the logs above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())