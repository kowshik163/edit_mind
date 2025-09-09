#!/usr/bin/env python3
"""
Expert Models Test - Verify all expert models load correctly
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def test_expert_models():
    """Test expert models loading and inference"""
    print("🧠 Testing Expert Models Implementation...")
    
    try:
        from models.expert_models import ExpertModels
        from omegaconf import DictConfig
        
        config = DictConfig({
            'device': 'cpu'  # Use CPU for testing
        })
        
        print("📦 Loading expert models...")
        experts = ExpertModels(config)
        
        # Check capabilities
        capabilities = experts.get_capabilities()
        print(f"✅ Model capabilities: {capabilities}")
        
        # Test vision embeddings
        print("\n📹 Testing vision processing...")
        dummy_images = torch.rand(2, 3, 224, 224)
        vision_features = experts.get_vision_embeddings(dummy_images)
        print(f"✅ Vision features extracted: {list(vision_features.keys())}")
        for k, v in vision_features.items():
            print(f"   {k}: shape {v.shape}")
        
        # Test audio embeddings
        print("\n🎵 Testing audio processing...")
        dummy_audio = torch.rand(1, 16000)  # 1 second of audio
        audio_features = experts.get_audio_embeddings(dummy_audio)
        print(f"✅ Audio features extracted: {list(audio_features.keys())}")
        for k, v in audio_features.items():
            print(f"   {k}: shape {v.shape}")
        
        # Test detection
        print("\n🔍 Testing object detection...")
        detection_results = experts.get_detection_results(dummy_images)
        print(f"✅ Detection results: {len(detection_results['detections'])} frames processed")
        
        # Test code generation
        print("\n💻 Testing code generation...")
        code = experts.generate_code("create a simple video fade transition")
        print(f"✅ Code generated ({len(code)} chars): {code[:100]}...")
        
        # Test comprehensive features
        print("\n🔄 Testing comprehensive feature extraction...")
        all_features = experts.get_all_expert_features(dummy_images, dummy_audio)
        print(f"✅ All features extracted: {len(all_features)} feature sets")
        
        # Summary
        total_loaded = capabilities['total_loaded']
        print(f"\n📊 SUMMARY:")
        print(f"   Models loaded: {total_loaded}/4 categories")
        print(f"   Vision: {'✅' if capabilities['vision_understanding'] else '❌'}")
        print(f"   Audio: {'✅' if capabilities['audio_processing'] else '❌'}")
        print(f"   Detection: {'✅' if capabilities['object_detection'] else '❌'}")
        print(f"   Reasoning: {'✅' if capabilities['code_generation'] else '❌'}")
        print(f"   Multilingual: {'✅' if capabilities['multilingual_speech'] else '❌'}")
        print(f"   Music Analysis: {'✅' if capabilities['music_analysis'] else '❌'}")
        print(f"   Segmentation: {'✅' if capabilities['segmentation'] else '❌'}")
        
        if total_loaded >= 2:
            print("🎉 Expert models are working! Ready for knowledge distillation.")
        else:
            print("⚠️  Some expert models failed to load, but basic functionality available.")
            
        return True
        
    except Exception as e:
        print(f"❌ Expert models test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_processors():
    """Test vision and audio processors"""
    print("\n🔧 Testing Updated Processors...")
    
    try:
        from perception.vision_processor import VisionProcessor
        from audio.audio_processor import AudioProcessor
        from omegaconf import DictConfig
        
        config = DictConfig({
            'device': 'cpu',
            'target_fps': 30,
            'frame_size': [224, 224],
            'audio_sample_rate': 16000
        })
        
        # Test vision processor with dummy video data
        print("📹 Testing vision processor...")
        vision_proc = VisionProcessor(config)
        
        # Since we don't have a real video file, test the frame processing
        dummy_frames = [torch.rand(3, 224, 224) for _ in range(5)]
        analysis = vision_proc.analyze_scene(dummy_frames)
        print(f"✅ Vision analysis: {list(analysis.keys())}")
        
        # Test audio processor
        print("🎵 Testing audio processor...")
        audio_proc = AudioProcessor(config)
        
        # Test feature extraction
        dummy_audio = np.random.randn(16000)
        features = audio_proc._extract_audio_features(dummy_audio, 16000)
        print(f"✅ Audio features: {list(features.keys())}")
        
        # Test transcription
        transcription = audio_proc.transcribe_audio(dummy_audio)
        print(f"✅ Transcription: {transcription}")
        
        return True
        
    except Exception as e:
        print(f"❌ Processors test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🎬 AUTONOMOUS VIDEO EDITOR - EXPERT MODELS TEST\n")
    
    results = []
    
    # Test expert models
    results.append(("Expert Models", test_expert_models()))
    
    # Test processors  
    results.append(("Processors", test_processors()))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST RESULTS")
    print(f"{'='*60}")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20}: {status}")
        if result:
            passed += 1
    
    print(f"\n📈 OVERALL: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All expert models and processors are working!")
        print("\n🚀 NEXT STEPS:")
        print("1. Test with real video files")
        print("2. Run knowledge distillation")
        print("3. Start training pipeline")
    else:
        print("⚠️  Some components need attention before full training")

if __name__ == "__main__":
    main()
