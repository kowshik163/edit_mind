#!/usr/bin/env python3
"""
End-to-End Pipeline Test for Autonomous Video Editor
Creates sample data and tests the complete pipeline
"""

import os
import sys
import torch
import numpy as np
import logging
from pathlib import Path
import json

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from omegaconf import OmegaConf


def create_minimal_config():
    """Create a minimal configuration for testing"""
    config = {
        'model': {
            'backbone': 'microsoft/DialoGPT-medium',  # Smaller model for testing
            'vision_encoder': 'openai/clip-vit-base-patch32',
            'audio_encoder': 'openai/whisper-base',
            'text_dim': 768,
            'vision_dim': 512,
            'audio_dim': 512,
            'fusion_dim': 1024,
            'hidden_dim': 1024,
            'num_attention_heads': 8
        },
        'training': {
            'batch_size': 2,
            'learning_rate': 1e-4,
            'max_epochs': 2,
            'warmup_steps': 100,
            'gradient_clip_val': 1.0,
            'accumulate_grad_batches': 1
        },
        'data': {
            'max_frames': 16,
            'max_audio_length': 32000,  # 2 seconds at 16kHz
            'num_workers': 0
        },
        'data_dir': 'data/test_dataset',
        'output_dir': 'output/test',
        'checkpoints_dir': 'checkpoints/test',
        'device': 'cpu'  # Force CPU for testing
    }
    
    return OmegaConf.create(config)


def create_test_video_data():
    """Create synthetic video data for testing"""
    import cv2
    
    data_dir = Path('data/test_dataset')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a simple synthetic video
    video_path = data_dir / 'test_video_0.mp4'
    
    if not video_path.exists():
        print("📹 Creating synthetic test video...")
        
        # Video properties
        width, height = 224, 224
        fps = 30
        duration = 3  # seconds
        total_frames = fps * duration
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        
        # Generate frames with moving colored rectangle
        for frame_num in range(total_frames):
            # Create a frame with a moving colored rectangle
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Background color changes over time
            bg_intensity = int(50 + 50 * np.sin(frame_num / 10))
            frame[:] = (bg_intensity, bg_intensity//2, bg_intensity//3)
            
            # Moving rectangle
            rect_size = 50
            center_x = int(width//2 + 50 * np.cos(frame_num / 15))
            center_y = int(height//2 + 30 * np.sin(frame_num / 10))
            
            x1 = max(0, center_x - rect_size//2)
            y1 = max(0, center_y - rect_size//2)
            x2 = min(width, center_x + rect_size//2)
            y2 = min(height, center_y + rect_size//2)
            
            # Rectangle color changes
            rect_color = (
                int(255 * np.sin(frame_num / 20) ** 2),
                int(255 * np.cos(frame_num / 15) ** 2),
                int(255 * np.sin(frame_num / 25) ** 2)
            )
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), rect_color, -1)
            
            out.write(frame)
        
        out.release()
        print(f"✅ Created test video: {video_path}")
    
    return str(video_path)


def create_test_audio_data():
    """Create synthetic audio data for testing"""
    import soundfile as sf
    
    data_dir = Path('data/test_dataset')
    audio_path = data_dir / 'test_audio_0.wav'
    
    if not audio_path.exists():
        print("🎵 Creating synthetic test audio...")
        
        # Generate 3 seconds of audio with varying tones
        duration = 3
        sample_rate = 16000
        t = np.linspace(0, duration, duration * sample_rate, False)
        
        # Generate audio with multiple frequency components
        audio = (
            0.3 * np.sin(2 * np.pi * 440 * t) +  # A4 note
            0.2 * np.sin(2 * np.pi * 660 * t) +  # E5 note
            0.1 * np.sin(2 * np.pi * 880 * t) +  # A5 note
            0.1 * np.random.normal(0, 0.1, len(t))  # Background noise
        )
        
        # Apply envelope
        envelope = np.exp(-t / 2)  # Decay envelope
        audio *= envelope
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        # Save audio
        sf.write(audio_path, audio, sample_rate)
        print(f"✅ Created test audio: {audio_path}")
    
    return str(audio_path)


def create_test_dataset():
    """Create a complete test dataset"""
    print("🗂️  Creating test dataset...")
    
    data_dir = Path('data/test_dataset')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test media files
    video_path = create_test_video_data()
    audio_path = create_test_audio_data()
    
    # Create dataset index
    for split in ['train', 'val']:
        samples = []
        num_samples = 5 if split == 'train' else 2
        
        for i in range(num_samples):
            samples.append({
                'video_id': f'{split}_video_{i}',
                'video_path': video_path,  # Reuse the same video
                'audio_path': audio_path,  # Reuse the same audio
                'prompt': f"Create a {['cinematic', 'dynamic', 'artistic'][i % 3]} edit with smooth transitions",
                'target_timeline': {
                    'cuts': [1.0, 2.0, 2.5],
                    'transitions': ['fade', 'cut', 'dissolve'],
                    'effects': ['color_grade', 'zoom', 'speed_ramp']
                }
            })
        
        index_path = data_dir / f'{split}_index.json'
        with open(index_path, 'w') as f:
            json.dump(samples, f, indent=2)
        
        print(f"✅ Created {split} index with {num_samples} samples")
    
    print("✅ Test dataset created successfully!")
    return str(data_dir)


def test_data_loading():
    """Test data loading pipeline"""
    print("\n📊 Testing data loading...")
    
    try:
        from utils.data_loader import MultiModalDataLoader
        
        config = create_minimal_config()
        data_dir = create_test_dataset()
        
        # Create data loader
        data_loader = MultiModalDataLoader(config)
        train_loader = data_loader.get_train_loader(data_dir)
        
        # Test loading a batch
        batch = next(iter(train_loader))
        
        print(f"✅ Batch loaded successfully!")
        print(f"   Video frames shape: {batch['video_frames'].shape}")
        print(f"   Audio features shape: {batch['audio_features'].shape}")
        print(f"   Input IDs shape: {batch['input_ids'].shape}")
        print(f"   Timeline targets shape: {batch['timeline_targets'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False


def test_model_initialization():
    """Test model initialization"""
    print("\n🧠 Testing model initialization...")
    
    try:
        from core.hybrid_ai import HybridVideoAI
        
        config = create_minimal_config()
        
        print("   Initializing HybridVideoAI...")
        model = HybridVideoAI(config)
        
        print(f"✅ Model initialized successfully!")
        print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model initialization test failed: {e}")
        return False


def test_forward_pass():
    """Test forward pass through the model"""
    print("\n🔄 Testing forward pass...")
    
    try:
        from core.hybrid_ai import HybridVideoAI
        
        config = create_minimal_config()
        model = HybridVideoAI(config)
        
        # Create dummy inputs
        batch_size = 2
        video_frames = torch.randn(batch_size, 16, 3, 224, 224)  # 16 frames
        audio_features = torch.randn(batch_size, 32000)  # 2 seconds of audio
        text_input_ids = torch.randint(0, 1000, (batch_size, 64))  # 64 tokens
        text_attention_mask = torch.ones(batch_size, 64)
        
        print("   Running forward pass...")
        with torch.no_grad():
            outputs = model.forward(
                video_frames=video_frames,
                audio_features=audio_features,
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                return_timeline=True
            )
        
        print(f"✅ Forward pass successful!")
        print(f"   Output keys: {list(outputs.keys())}")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key} shape: {value.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass test failed: {e}")
        import traceback
        print(f"   Full error: {traceback.format_exc()}")
        return False


def test_vision_processor():
    """Test vision processor"""
    print("\n👁️  Testing vision processor...")
    
    try:
        from perception.vision_processor import VisionProcessor
        
        config = create_minimal_config()
        processor = VisionProcessor(config)
        
        # Test with synthetic video
        video_path = create_test_video_data()
        
        print("   Loading and processing video...")
        result = processor.load_video(video_path)
        
        print(f"✅ Vision processor test successful!")
        print(f"   Loaded {result['num_frames']} frames")
        print(f"   Video duration: {result['duration']:.2f} seconds")
        print(f"   Frames tensor shape: {result['frames'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Vision processor test failed: {e}")
        return False


def test_audio_processor():
    """Test audio processor"""
    print("\n🎵 Testing audio processor...")
    
    try:
        from audio.audio_processor import AudioProcessor
        
        config = create_minimal_config()
        processor = AudioProcessor(config)
        
        # Test with synthetic audio
        audio_path = create_test_audio_data()
        
        print("   Loading and processing audio...")
        result = processor.load_audio(f"data/test_dataset/test_video_0.mp4")  # Extract from video
        
        print(f"✅ Audio processor test successful!")
        print(f"   Audio duration: {result['duration']:.2f} seconds")
        print(f"   Audio tensor shape: {result['audio'].shape}")
        print(f"   Number of features: {len(result['features'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Audio processor test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🎬 Autonomous Video Editor - Pipeline Test\n")
    
    # Configure logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise during testing
        format='%(levelname)s: %(message)s'
    )
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Model Initialization", test_model_initialization),
        ("Vision Processor", test_vision_processor),
        ("Audio Processor", test_audio_processor),
        ("Forward Pass", test_forward_pass),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<25}: {status}")
    
    print(f"\n📈 OVERALL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! The pipeline is working correctly.")
        print("\n🚀 Next steps:")
        print("   1. Run: python scripts/train.py --phase pretraining --debug")
        print("   2. Test with real video files")
        print("   3. Setup larger datasets for production training")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please review the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
