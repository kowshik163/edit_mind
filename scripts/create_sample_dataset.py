#!/usr/bin/env python3
"""
Create Sample Dataset - Generates synthetic video/audio data for testing
"""

import os
import cv2
import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import librosa

logger = logging.getLogger(__name__)


def create_synthetic_video(output_path: str, duration: float = 10.0, fps: int = 30, 
                          width: int = 640, height: int = 480) -> str:
    """Create a synthetic video with moving patterns"""
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    total_frames = int(duration * fps)
    
    for frame_idx in range(total_frames):
        # Create dynamic frame with moving patterns
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Background gradient
        for y in range(height):
            for x in range(width):
                frame[y, x, 0] = int(128 + 127 * np.sin(frame_idx * 0.1 + x * 0.02))  # Red
                frame[y, x, 1] = int(128 + 127 * np.sin(frame_idx * 0.15 + y * 0.02))  # Green  
                frame[y, x, 2] = int(128 + 127 * np.sin(frame_idx * 0.12))  # Blue
        
        # Moving circle
        center_x = int(width/2 + 100 * np.sin(frame_idx * 0.05))
        center_y = int(height/2 + 50 * np.cos(frame_idx * 0.07))
        cv2.circle(frame, (center_x, center_y), 30, (255, 255, 255), -1)
        
        # Moving rectangle  
        rect_x = int(50 + frame_idx * 2) % (width - 100)
        rect_y = int(height * 0.7)
        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + 80, rect_y + 40), (0, 255, 255), -1)
        
        # Text overlay
        cv2.putText(frame, f"Frame {frame_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        writer.write(frame)
    
    writer.release()
    logger.info(f"Created synthetic video: {output_path}")
    return str(output_path)


def create_synthetic_audio(output_path: str, duration: float = 10.0, sample_rate: int = 16000) -> str:
    """Create synthetic audio with beats and tones"""
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Time array
    t = np.linspace(0, duration, int(duration * sample_rate), False)
    
    # Base tone (440Hz A note)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    
    # Add harmonic
    audio += 0.2 * np.sin(2 * np.pi * 880 * t)
    
    # Add beat pattern (120 BPM)
    beat_interval = 60.0 / 120  # 0.5 seconds per beat
    for beat_time in np.arange(0, duration, beat_interval):
        beat_start = int(beat_time * sample_rate)
        beat_length = int(0.1 * sample_rate)  # 100ms beat
        
        if beat_start + beat_length < len(audio):
            # Add kick drum sound
            beat_audio = 0.7 * np.sin(2 * np.pi * 60 * t[beat_start:beat_start + beat_length])
            beat_audio *= np.exp(-5 * np.arange(beat_length) / beat_length)  # Decay envelope
            audio[beat_start:beat_start + beat_length] += beat_audio
    
    # Add some variation over time  
    audio *= (1 + 0.3 * np.sin(2 * np.pi * 0.2 * t))  # Slow amplitude modulation
    
    # Normalize
    audio = np.clip(audio, -1, 1)
    
    # Save as WAV
    import soundfile as sf
    sf.write(str(output_path), audio, sample_rate)
    
    logger.info(f"Created synthetic audio: {output_path}")
    return str(output_path)


def create_sample_annotations(video_path: str, audio_path: str, output_dir: str) -> str:
    """Create sample annotations for the synthetic media with detailed parameters"""
    
    annotations = {
        "video_info": {
            "path": video_path,
            "duration": 10.0,
            "fps": 30,
            "width": 640,
            "height": 480,
            "total_frames": 300
        },
        "audio_info": {
            "path": audio_path,
            "duration": 10.0,
            "sample_rate": 16000,
            "total_samples": 160000
        },
        "editing_annotations": {
            "cuts": [2.5, 5.0, 7.5],  # Suggested cut points
            "scenes": [
                {"start": 0.0, "end": 2.5, "type": "intro"},
                {"start": 2.5, "end": 5.0, "type": "action"}, 
                {"start": 5.0, "end": 7.5, "type": "climax"},
                {"start": 7.5, "end": 10.0, "type": "outro"}
            ],
            "beats": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 
                     5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5],  # Beat timings
            
            # Effects with detailed parameters
            "effects": [
                {
                    "type": "fade_in",
                    "start": 0.0,
                    "duration": 0.5,
                    "intensity": 1.0,
                    "speed": 1.0,
                    "scale": 1.0,
                    "direction": 0.0,
                    "ease_in": 0.2,
                    "ease_out": 0.0
                },
                {
                    "type": "zoom",
                    "start": 2.5,
                    "duration": 1.0,
                    "intensity": 0.8,
                    "speed": 1.0,
                    "scale": 1.3,
                    "direction": 0.5,
                    "ease_in": 0.3,
                    "ease_out": 0.3
                },
                {
                    "type": "color_enhance",
                    "start": 5.0,
                    "duration": 2.5,
                    "intensity": 0.7,
                    "speed": 1.0,
                    "scale": 1.0,
                    "direction": 0.0,
                    "ease_in": 0.1,
                    "ease_out": 0.1
                },
                {
                    "type": "speed_ramp",
                    "start": 6.0,
                    "duration": 1.5,
                    "intensity": 0.9,
                    "speed": 1.5,
                    "scale": 1.0,
                    "direction": 0.0,
                    "ease_in": 0.4,
                    "ease_out": 0.2
                },
                {
                    "type": "fade_out",
                    "start": 9.5,
                    "duration": 0.5,
                    "intensity": 1.0,
                    "speed": 1.0,
                    "scale": 1.0,
                    "direction": 0.0,
                    "ease_in": 0.0,
                    "ease_out": 0.2
                }
            ],
            
            # Transitions with detailed parameters
            "transitions": [
                {
                    "type": "fade",
                    "start": 2.5,
                    "duration": 0.3,
                    "intensity": 0.9,
                    "direction": 0.0,
                    "smoothness": 0.7,
                    "offset": 0.0,
                    "angle": 0.0,
                    "scale": 1.0
                },
                {
                    "type": "dissolve",
                    "start": 5.0,
                    "duration": 0.5,
                    "intensity": 1.0,
                    "direction": 0.0,
                    "smoothness": 0.8,
                    "offset": 0.0,
                    "angle": 0.0,
                    "scale": 1.0
                },
                {
                    "type": "wipe",
                    "start": 7.5,
                    "duration": 0.4,
                    "intensity": 1.0,
                    "direction": 0.25,  # Left to right
                    "smoothness": 0.5,
                    "offset": 0.0,
                    "angle": 0.0,
                    "scale": 1.0
                }
            ]
        },
        "text_prompts": [
            "Create an exciting montage with beat-synchronized cuts",
            "Make a smooth cinematic edit with dramatic lighting",
            "Generate a fast-paced action sequence",
            "Create a contemplative slow-motion sequence"
        ]
    }
    
    # Save annotations
    annotation_path = Path(output_dir) / "sample_annotations.json"
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(annotation_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    logger.info(f"Created sample annotations: {annotation_path}")
    return str(annotation_path)


def create_multiple_samples(output_dir: str, num_samples: int = 10) -> List[Dict[str, Any]]:
    """Create multiple synthetic samples for training"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    samples = []
    
    for i in range(num_samples):
        sample_dir = output_path / f"sample_{i:03d}"
        sample_dir.mkdir(exist_ok=True)
        
        # Vary parameters for each sample
        duration = np.random.uniform(5.0, 15.0)  # 5-15 seconds
        
        # Create video with different properties
        video_path = sample_dir / f"video_{i:03d}.mp4"
        width = np.random.choice([480, 640, 720])
        height = int(width * 0.75)  # 4:3 aspect ratio
        
        create_synthetic_video(
            str(video_path), 
            duration=duration,
            width=width, 
            height=height
        )
        
        # Create audio
        audio_path = sample_dir / f"audio_{i:03d}.wav"
        create_synthetic_audio(str(audio_path), duration=duration)
        
        # Create annotations
        annotation_path = create_sample_annotations(
            str(video_path), 
            str(audio_path), 
            str(sample_dir)
        )
        
        sample_info = {
            "id": f"sample_{i:03d}",
            "video_path": str(video_path),
            "audio_path": str(audio_path),
            "annotation_path": annotation_path,
            "duration": duration,
            "width": width,
            "height": height
        }
        
        samples.append(sample_info)
        
        if i % 5 == 0:
            logger.info(f"Created {i+1}/{num_samples} samples")
    
    # Save sample index
    index_path = output_path / "sample_index.json"
    with open(index_path, 'w') as f:
        json.dump(samples, f, indent=2)
    
    logger.info(f"✅ Created {num_samples} synthetic samples in {output_dir}")
    logger.info(f"📋 Sample index saved to: {index_path}")
    
    return samples


def main():
    """Create sample dataset for testing"""
    logging.basicConfig(level=logging.INFO)
    
    output_dir = "data/synthetic_samples"
    
    print("🎬 Creating synthetic video editing dataset...")
    print(f"📁 Output directory: {output_dir}")
    
    # Create sample dataset
    samples = create_multiple_samples(output_dir, num_samples=20)
    
    print(f"✅ Created {len(samples)} synthetic samples")
    print(f"📹 Video files: *.mp4")
    print(f"🎵 Audio files: *.wav") 
    print(f"📋 Annotations: *.json")
    
    print("\n🚀 Sample dataset ready for training and testing!")
    print(f"Use: python scripts/train.py --data_dir {output_dir}")


if __name__ == "__main__":
    main()
