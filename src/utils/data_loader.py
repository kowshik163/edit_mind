"""
Data loading utilities for multimodal video editing training
"""

import torch
import torch.utils.data as data
import numpy as np
import json
import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from transformers import AutoTokenizer
import cv2
import librosa
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class VideoEditingDataset(data.Dataset):
    """Dataset for video editing training with multimodal inputs"""
    
    def __init__(self, 
                 data_dir: str,
                 config: DictConfig,
                 split: str = "train",
                 max_frames: int = 32,
                 max_audio_length: int = 160000):  # 10 seconds at 16kHz
        
        self.data_dir = Path(data_dir)
        self.config = config
        self.split = split
        self.max_frames = max_frames
        self.max_audio_length = max_audio_length
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.backbone)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load dataset index
        self.samples = self._load_dataset_index()
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_dataset_index(self) -> List[Dict[str, Any]]:
        """Load dataset index file"""
        index_file = self.data_dir / f"{self.split}_index.json"
        
        if not index_file.exists():
            logger.warning(f"Index file {index_file} not found, creating dummy data")
            return self._create_dummy_samples()
        
        with open(index_file, 'r') as f:
            samples = json.load(f)
        
        return samples
    
    def _create_dummy_samples(self) -> List[Dict[str, Any]]:
        """Create dummy samples for testing"""
        samples = []
        for i in range(100):  # 100 dummy samples
            samples.append({
                'video_id': f'dummy_video_{i}',
                'video_path': f'dummy_video_{i}.mp4',
                'audio_path': f'dummy_audio_{i}.wav',
                'prompt': f"Edit this video to create a {['cinematic', 'AMV-style', 'TikTok-style'][i % 3]} video",
                'target_timeline': {
                    'cuts': [2.5, 5.0, 7.5],
                    'transitions': ['fade', 'cut', 'dissolve'],
                    'effects': ['color_grade', 'speed_ramp', 'zoom']
                }
            })
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training sample"""
        sample = self.samples[idx]
        
        try:
            # Load video frames
            video_frames = self._load_video_frames(sample.get('video_path', ''))
            
            # Load audio features
            audio_features = self._load_audio_features(sample.get('audio_path', ''))
            
            # Tokenize text prompt
            prompt = sample.get('prompt', '')
            text_inputs = self.tokenizer(
                prompt,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=512
            )
            
            # Generate timeline targets
            timeline_targets = self._process_timeline_targets(sample.get('target_timeline', {}))
            
            return {
                'video_frames': video_frames,
                'audio_features': audio_features,
                'input_ids': text_inputs.input_ids.squeeze(0),
                'attention_mask': text_inputs.attention_mask.squeeze(0),
                'timeline_targets': timeline_targets,
                'metadata': {
                    'video_id': sample.get('video_id', ''),
                    'prompt': prompt
                }
            }
            
        except Exception as e:
            logger.warning(f"Error loading sample {idx}: {e}")
            return self._get_dummy_sample()
    
    def _load_video_frames(self, video_path: str) -> torch.Tensor:
        """Load and process video frames"""
        if not video_path or not os.path.exists(video_path):
            # Return dummy frames
            return torch.randn(self.max_frames, 3, 224, 224)
        
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            # Sample frames uniformly
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = np.linspace(0, frame_count-1, self.max_frames, dtype=int)
            
            for i, frame_idx in enumerate(indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Convert BGR to RGB and resize
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (224, 224))
                    
                    # Convert to tensor and normalize
                    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                    frames.append(frame_tensor)
                else:
                    # Duplicate last frame if video is shorter
                    if frames:
                        frames.append(frames[-1].clone())
                    else:
                        frames.append(torch.zeros(3, 224, 224))
            
            cap.release()
            return torch.stack(frames)
            
        except Exception as e:
            logger.warning(f"Error loading video {video_path}: {e}")
            return torch.randn(self.max_frames, 3, 224, 224)
    
    def _load_audio_features(self, audio_path: str) -> torch.Tensor:
        """Load and process audio features"""
        if not audio_path or not os.path.exists(audio_path):
            # Return dummy audio
            return torch.randn(self.max_audio_length)
        
        try:
            # Load audio with librosa
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Pad or truncate to fixed length
            if len(audio) > self.max_audio_length:
                audio = audio[:self.max_audio_length]
            else:
                audio = np.pad(audio, (0, self.max_audio_length - len(audio)))
            
            return torch.from_numpy(audio).float()
            
        except Exception as e:
            logger.warning(f"Error loading audio {audio_path}: {e}")
            return torch.randn(self.max_audio_length)
    
    def _process_timeline_targets(self, timeline_data: Dict) -> torch.Tensor:
        """Process timeline targets into tensor format"""
        # For now, create a simple binary timeline (cut/no-cut for each frame)
        timeline = torch.zeros(self.max_frames, dtype=torch.long)
        
        if 'cuts' in timeline_data:
            # Assume cuts are in seconds, convert to frame indices
            fps = 30  # Default FPS
            cut_frames = [int(cut_time * fps * self.max_frames / (self.max_frames / fps)) 
                         for cut_time in timeline_data['cuts']]
            
            for cut_frame in cut_frames:
                if 0 <= cut_frame < self.max_frames:
                    timeline[cut_frame] = 1  # Mark as cut
        
        return timeline
    
    def _get_dummy_sample(self) -> Dict[str, torch.Tensor]:
        """Return a dummy sample when loading fails"""
        return {
            'video_frames': torch.randn(self.max_frames, 3, 224, 224),
            'audio_features': torch.randn(self.max_audio_length),
            'input_ids': torch.zeros(512, dtype=torch.long),
            'attention_mask': torch.ones(512, dtype=torch.long),
            'timeline_targets': torch.zeros(self.max_frames, dtype=torch.long),
            'metadata': {
                'video_id': 'dummy',
                'prompt': 'Create a video edit'
            }
        }


class MultiModalDataLoader:
    """DataLoader wrapper for multimodal video editing"""
    
    def __init__(self, config: DictConfig):
        self.config = config
    
    def get_train_loader(self, data_dir: str) -> data.DataLoader:
        """Get training data loader"""
        dataset = VideoEditingDataset(
            data_dir=data_dir,
            config=self.config,
            split="train",
            max_frames=self.config.data.get('max_frames', 32),
            max_audio_length=self.config.data.get('max_audio_length', 160000)
        )
        
        return data.DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.get('num_workers', 4),
            pin_memory=True,
            drop_last=True
        )
    
    def get_val_loader(self, data_dir: str) -> data.DataLoader:
        """Get validation data loader"""
        dataset = VideoEditingDataset(
            data_dir=data_dir,
            config=self.config,
            split="val",
            max_frames=self.config.data.get('max_frames', 32),
            max_audio_length=self.config.data.get('max_audio_length', 160000)
        )
        
        return data.DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.data.get('num_workers', 4),
            pin_memory=True
        )


# Helper functions for dataset preparation
def prepare_webvid_dataset(data_dir: str, output_dir: str):
    """Prepare WebVid dataset for training"""
    logger.info(f"Preparing WebVid dataset from {data_dir} to {output_dir}")
    
    import json
    import shutil
    from pathlib import Path
    
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Look for WebVid metadata files
        webvid_files = list(data_path.glob("*webvid*.json")) + list(data_path.glob("*webvid*.csv"))
        
        if not webvid_files:
            logger.warning("No WebVid metadata files found, creating synthetic dataset")
            return create_sample_dataset(output_dir, num_samples=1000)
        
        # Process found WebVid files
        processed_samples = []
        for file in webvid_files[:5]:  # Limit to first 5 files for demo
            try:
                if file.suffix == '.json':
                    with open(file, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            processed_samples.extend(data[:200])  # Limit samples
                        else:
                            processed_samples.append(data)
                logger.info(f"Processed {len(processed_samples)} samples from {file.name}")
            except Exception as e:
                logger.warning(f"Could not process {file}: {e}")
        
        # Save processed dataset
        if processed_samples:
            output_file = output_path / "webvid_processed.json"
            with open(output_file, 'w') as f:
                json.dump(processed_samples, f, indent=2)
            logger.info(f"Saved {len(processed_samples)} WebVid samples to {output_file}")
        else:
            logger.warning("No WebVid samples processed, creating synthetic dataset")
            return create_sample_dataset(output_dir, num_samples=1000)
            
    except Exception as e:
        logger.error(f"WebVid dataset preparation failed: {e}")
        return create_sample_dataset(output_dir, num_samples=1000)


def prepare_activitynet_dataset(data_dir: str, output_dir: str):
    """Prepare ActivityNet dataset for training"""
    logger.info(f"Preparing ActivityNet dataset from {data_dir} to {output_dir}")
    
    import json
    from pathlib import Path
    
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Look for ActivityNet annotation files
        annotation_files = list(data_path.glob("*activity*.json")) + list(data_path.glob("*ActivityNet*.json"))
        
        if not annotation_files:
            logger.warning("No ActivityNet files found, creating synthetic dataset")
            return create_sample_dataset(output_dir, num_samples=500)
        
        processed_activities = []
        for file in annotation_files[:3]:  # Limit files
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    
                    # Extract activity information
                    if 'database' in data:  # ActivityNet format
                        for video_id, video_info in data['database'].items():
                            activity = {
                                'video_id': video_id,
                                'duration': video_info.get('duration', 30.0),
                                'annotations': video_info.get('annotations', []),
                                'url': video_info.get('url', ''),
                                'subset': video_info.get('subset', 'training')
                            }
                            processed_activities.append(activity)
                            
                            if len(processed_activities) >= 1000:  # Limit samples
                                break
                    
                logger.info(f"Processed {len(processed_activities)} activities from {file.name}")
            except Exception as e:
                logger.warning(f"Could not process {file}: {e}")
        
        # Save processed dataset
        if processed_activities:
            output_file = output_path / "activitynet_processed.json"
            with open(output_file, 'w') as f:
                json.dump(processed_activities, f, indent=2)
            logger.info(f"Saved {len(processed_activities)} ActivityNet samples to {output_file}")
        else:
            logger.warning("No ActivityNet samples processed, creating synthetic dataset")
            return create_sample_dataset(output_dir, num_samples=500)
            
    except Exception as e:
        logger.error(f"ActivityNet dataset preparation failed: {e}")
        return create_sample_dataset(output_dir, num_samples=500)


def create_sample_dataset(output_dir: str, num_samples: int = 100):
    """Create a small sample dataset for testing"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create dummy index files
    for split in ['train', 'val']:
        samples = []
        split_size = int(num_samples * 0.8) if split == 'train' else int(num_samples * 0.2)
        
        for i in range(split_size):
            samples.append({
                'video_id': f'{split}_video_{i}',
                'video_path': f'{split}_video_{i}.mp4',  # These won't exist but will use dummy data
                'audio_path': f'{split}_audio_{i}.wav',
                'prompt': f"Edit this video to create a {['cinematic', 'AMV-style', 'TikTok-style'][i % 3]} video with smooth transitions and engaging pacing.",
                'target_timeline': {
                    'cuts': [2.0 + i*0.5, 4.0 + i*0.5, 6.0 + i*0.5],
                    'transitions': ['fade', 'cut', 'dissolve'],
                    'effects': ['color_grade', 'speed_ramp', 'zoom']
                }
            })
        
        with open(output_path / f'{split}_index.json', 'w') as f:
            json.dump(samples, f, indent=2)
    
    logger.info(f"Created sample dataset with {num_samples} samples in {output_dir}")


if __name__ == "__main__":
    # Create a sample dataset for testing
    create_sample_dataset("data/sample_dataset", num_samples=50)
