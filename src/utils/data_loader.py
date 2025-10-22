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
                 datasets: List[str] = None,
                 max_frames: int = 32,
                 frame_sample_rate: int = 2,
                 template_pairs: List[Dict] = None,
                 split: str = "train",
                 max_audio_length: int = 160000):  # 10 seconds at 16kHz
        
        self.data_dir = Path(data_dir)
        self.datasets = datasets or ['webvid', 'audioset']
        self.split = split
        self.max_frames = max_frames
        self.frame_sample_rate = frame_sample_rate
        self.max_audio_length = max_audio_length
        self.template_pairs = template_pairs or []
        
        # Initialize tokenizer (use a simple default for now)
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except:
            self.tokenizer = None
        
        # Load dataset samples
        self.samples = self._load_all_samples()
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split (including {len(self.template_pairs)} template pairs)")
    
    def _load_all_samples(self) -> List[Dict[str, Any]]:
        """Load samples from all specified datasets + template pairs + synthetic data"""
        all_samples = []
        
        # 1. Load traditional dataset samples
        for dataset_name in self.datasets:
            if dataset_name == 'synthetic':
                # Load synthetic data
                synthetic_samples = self._load_synthetic_samples()
                all_samples.extend(synthetic_samples)
            else:
                # Load regular dataset
                dataset_samples = self._load_dataset_samples(dataset_name)
                all_samples.extend(dataset_samples)
        
        # 2. Load template-based pairs
        template_samples = self._load_template_samples()
        all_samples.extend(template_samples)
        
        # 3. Create dummy samples if nothing else is available
        if not all_samples:
            logger.warning("No dataset samples found, creating dummy data")
            all_samples = self._create_dummy_samples()
        
        return all_samples
    
    def _load_dataset_samples(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Load samples from a specific dataset"""
        index_file = self.data_dir / dataset_name / f"{self.split}_index.json"
        
        if index_file.exists():
            with open(index_file, 'r') as f:
                samples = json.load(f)
            logger.info(f"Loaded {len(samples)} samples from {dataset_name}")
            return samples
        else:
            logger.warning(f"No index file found for {dataset_name}")
            return []
    
    def _load_synthetic_samples(self) -> List[Dict[str, Any]]:
        """Load synthetic training data"""
        synthetic_metadata_file = Path("data/synthetic/synthetic_dataset_metadata.json")
        
        if not synthetic_metadata_file.exists():
            logger.warning("No synthetic dataset metadata found")
            return []
        
        try:
            with open(synthetic_metadata_file, 'r') as f:
                metadata = json.load(f)
            
            samples = []
            for sample in metadata.get('samples', []):
                # Convert synthetic sample to training format
                training_sample = {
                    'video_id': sample['id'],
                    'video_path': sample['raw_footage_path'],
                    'edited_video_path': sample['edited_video_path'],
                    'prompt': f"Apply {sample['metadata'].get('generation_method', 'editing')} style",
                    'target_timeline': {
                        'instructions': sample['editing_instructions'],
                        'style': sample['metadata'].get('style', 'general')
                    },
                    'source': 'synthetic'
                }
                samples.append(training_sample)
            
            logger.info(f"Loaded {len(samples)} synthetic samples")
            return samples
            
        except Exception as e:
            logger.error(f"Error loading synthetic samples: {e}")
            return []
    
    def _load_template_samples(self) -> List[Dict[str, Any]]:
        """Load template-based training pairs"""
        samples = []
        
        for i, pair in enumerate(self.template_pairs):
            training_sample = {
                'video_id': pair.get('id', f'template_pair_{i}'),
                'video_path': pair.get('raw_footage', ''),
                'template_path': pair.get('template', ''),
                'expected_output': pair.get('expected_output', ''),
                'prompt': f"Apply template editing style",
                'target_timeline': pair.get('template_metadata', {}),
                'source': 'template'
            }
            samples.append(training_sample)
        
        if samples:
            logger.info(f"Loaded {len(samples)} template-based samples")
        
        return samples
    
    def collate_fn(self, batch):
        """Collate function for DataLoader"""
        try:
            # Simple collation - stack tensors where possible
            collated = {}
            
            # Handle video frames
            if 'video_frames' in batch[0]:
                video_frames = [item['video_frames'] for item in batch]
                collated['video_frames'] = torch.stack(video_frames)
            
            # Handle audio features
            if 'audio_features' in batch[0]:
                audio_features = [item['audio_features'] for item in batch]
                collated['audio_features'] = torch.stack(audio_features)
            
            # Handle text inputs
            if 'input_ids' in batch[0]:
                input_ids = [item['input_ids'] for item in batch]
                attention_mask = [item['attention_mask'] for item in batch]
                collated['input_ids'] = torch.cat(input_ids, dim=0)
                collated['attention_mask'] = torch.cat(attention_mask, dim=0)
            
            # Handle other fields
            for key in ['video_id', 'prompt', 'target_timeline']:
                if key in batch[0]:
                    collated[key] = [item[key] for item in batch]
            
            return collated
            
        except Exception as e:
            logger.error(f"Error in collate_fn: {e}")
            # Return a minimal batch on error
            return {
                'video_frames': torch.zeros(len(batch), 3, self.max_frames, 224, 224),
                'audio_features': torch.zeros(len(batch), self.max_audio_length),
                'input_ids': torch.zeros(len(batch), 512, dtype=torch.long),
                'attention_mask': torch.ones(len(batch), 512, dtype=torch.long)
            }
        
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
            split="train",
            max_frames=self.config.data.get('max_frames', 32) if hasattr(self.config, 'data') else 32,
            max_audio_length=self.config.data.get('max_audio_length', 160000) if hasattr(self.config, 'data') else 160000
        )
        
        return data.DataLoader(
            dataset,
            batch_size=int(self.config.training.batch_size),
            shuffle=True,
            num_workers=int(self.config.data.get('num_workers', 4)) if hasattr(self.config, 'data') else 4,
            pin_memory=True,
            drop_last=True
        )
    
    def get_val_loader(self, data_dir: str) -> data.DataLoader:
        """Get validation data loader"""
        dataset = VideoEditingDataset(
            data_dir=data_dir,
            split="val",
            max_frames=self.config.data.get('max_frames', 32) if hasattr(self.config, 'data') else 32,
            max_audio_length=self.config.data.get('max_audio_length', 160000) if hasattr(self.config, 'data') else 160000
        )
        
        return data.DataLoader(
            dataset,
            batch_size=int(self.config.training.batch_size),
            shuffle=False,
            num_workers=int(self.config.data.get('num_workers', 4)) if hasattr(self.config, 'data') else 4,
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
