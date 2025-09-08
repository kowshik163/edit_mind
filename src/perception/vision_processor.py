"""
Vision Processor - Advanced video understanding with CLIP and object detection
"""

import cv2
import torch
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from omegaconf import DictConfig
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

logger = logging.getLogger(__name__)


class VisionProcessor:
    """Advanced vision processor for video understanding with CLIP and RT-DETR"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Initialize CLIP model for frame encoding
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.to(self.device)
        
        # Frame extraction settings
        self.target_fps = config.get('target_fps', 30)
        self.frame_size = config.get('frame_size', (224, 224))
        
        logger.info(f"VisionProcessor initialized on {self.device}")
    
    def load_video(self, video_path: str) -> Dict[str, Any]:
        """Load and preprocess video"""
        try:
            frames = self.extract_frames(video_path)
            
            # Get video metadata
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            return {
                'frames': torch.stack(frames) if frames else torch.zeros(0, 3, *self.frame_size),
                'fps': fps,
                'duration': duration,
                'num_frames': len(frames)
            }
            
        except Exception as e:
            logger.error(f"Error loading video {video_path}: {e}")
            return {
                'frames': torch.zeros(0, 3, *self.frame_size),
                'fps': 30,
                'duration': 0.0,
                'num_frames': 0
            }
    
    def extract_frames(self, video_path: str, max_frames: int = 1000) -> List[torch.Tensor]:
        """Extract frames from video using OpenCV"""
        frames = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame sampling rate
            if frame_count > max_frames:
                step = frame_count // max_frames
            else:
                step = 1
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % step == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Resize frame
                    frame_resized = cv2.resize(frame_rgb, self.frame_size)
                    
                    # Convert to tensor
                    frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
                    frames.append(frame_tensor)
                    
                    if len(frames) >= max_frames:
                        break
                
                frame_idx += 1
            
            cap.release()
            logger.info(f"Extracted {len(frames)} frames from {video_path}")
            
        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {e}")
            
        return frames
    
    def encode_frames(self, frames: List[torch.Tensor]) -> torch.Tensor:
        """Encode frames using CLIP vision encoder"""
        if not frames:
            return torch.zeros((0, 512))  # CLIP base embedding size
        
        try:
            # Stack frames into batch
            frame_batch = torch.stack(frames)
            
            # Process with CLIP
            with torch.no_grad():
                inputs = {"pixel_values": frame_batch.to(self.device)}
                vision_outputs = self.clip_model.vision_model(**inputs)
                
                # Get pooled features
                embeddings = vision_outputs.pooler_output
            
            logger.info(f"Encoded {len(frames)} frames to embeddings shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error encoding frames: {e}")
            return torch.zeros((len(frames), 512))
    
    def detect_objects(self, frames: List[torch.Tensor]) -> List[Dict]:
        """Detect objects using basic OpenCV methods (placeholder for RT-DETR)"""
        detections = []
        
        for i, frame in enumerate(frames):
            try:
                # Convert tensor back to numpy array for OpenCV
                frame_np = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                
                # Basic edge detection as placeholder
                gray = cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                frame_detections = []
                for contour in contours[:10]:  # Limit to top 10 contours
                    x, y, w, h = cv2.boundingRect(contour)
                    if w > 20 and h > 20:  # Filter small detections
                        frame_detections.append({
                            'bbox': [x, y, x+w, y+h],
                            'confidence': 0.5,  # Placeholder confidence
                            'class': 'object'   # Placeholder class
                        })
                
                detections.append({
                    'frame_idx': i,
                    'objects': frame_detections
                })
                
            except Exception as e:
                logger.warning(f"Error detecting objects in frame {i}: {e}")
                detections.append({'frame_idx': i, 'objects': []})
        
        return detections
    
    def analyze_scene(self, frames: List[torch.Tensor]) -> Dict[str, Any]:
        """Comprehensive scene analysis"""
        if not frames:
            return {'embeddings': torch.zeros((0, 512)), 'detections': []}
        
        try:
            # Get frame embeddings
            embeddings = self.encode_frames(frames)
            
            # Get object detections
            detections = self.detect_objects(frames)
            
            # Basic scene statistics
            scene_stats = {
                'num_frames': len(frames),
                'avg_brightness': self._calculate_brightness(frames),
                'motion_intensity': self._estimate_motion(frames)
            }
            
            return {
                'embeddings': embeddings,
                'detections': detections,
                'scene_stats': scene_stats,
                'temporal_features': self._extract_temporal_features(embeddings)
            }
            
        except Exception as e:
            logger.error(f"Error in scene analysis: {e}")
            return {'embeddings': torch.zeros((len(frames), 512)), 'detections': []}
    
    def _calculate_brightness(self, frames: List[torch.Tensor]) -> float:
        """Calculate average brightness across frames"""
        if not frames:
            return 0.0
        
        brightness_values = []
        for frame in frames:
            # Convert to grayscale and calculate mean
            gray = 0.299 * frame[0] + 0.587 * frame[1] + 0.114 * frame[2]
            brightness_values.append(gray.mean().item())
        
        return sum(brightness_values) / len(brightness_values)
    
    def _estimate_motion(self, frames: List[torch.Tensor]) -> float:
        """Estimate motion intensity between consecutive frames"""
        if len(frames) < 2:
            return 0.0
        
        motion_values = []
        for i in range(1, len(frames)):
            # Calculate frame difference
            diff = torch.abs(frames[i] - frames[i-1])
            motion_values.append(diff.mean().item())
        
        return sum(motion_values) / len(motion_values)
    
    def _extract_temporal_features(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Extract temporal features from frame embeddings"""
        if embeddings.size(0) < 2:
            return embeddings.mean(dim=0, keepdim=True) if embeddings.size(0) > 0 else torch.zeros(1, embeddings.size(1))
        
        # Simple temporal pooling - can be enhanced with LSTM/Transformer
        temporal_features = torch.stack([
            embeddings.mean(dim=0),  # Average pooling
            embeddings.max(dim=0)[0],  # Max pooling  
            embeddings.std(dim=0),   # Standard deviation
        ])
        
        return temporal_features
