"""
Vision Processor - Advanced video understanding with CLIP and object detection
"""

import os
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
        """Load and preprocess video - FIXED to return actual data"""
        try:
            # Check if file exists
            if not os.path.exists(video_path):
                logger.error(f"Video file not found: {video_path}")
                return self._get_empty_video_data()
            
            frames = self.extract_frames(video_path)
            
            if not frames:
                logger.warning(f"No frames extracted from {video_path}")
                return self._get_empty_video_data()
            
            # Get video metadata
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            # Stack frames into tensor (B, C, H, W)
            frames_tensor = torch.stack(frames)
            
            # Get comprehensive analysis
            analysis = self.analyze_scene(frames)
            
            return {
                'frames': frames_tensor,  # ACTUAL TENSOR DATA
                'fps': fps,
                'duration': duration,
                'num_frames': len(frames),
                'width': width,
                'height': height,
                'embeddings': analysis.get('embeddings'),
                'detections': analysis.get('detections', []),
                'scene_stats': analysis.get('scene_stats', {}),
                'temporal_features': analysis.get('temporal_features')
            }
            
        except Exception as e:
            logger.error(f"Error loading video {video_path}: {e}")
            return self._get_empty_video_data()
    
    def _get_empty_video_data(self) -> Dict[str, Any]:
        """Return empty video data structure"""
        return {
            'frames': torch.zeros(1, 3, *self.frame_size),
            'fps': 30.0,
            'duration': 0.0,
            'num_frames': 0,
            'width': self.frame_size[0],
            'height': self.frame_size[1],
            'embeddings': torch.zeros(1, 512),
            'detections': [],
            'scene_stats': {'avg_brightness': 0.0, 'motion_intensity': 0.0},
            'temporal_features': torch.zeros(3, 512)
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
        """
        Advanced object detection using RT-DETR/DETR or fallback methods.
        Provides professional-grade object detection for video understanding.
        """
        detections = []
        
        # Try to use advanced object detection models
        advanced_detector = self._initialize_advanced_detector()
        
        if advanced_detector:
            logger.info("Using advanced object detection (DETR/RT-DETR)")
            detections = self._detect_with_advanced_model(frames, advanced_detector)
        else:
            logger.info("Using enhanced OpenCV object detection fallback")
            detections = self._detect_with_enhanced_opencv(frames)
        
        return detections
    
    def _initialize_advanced_detector(self):
        """Initialize RT-DETR or DETR for object detection"""
        
        try:
            # Try to load DETR model (available alternative to RT-DETR)
            from transformers import DetrImageProcessor, DetrForObjectDetection
            
            model_name = "facebook/detr-resnet-50"
            processor = DetrImageProcessor.from_pretrained(model_name)
            model = DetrForObjectDetection.from_pretrained(model_name)
            model.to(self.device)
            model.eval()
            
            logger.info("✅ Loaded DETR model for advanced object detection")
            return {
                'model': model,
                'processor': processor,
                'type': 'detr'
            }
            
        except Exception as e:
            logger.warning(f"Advanced object detection not available: {e}")
            
            # Try RT-DETR if available
            try:
                from transformers import RTDetrImageProcessor, RTDetrForObjectDetection
                
                model_name = "microsoft/rt-detr-resnet-50"
                processor = RTDetrImageProcessor.from_pretrained(model_name)
                model = RTDetrForObjectDetection.from_pretrained(model_name)
                model.to(self.device)
                model.eval()
                
                logger.info("✅ Loaded RT-DETR model for real-time object detection")
                return {
                    'model': model,
                    'processor': processor,
                    'type': 'rt_detr'
                }
                
            except Exception as e2:
                logger.warning(f"RT-DETR also not available: {e2}")
                return None
    
    def _detect_with_advanced_model(self, frames: List[torch.Tensor], detector: Dict) -> List[Dict]:
        """Perform object detection using advanced models (DETR/RT-DETR)"""
        
        detections = []
        model = detector['model']
        processor = detector['processor']
        
        for i, frame in enumerate(frames):
            try:
                # Convert tensor to PIL Image
                frame_np = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                pil_image = Image.fromarray(frame_np)
                
                # Process with advanced detector
                inputs = processor(images=pil_image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Process outputs based on model type
                if detector['type'] in ['detr', 'rt_detr']:
                    frame_detections = self._process_detr_outputs(outputs, processor, pil_image.size)
                else:
                    frame_detections = []
                
                detections.append({
                    'frame_idx': i,
                    'objects': frame_detections,
                    'detection_method': 'advanced_transformer'
                })
                
            except Exception as e:
                logger.warning(f"Advanced detection failed for frame {i}: {e}")
                # Fallback to enhanced OpenCV for this frame
                fallback_detection = self._detect_single_frame_opencv(frame, i)
                detections.append(fallback_detection)
        
        return detections
    
    def _process_detr_outputs(self, outputs, processor, image_size: Tuple[int, int]) -> List[Dict]:
        """Process DETR model outputs into standardized detection format"""
        
        # Get predictions
        target_sizes = torch.tensor([image_size[::-1]], device=self.device)  # (height, width)
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]
        
        detections = []
        
        # Extract high-confidence detections
        confidence_threshold = 0.7
        
        for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
            if score > confidence_threshold:
                # Convert box coordinates to standard format
                x1, y1, x2, y2 = box.tolist()
                
                # Get class name (COCO classes)
                class_id = label.item()
                class_name = self._get_coco_class_name(class_id)
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(score),
                    'class': class_name,
                    'class_id': class_id
                })
        
        return detections
    
    def _get_coco_class_name(self, class_id: int) -> str:
        """Get COCO class name from class ID"""
        
        # COCO class names (simplified list)
        coco_classes = {
            1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
            6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic_light',
            11: 'fire_hydrant', 13: 'stop_sign', 14: 'parking_meter', 15: 'bench',
            16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
            21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
            27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
            34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports_ball', 38: 'kite',
            39: 'baseball_bat', 40: 'baseball_glove', 41: 'skateboard', 42: 'surfboard',
            43: 'tennis_racket', 44: 'bottle', 46: 'wine_glass', 47: 'cup', 48: 'fork',
            49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
            54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot_dog',
            59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
            64: 'potted_plant', 65: 'bed', 67: 'dining_table', 70: 'toilet',
            72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
            77: 'cell_phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
            82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
            88: 'teddy_bear', 89: 'hair_drier', 90: 'toothbrush'
        }
        
        return coco_classes.get(class_id, f'unknown_{class_id}')
    
    def _detect_with_enhanced_opencv(self, frames: List[torch.Tensor]) -> List[Dict]:
        """Enhanced OpenCV-based object detection with multiple algorithms"""
        
        detections = []
        
        for i, frame in enumerate(frames):
            detection = self._detect_single_frame_opencv(frame, i)
            detections.append(detection)
        
        return detections
    
    def _detect_single_frame_opencv(self, frame: torch.Tensor, frame_idx: int) -> Dict:
        """Advanced OpenCV object detection for single frame"""
        
        try:
            # Convert tensor to numpy array
            frame_np = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            
            frame_detections = []
            
            # Method 1: Contour-based detection (enhanced)
            gray = cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY)
            
            # Apply multiple preprocessing techniques
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # Morphological operations to improve contour detection
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Enhanced contour filtering and analysis
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Filter based on reasonable aspect ratios and sizes
                    if 0.2 < aspect_ratio < 5.0 and w > 30 and h > 30:
                        
                        # Calculate additional features
                        hull = cv2.convexHull(contour)
                        hull_area = cv2.contourArea(hull)
                        solidity = area / hull_area if hull_area > 0 else 0
                        
                        # Estimate object type based on shape characteristics
                        object_class = self._classify_contour_shape(contour, aspect_ratio, solidity)
                        
                        # Confidence based on contour quality
                        confidence = min(0.9, 0.3 + (solidity * 0.4) + (min(area, 5000) / 5000 * 0.3))
                        
                        frame_detections.append({
                            'bbox': [x, y, x+w, y+h],
                            'confidence': float(confidence),
                            'class': object_class,
                            'area': float(area),
                            'aspect_ratio': float(aspect_ratio),
                            'solidity': float(solidity)
                        })
            
            # Method 2: Corner detection for geometric objects
            corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
            if corners is not None:
                # Group nearby corners into potential objects
                corner_groups = self._group_corners(corners, distance_threshold=50)
                
                for group in corner_groups:
                    if len(group) >= 4:  # Potential rectangular object
                        x_coords = [corner[0][0] for corner in group]
                        y_coords = [corner[0][1] for corner in group]
                        
                        x, y = int(min(x_coords)), int(min(y_coords))
                        w, h = int(max(x_coords) - x), int(max(y_coords) - y)
                        
                        if w > 20 and h > 20:
                            frame_detections.append({
                                'bbox': [x, y, x+w, y+h],
                                'confidence': 0.6,
                                'class': 'geometric_object',
                                'detection_method': 'corner_based'
                            })
            
            return {
                'frame_idx': frame_idx,
                'objects': frame_detections[:15],  # Limit to top 15 detections
                'detection_method': 'enhanced_opencv'
            }
            
        except Exception as e:
            logger.warning(f"Enhanced OpenCV detection failed for frame {frame_idx}: {e}")
            return {'frame_idx': frame_idx, 'objects': [], 'detection_method': 'failed'}
    
    def _classify_contour_shape(self, contour, aspect_ratio: float, solidity: float) -> str:
        """Classify object type based on contour characteristics"""
        
        # Calculate additional shape features
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        
        if area > 0:
            circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        else:
            circularity = 0
        
        # Classification based on shape characteristics
        if circularity > 0.7:
            return 'circular_object'
        elif 0.8 < aspect_ratio < 1.2 and solidity > 0.8:
            return 'square_object'
        elif aspect_ratio > 2.0 and solidity > 0.7:
            return 'elongated_object'
        elif solidity < 0.5:
            return 'complex_shape'
        else:
            return 'general_object'
    
    def _group_corners(self, corners, distance_threshold: float = 50) -> List[List]:
        """Group nearby corners into potential objects"""
        
        if corners is None or len(corners) == 0:
            return []
        
        corner_groups = []
        used_corners = set()
        
        for i, corner in enumerate(corners):
            if i in used_corners:
                continue
                
            current_group = [corner]
            used_corners.add(i)
            
            # Find nearby corners
            for j, other_corner in enumerate(corners):
                if j in used_corners:
                    continue
                    
                distance = np.linalg.norm(corner[0] - other_corner[0])
                if distance < distance_threshold:
                    current_group.append(other_corner)
                    used_corners.add(j)
            
            if len(current_group) >= 3:  # Minimum corners for meaningful group
                corner_groups.append(current_group)
        
        return corner_groups
    
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
