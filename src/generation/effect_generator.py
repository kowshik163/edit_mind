"""
Effect Generator - Creates custom video effects and transitions
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class EffectGenerator:
    """
    Generates custom video effects using AI and traditional techniques
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Effect templates
        self.effect_templates = {
            'fade_in': self._fade_in,
            'fade_out': self._fade_out,
            'dissolve': self._dissolve,
            'wipe_left': self._wipe_left,
            'wipe_right': self._wipe_right,
            'zoom_in': self._zoom_in,
            'zoom_out': self._zoom_out,
            'color_grade_cinematic': self._cinematic_color_grade,
            'cinematic_color_grade': self._cinematic_color_grade,  # Alias
            'color_grade_dramatic': self._dramatic_color_grade,
            'blur_motion': self._motion_blur,
            'sharpen': self._sharpen,
            'vintage': self._vintage_effect,
            'cyberpunk': self._cyberpunk_effect,
            'film_grain': self._film_grain,
            'vignette': self._vignette
        }
        
        logger.info(f"🎨 Effect Generator initialized with {len(self.effect_templates)} effects")
    
    def generate_effect(self, effect_name: str, frames: List[np.ndarray], 
                       parameters: Optional[Dict[str, Any]] = None) -> List[np.ndarray]:
        """
        Apply specified effect to video frames
        
        Args:
            effect_name: Name of effect to apply
            frames: List of video frames as numpy arrays
            parameters: Effect-specific parameters
            
        Returns:
            List of processed frames
        """
        if effect_name not in self.effect_templates:
            logger.warning(f"Unknown effect '{effect_name}', applying passthrough")
            return frames
        
        effect_func = self.effect_templates[effect_name]
        parameters = parameters or {}
        
        try:
            return effect_func(frames, **parameters)
        except Exception as e:
            logger.error(f"Effect '{effect_name}' failed: {e}")
            return frames  # Return original frames on error
    
    def _fade_in(self, frames: List[np.ndarray], duration_frames: int = 30) -> List[np.ndarray]:
        """Fade in effect over specified frames"""
        result = frames.copy()
        fade_frames = min(duration_frames, len(frames))
        
        for i in range(fade_frames):
            alpha = i / fade_frames
            result[i] = (frames[i] * alpha).astype(np.uint8)
        
        return result
    
    def _fade_out(self, frames: List[np.ndarray], duration_frames: int = 30) -> List[np.ndarray]:
        """Fade out effect over specified frames"""
        result = frames.copy()
        fade_frames = min(duration_frames, len(frames))
        start_idx = len(frames) - fade_frames
        
        for i in range(fade_frames):
            idx = start_idx + i
            alpha = 1.0 - (i / fade_frames)
            result[idx] = (frames[idx] * alpha).astype(np.uint8)
        
        return result
    
    def _dissolve(self, frames: List[np.ndarray], next_frames: Optional[List[np.ndarray]] = None,
                  duration_frames: int = 30) -> List[np.ndarray]:
        """Dissolve transition between two clips"""
        if next_frames is None:
            return self._fade_out(frames, duration_frames)
        
        result = frames.copy()
        transition_frames = min(duration_frames, len(frames), len(next_frames))
        
        for i in range(transition_frames):
            alpha = i / transition_frames
            blended = (frames[i] * (1 - alpha) + next_frames[i] * alpha).astype(np.uint8)
            result[i] = blended
        
        return result
    
    def _wipe_left(self, frames: List[np.ndarray], duration_frames: int = 30) -> List[np.ndarray]:
        """Left wipe transition"""
        result = frames.copy()
        
        for i, frame in enumerate(frames[:duration_frames]):
            progress = i / duration_frames
            width = frame.shape[1]
            wipe_position = int(width * progress)
            
            # Create mask for wipe effect
            mask = np.zeros_like(frame)
            mask[:, :wipe_position] = frame[:, :wipe_position]
            result[i] = mask
        
        return result
    
    def _wipe_right(self, frames: List[np.ndarray], duration_frames: int = 30) -> List[np.ndarray]:
        """Right wipe transition"""
        result = frames.copy()
        
        for i, frame in enumerate(frames[:duration_frames]):
            progress = i / duration_frames
            width = frame.shape[1]
            wipe_position = int(width * (1 - progress))
            
            # Create mask for wipe effect
            mask = np.zeros_like(frame)
            mask[:, wipe_position:] = frame[:, wipe_position:]
            result[i] = mask
        
        return result
    
    def _zoom_in(self, frames: List[np.ndarray], zoom_factor: float = 1.5,
                 center: Optional[Tuple[int, int]] = None) -> List[np.ndarray]:
        """Zoom in effect"""
        result = []
        
        for frame in frames:
            h, w = frame.shape[:2]
            if center is None:
                center = (w // 2, h // 2)
            
            # Calculate crop region for zoom
            crop_w = int(w / zoom_factor)
            crop_h = int(h / zoom_factor)
            
            start_x = max(0, center[0] - crop_w // 2)
            start_y = max(0, center[1] - crop_h // 2)
            end_x = min(w, start_x + crop_w)
            end_y = min(h, start_y + crop_h)
            
            # Crop and resize
            cropped = frame[start_y:end_y, start_x:end_x]
            zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)
            result.append(zoomed)
        
        return result
    
    def _zoom_out(self, frames: List[np.ndarray], zoom_factor: float = 0.7) -> List[np.ndarray]:
        """Zoom out effect"""
        result = []
        
        for frame in frames:
            h, w = frame.shape[:2]
            
            # Resize frame down
            new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Place in center of original size canvas
            canvas = np.zeros_like(frame)
            start_y = (h - new_h) // 2
            start_x = (w - new_w) // 2
            canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized
            
            result.append(canvas)
        
        return result
    
    def _cinematic_color_grade(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply cinematic color grading"""
        result = []
        
        for frame in frames:
            # Convert to LAB color space for better color manipulation
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Enhance contrast in L channel
            l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
            
            # Add orange/teal look - warm highlights, cool shadows
            a = cv2.add(a, 10)  # Push toward orange
            b = cv2.subtract(b, 5)  # Push toward teal in shadows
            
            # Merge and convert back
            enhanced = cv2.merge([l, a, b])
            final = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            result.append(final)
        
        return result
    
    def _dramatic_color_grade(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply dramatic color grading"""
        result = []
        
        for frame in frames:
            # Increase contrast and saturation
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # Boost saturation
            s = cv2.multiply(s, 1.3)
            
            # Increase contrast
            v = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(v)
            
            # Merge and convert back
            enhanced = cv2.merge([h, s, v])
            final = cv2.cvtColor(enhanced, cv2.COLOR_HSV2BGR)
            result.append(final)
        
        return result
    
    def _motion_blur(self, frames: List[np.ndarray], intensity: int = 15) -> List[np.ndarray]:
        """Apply motion blur effect"""
        result = []
        kernel = cv2.getRotationMatrix2D((intensity//2, intensity//2), 0, 1)
        
        for frame in frames:
            blurred = cv2.filter2D(frame, -1, kernel[:1])
            result.append(blurred)
        
        return result
    
    def _sharpen(self, frames: List[np.ndarray], intensity: float = 1.0) -> List[np.ndarray]:
        """Apply sharpening effect"""
        result = []
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) * intensity
        
        for frame in frames:
            sharpened = cv2.filter2D(frame, -1, kernel)
            result.append(np.clip(sharpened, 0, 255).astype(np.uint8))
        
        return result
    
    def _vintage_effect(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply vintage film effect"""
        result = []
        
        for frame in frames:
            # Sepia tone
            sepia_filter = np.array([[0.272, 0.534, 0.131],
                                   [0.349, 0.686, 0.168],
                                   [0.393, 0.769, 0.189]])
            
            sepia = cv2.transform(frame, sepia_filter.T)
            
            # Add some noise for film grain
            noise = np.random.normal(0, 10, frame.shape).astype(np.int16)
            vintage = np.clip(sepia.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            result.append(vintage)
        
        return result
    
    def _cyberpunk_effect(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply cyberpunk aesthetic"""
        result = []
        
        for frame in frames:
            # Enhance blue and magenta channels
            b, g, r = cv2.split(frame)
            
            # Boost blue channel
            b = cv2.multiply(b, 1.2)
            
            # Add magenta tint
            enhanced = cv2.merge([b, g, np.clip(r * 1.1, 0, 255).astype(np.uint8)])
            
            # High contrast
            enhanced = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(
                cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            )
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            
            result.append(enhanced)
        
        return result
    
    def _film_grain(self, frames: List[np.ndarray], intensity: float = 0.1) -> List[np.ndarray]:
        """Add film grain effect"""
        result = []
        
        for frame in frames:
            noise = np.random.normal(0, intensity * 255, frame.shape).astype(np.int16)
            grainy = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            result.append(grainy)
        
        return result
    
    def _vignette(self, frames: List[np.ndarray], intensity: float = 0.3) -> List[np.ndarray]:
        """Add vignette effect"""
        result = []
        
        for frame in frames:
            h, w = frame.shape[:2]
            
            # Create vignette mask
            X, Y = np.ogrid[:h, :w]
            center_x, center_y = w/2, h/2
            
            # Calculate distance from center
            distance = np.sqrt((X - center_y)**2 + (Y - center_x)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            
            # Create vignette
            vignette_mask = 1 - (distance / max_distance) * intensity
            vignette_mask = np.clip(vignette_mask, 0, 1)
            
            # Apply vignette to each channel
            vignetted = frame * vignette_mask[:, :, np.newaxis]
            result.append(vignetted.astype(np.uint8))
        
        return result
