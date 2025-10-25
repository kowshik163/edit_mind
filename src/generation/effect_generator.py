"""
Advanced Effect Generator - Creates a wide range of parameterized and keyframable video effects
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from pathlib import Path
import math

logger = logging.getLogger(__name__)

# --- Keyframing Helper ---

class Keyframer:
    """Handles simple linear interpolation for keyframed parameters."""

    @staticmethod
    def get_value(param_data: Any, frame_idx: int, total_frames: int) -> Any:
        """
        Gets the value for a parameter at a specific frame index.
        Handles static values or keyframed lists.

        Args:
            param_data: Either a static value or a list of keyframe dicts
                        (e.g., [{'frame': 0, 'value': 1.0}, {'frame': 30, 'value': 1.5}])
            frame_idx: The current frame index.
            total_frames: The total number of frames in the clip/sequence.

        Returns:
            The interpolated or static value for the current frame.
        """
        if not isinstance(param_data, list) or not param_data:
            return param_data # Static value

        # Ensure keyframes are sorted by frame index
        keyframes = sorted(param_data, key=lambda k: k.get('frame', 0))

        # Find surrounding keyframes
        prev_kf = keyframes[0]
        next_kf = keyframes[-1]

        for i in range(len(keyframes)):
            if keyframes[i]['frame'] <= frame_idx:
                prev_kf = keyframes[i]
            if keyframes[i]['frame'] >= frame_idx:
                next_kf = keyframes[i]
                break

        # Interpolate
        if prev_kf['frame'] == next_kf['frame'] or prev_kf['frame'] > frame_idx :
             # Before first keyframe or exactly on a keyframe
             # Ensure the value exists before returning
             return prev_kf.get('value') if 'value' in prev_kf else None # Or a default value

        frame_diff = next_kf['frame'] - prev_kf['frame']
        if frame_diff <= 0:
            return prev_kf.get('value') # Avoid division by zero, return previous value

        progress = (frame_idx - prev_kf['frame']) / frame_diff

        # Linear interpolation for numerical values
        prev_val = prev_kf.get('value')
        next_val = next_kf.get('value')

        if isinstance(prev_val, (int, float)) and isinstance(next_val, (int, float)):
            return prev_val + (next_val - prev_val) * progress
        elif isinstance(prev_val, (tuple, list)) and isinstance(next_val, (tuple, list)) and len(prev_val) == len(next_val):
             # Interpolate tuples/lists element-wise (e.g., for position)
             interp_list = [pv + (nv - pv) * progress for pv, nv in zip(prev_val, next_val)]
             return tuple(interp_list) if isinstance(prev_val, tuple) else interp_list
        else:
            # For non-numeric types, return the value of the previous keyframe
            return prev_val


# --- Advanced Effect Generator ---

class AdvancedEffectGenerator:
    """
    Applies a wide variety of parameterized and keyframable video effects.
    """

    def __init__(self, quality: str = 'high', gpu_acceleration: bool = False):
        self.quality = quality
        # Basic check, full GPU requires OpenCV compiled with CUDA & specific functions
        self.use_gpu = gpu_acceleration and hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0

        self.effect_registry: Dict[str, Callable] = self._register_effects()
        self.keyframer = Keyframer()

        if self.use_gpu:
            logger.info("GPU acceleration hint: Enabled (requires OpenCV compiled with CUDA)")
        else:
            logger.info("GPU acceleration: Disabled or unavailable")
        logger.info(f"🎨 Advanced Effect Generator initialized with {len(self.effect_registry)} effects.")

    def _register_effects(self) -> Dict[str, Callable]:
        """Dynamically register all effect methods starting with '_effect_'."""
        effects = {}
        for name in dir(self):
            if name.startswith("_effect_"):
                effect_name = name[len("_effect_"):]
                effects[effect_name] = getattr(self, name)
        # Add aliases
        effects['brightness_contrast'] = effects['basic_color']
        effects['color_temperature'] = effects['color_temp_tint']
        effects['curves_rgb'] = effects['curves_poly'] # Alias for simplified curves
        effects['gaussian_blur'] = effects['blur_gaussian']
        effects['sharpen'] = effects['sharpen_unsharpmask']
        effects['black_and_white'] = effects['desaturate']
        effects['greyscale'] = effects['desaturate']
        effects['picture_in_picture'] = effects['overlay_image']
        effects['pip'] = effects['overlay_image']
        effects['transform'] = effects['geometric_transform']
        effects['crop'] = effects['geometric_crop']
        effects['rotate'] = effects['geometric_rotate']
        effects['flip'] = effects['geometric_flip']
        effects['crossfade'] = effects['transition_dissolve'] # Needs timeline context
        effects['fade_to_black'] = effects['transition_fade'] # Needs timeline context
        effects['rgb_shift'] = effects['glitch_rgb_shift']
        return effects

    def get_available_effects(self) -> List[str]:
        """Returns a list of available effect names."""
        return sorted(list(self.effect_registry.keys()))

    def apply_effect(self, frame: np.ndarray, effect_name: str, frame_idx: int = 0, total_frames: int = 1, **params) -> np.ndarray:
        """
        Applies a registered effect to a single frame with parameters and keyframing.

        Args:
            frame: Input frame (numpy array BGR).
            effect_name: Name of the effect to apply.
            frame_idx: Current frame index (for keyframing).
            total_frames: Total frames in the sequence (for keyframing).
            **params: Dictionary of parameters for the effect.
                      Can contain static values or keyframe lists.

        Returns:
            Processed frame (numpy array BGR).
        """
        if effect_name not in self.effect_registry:
            logger.warning(f"Effect '{effect_name}' not found. Returning original frame.")
            return frame

        effect_func = self.effect_registry[effect_name]

        # Resolve keyframed parameters for the current frame
        resolved_params = {}
        for key, value in params.items():
            resolved_params[key] = self.keyframer.get_value(value, frame_idx, total_frames)

        try:
            # Ensure frame is writeable if needed
            frame_copy = frame.copy()
            processed_frame = effect_func(frame_copy, frame_idx=frame_idx, total_frames=total_frames, **resolved_params)
            # Ensure output is valid uint8 BGR
            if processed_frame is None or not isinstance(processed_frame, np.ndarray) or processed_frame.dtype != np.uint8:
                 logger.warning(f"Effect '{effect_name}' returned invalid output type {type(processed_frame)}. Returning original frame.")
                 return frame
            # Ensure shape matches, resize if necessary (though effects ideally shouldn't change size unless intended like crop/zoom)
            if processed_frame.shape != frame.shape:
                 if len(processed_frame.shape) == len(frame.shape) and processed_frame.shape[2] == frame.shape[2]: # Allow size changes
                      pass # Size change might be intended (e.g. crop result)
                 else:
                      logger.warning(f"Effect '{effect_name}' changed frame shape/channels from {frame.shape} to {processed_frame.shape}. Attempting resize.")
                      try:
                           processed_frame = cv2.resize(processed_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                      except Exception as resize_err:
                           logger.error(f"Failed to resize output of '{effect_name}': {resize_err}. Returning original frame.")
                           return frame

            return processed_frame

        except Exception as e:
            logger.error(f"Error applying effect '{effect_name}' with params {resolved_params}: {e}", exc_info=True)
            return frame # Return original frame on error

    # --- Effect Implementations ---

    # --- Color Correction & Grading ---

    def _effect_basic_color(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                            brightness: float = 0.0, contrast: float = 1.0,
                            saturation: float = 1.0, **kwargs) -> np.ndarray:
        """Adjusts brightness, contrast, and saturation.
           Brightness: [-100, 100], Contrast: [0.0, 3.0], Saturation: [0.0, 3.0]"""
        brightness = np.clip(brightness, -100, 100)
        contrast = np.clip(contrast, 0.0, 3.0)
        saturation = np.clip(saturation, 0.0, 3.0)

        # Brightness & Contrast (Linear transformation)
        # Apply contrast first, then brightness
        out = frame.astype(np.float32)
        out = out * contrast + brightness
        out = np.clip(out, 0, 255)

        # Saturation
        if abs(saturation - 1.0) > 1e-6:
            hsv = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
            out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return out.astype(np.uint8)

    def _effect_hue_shift(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                          shift: float = 0.0, **kwargs) -> np.ndarray:
        """Shifts the hue of the image. Shift: [-180, 180] degrees."""
        shift = np.clip(shift, -180, 180)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        # OpenCV hue is [0, 179]
        hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _effect_color_temp_tint(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                                temperature: float = 0.0, tint: float = 0.0, **kwargs) -> np.ndarray:
        """Adjusts color temperature (blue/orange) and tint (green/magenta).
           Temperature/Tint: [-100, 100]"""
        temperature = np.clip(temperature, -100, 100) / 100.0 # Normalize to -1 to 1
        tint = np.clip(tint, -100, 100) / 100.0 # Normalize to -1 to 1

        out = frame.astype(np.float32)
        # Temperature: Add/subtract blue/red
        out[:, :, 0] = np.clip(out[:, :, 0] * (1 - temperature * 0.5), 0, 255) # Blue
        out[:, :, 2] = np.clip(out[:, :, 2] * (1 + temperature * 0.5), 0, 255) # Red
        # Tint: Add/subtract green/magenta (approximated by R/B adjust)
        out[:, :, 1] = np.clip(out[:, :, 1] * (1 + tint * 0.5), 0, 255) # Green
        # Adjust R/B inversely for magenta tint effect
        out[:, :, 0] = np.clip(out[:, :, 0] * (1 - tint * 0.25), 0, 255)
        out[:, :, 2] = np.clip(out[:, :, 2] * (1 - tint * 0.25), 0, 255)

        return out.astype(np.uint8)

    def _effect_vibrance(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                         amount: float = 0.0, **kwargs) -> np.ndarray:
        """Increases saturation of less saturated colors more. Amount: [-100, 100]"""
        amount = np.clip(amount, -100, 100) / 100.0 # Normalize -1 to 1
        if abs(amount) < 1e-6:
            return frame

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        s = hsv[:, :, 1]

        # Calculate mask based on saturation (less saturated pixels get higher weight)
        # Avoid dividing by zero if s is zero
        sat_weight = 1.0 - np.clip(s / 255.0, 0, 1) # Weight is higher for low saturation

        # Apply vibrance adjustment: delta = amount * weight * saturation
        delta_s = amount * sat_weight * s
        hsv[:, :, 1] = np.clip(s + delta_s, 0, 255)

        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _effect_shadows_highlights(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                                   shadows: float = 0.0, highlights: float = 0.0, **kwargs) -> np.ndarray:
        """Adjusts shadows and highlights. Shadows/Highlights: [-100, 100]"""
        shadows = np.clip(shadows, -100, 100) / 100.0
        highlights = np.clip(highlights, -100, 100) / 100.0
        if abs(shadows) < 1e-6 and abs(highlights) < 1e-6:
            return frame

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        l = lab[:, :, 0]

        # Shadow mask (lower L values) - Sigmoid transition
        shadow_mask = 1.0 / (1.0 + np.exp((l - 70) / 15.0)) # Affects L < ~85
        # Highlight mask (higher L values) - Sigmoid transition
        highlight_mask = 1.0 / (1.0 + np.exp(-(l - 180) / 15.0)) # Affects L > ~165

        # Apply adjustments
        l_adjusted = l + shadows * shadow_mask * 50 # Shadow adjustment strength
        l_adjusted = l_adjusted + highlights * highlight_mask * 50 # Highlight adjustment strength

        lab[:, :, 0] = np.clip(l_adjusted, 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    def _effect_curves_poly(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                            # Example: coefficients for R, G, B channels (ax^2 + bx + c)
                            r_coeffs: List[float] = [0.0, 1.0, 0.0],
                            g_coeffs: List[float] = [0.0, 1.0, 0.0],
                            b_coeffs: List[float] = [0.0, 1.0, 0.0],
                            **kwargs) -> np.ndarray:
        """Applies simple polynomial curves to RGB channels. Coeffs=[a, b, c] for ax^2+bx+c."""
        if len(r_coeffs) != 3 or len(g_coeffs) != 3 or len(b_coeffs) != 3:
             logger.warning("Curve coefficients must be lists of 3 floats [a, b, c]. Using defaults.")
             r_coeffs, g_coeffs, b_coeffs = [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]

        out = frame.astype(np.float32) / 255.0 # Normalize to 0-1

        def apply_poly(channel, coeffs):
            a, b, c = coeffs
            # Ensure output stays within [0, 1] after polynomial application
            return np.clip(a * channel**2 + b * channel + c, 0, 1)


        out[:, :, 2] = apply_poly(out[:, :, 2], r_coeffs) # R channel
        out[:, :, 1] = apply_poly(out[:, :, 1], g_coeffs) # G channel
        out[:, :, 0] = apply_poly(out[:, :, 0], b_coeffs) # B channel

        return (out * 255).astype(np.uint8)

    def _effect_lut_apply(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                          lut_path: str = None, intensity: float = 1.0, **kwargs) -> np.ndarray:
        """Applies a 3D LUT (Look-Up Table) file (.cube)."""
        intensity = np.clip(intensity, 0.0, 1.0)
        # LUT application is complex involving parsing the .cube file and interpolating.
        # This is a placeholder. Libraries like 'pylut' or implementing the interpolation
        # logic manually would be needed for a full implementation.
        logger.warning("LUT application is complex and currently a placeholder. Applying simple color shift.")
        if lut_path and Path(lut_path).exists():
             # Placeholder: apply a generic 'cinematic' tint based on intensity
             out = frame.astype(np.float32)
             out[:, :, 0] *= (1 - intensity * 0.1) # Less blue
             out[:, :, 2] *= (1 + intensity * 0.1) # More red
             return np.clip(out, 0, 255).astype(np.uint8)
        return frame


    # --- Blurs ---

    def _effect_blur_gaussian(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                              strength: float = 5.0, **kwargs) -> np.ndarray:
        """Applies Gaussian blur. Strength corresponds roughly to pixel radius."""
        ksize = int(strength * 2) * 2 + 1 # Must be odd
        ksize = max(1, ksize)
        return cv2.GaussianBlur(frame, (ksize, ksize), 0)

    def _effect_blur_box(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                         strength: float = 5.0, **kwargs) -> np.ndarray:
        """Applies Box blur."""
        ksize = int(strength * 2) + 1
        ksize = max(1, ksize)
        return cv2.blur(frame, (ksize, ksize))

    def _effect_blur_median(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                            strength: float = 5.0, **kwargs) -> np.ndarray:
        """Applies Median blur (good for noise reduction)."""
        ksize = int(strength * 2) + 1 # Must be odd
        ksize = max(3, ksize) # Minimum kernel size for median is 3
        return cv2.medianBlur(frame, ksize)


    def _effect_blur_motion(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                            angle: float = 0.0, distance: float = 20.0, **kwargs) -> np.ndarray:
        """Applies directional motion blur."""
        distance = max(1, int(distance))
        kernel_size = distance
        center = kernel_size // 2

        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        rad_angle = np.radians(angle)

        # Draw a line in the kernel
        # This is a simplified way to create the kernel line
        for i in range(kernel_size):
            dx = (i - center) * np.cos(rad_angle)
            dy = (i - center) * np.sin(rad_angle)
            x = int(round(center + dx))
            y = int(round(center + dy))
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1.0 # Use 1.0 for equal weight

        # Correct edge case where kernel might be empty if distance=1 and angle makes points outside
        if np.sum(kernel) == 0:
            kernel[center, center] = 1.0


        kernel = kernel / np.sum(kernel) # Normalize
        return cv2.filter2D(frame, -1, kernel)

    def _effect_blur_radial(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                            strength: float = 0.1, center_x: float = 0.5, center_y: float = 0.5,
                            samples: int = 32, **kwargs) -> np.ndarray:
        """Applies radial (zoom) blur."""
        strength = np.clip(strength, 0.0, 0.5)
        samples = max(4, int(samples))
        h, w = frame.shape[:2]
        cx, cy = int(center_x * w), int(center_y * h)

        frame_float = frame.astype(np.float32)
        accum = frame_float.copy() * (1.0 / samples) # Start with first sample

        # Generate multiple scaled versions and accumulate
        for i in range(1, samples):
            progress = i / (samples - 1)
            scale = 1.0 + progress * strength * 2 # Scale from 1.0 up to 1.0 + strength*2

            # Calculate ROI to scale
            roi_w = int(w / scale)
            roi_h = int(h / scale)
            roi_x = max(0, cx - roi_w // 2)
            roi_y = max(0, cy - roi_h // 2)

            # Ensure ROI dimensions are valid
            roi_w = min(roi_w, w - roi_x)
            roi_h = min(roi_h, h - roi_y)

            if roi_w <= 0 or roi_h <= 0: continue # Skip if ROI is invalid

            # Extract ROI and resize back to full frame
            roi = frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
            scaled_roi = cv2.resize(roi, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

            accum += scaled_roi * (1.0 / samples)

        return np.clip(accum, 0, 255).astype(np.uint8)

    def _effect_blur_bokeh(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                          radius: int = 5, threshold: int = 200, **kwargs) -> np.ndarray:
        """Simulates Bokeh blur effect (blurs non-highlight areas)."""
        radius = max(1, int(radius))
        ksize = radius * 2 + 1
        threshold = np.clip(threshold, 0, 255)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Create mask for highlights (areas above threshold)
        highlight_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]

        # Blur the entire image
        blurred = cv2.GaussianBlur(frame, (ksize, ksize), 0)

        # Combine blurred background with original highlights
        # Ensure mask is 3 channels
        highlight_mask_3ch = cv2.cvtColor(highlight_mask, cv2.COLOR_GRAY2BGR)
        result = np.where(highlight_mask_3ch == 255, frame, blurred)

        return result.astype(np.uint8)


    # --- Sharpening ---

    def _effect_sharpen_basic(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                              amount: float = 1.0, **kwargs) -> np.ndarray:
        """Applies a basic sharpening kernel."""
        amount = np.clip(amount, 0.0, 3.0)
        # Basic sharpening kernel
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]], dtype=np.float32)
        # Adjust center weight based on amount (9 is default 1.0 amount)
        kernel[1, 1] = 8 * amount + 1

        sharpened = cv2.filter2D(frame, -1, kernel)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def _effect_sharpen_unsharpmask(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                                    radius: float = 1.0, amount: float = 1.0, threshold: int = 0, **kwargs) -> np.ndarray:
        """Applies Unsharp Masking for more controlled sharpening."""
        radius = max(1, int(radius))
        ksize = radius * 2 + 1
        amount = np.clip(amount, 0.0, 5.0)
        threshold = np.clip(threshold, 0, 255)

        # Gaussian blur
        blurred = cv2.GaussianBlur(frame, (ksize, ksize), 0)

        # Calculate the mask (difference between original and blurred)
        mask = cv2.subtract(frame, blurred)

        # Apply threshold to the mask
        if threshold > 0:
            mask[np.abs(mask) < threshold] = 0

        # Add weighted mask back to original
        sharpened = cv2.addWeighted(frame, 1.0 + amount, blurred, -amount, 0)

        return sharpened


    # --- Distortions ---

    def _effect_distort_ripple(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                              amplitude: float = 10.0, frequency: float = 0.05,
                              center_x: float = 0.5, center_y: float = 0.5, **kwargs) -> np.ndarray:
        """Creates a ripple distortion effect from a center point."""
        amplitude = max(0, amplitude)
        frequency = max(0.01, frequency)
        h, w = frame.shape[:2]
        cx, cy = int(center_x * w), int(center_y * h)

        # Create coordinate maps
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)

        for y in range(h):
            for x in range(w):
                dx = x - cx
                dy = y - cy
                distance = np.sqrt(dx*dx + dy*dy) + 1e-6 # Add epsilon
                # Calculate displacement based on distance
                displacement = amplitude * np.sin(distance * frequency)
                # Calculate new coordinates (polar distortion)
                angle = np.arctan2(dy, dx)
                new_x = x + displacement * np.cos(angle)
                new_y = y + displacement * np.sin(angle)
                map_x[y, x] = np.clip(new_x, 0, w - 1)
                map_y[y, x] = np.clip(new_y, 0, h - 1)

        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)

    def _effect_distort_wave(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                             amplitude: float = 10.0, frequency: float = 0.05,
                             direction: str = 'horizontal', phase_shift: float = 0.0, **kwargs) -> np.ndarray:
        """Creates a wave distortion effect."""
        amplitude = max(0, amplitude)
        frequency = max(0.01, frequency)
        h, w = frame.shape[:2]

        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))

        if direction == 'horizontal':
            displacement = amplitude * np.sin(y_coords * frequency + phase_shift)
            map_x = np.clip(x_coords + displacement, 0, w - 1)
            map_y = y_coords.astype(np.float32)
        elif direction == 'vertical':
            displacement = amplitude * np.sin(x_coords * frequency + phase_shift)
            map_y = np.clip(y_coords + displacement, 0, h - 1)
            map_x = x_coords.astype(np.float32)
        else: # both
            disp_x = amplitude * np.sin(y_coords * frequency + phase_shift)
            disp_y = amplitude * np.cos(x_coords * frequency * 0.7 + phase_shift * 0.5) # Slightly different freq/phase
            map_x = np.clip(x_coords + disp_x, 0, w - 1)
            map_y = np.clip(y_coords + disp_y, 0, h - 1)


        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)

    def _effect_distort_pinch_punch(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                                    strength: float = 0.5, radius_factor: float = 0.5,
                                    center_x: float = 0.5, center_y: float = 0.5, **kwargs) -> np.ndarray:
        """Creates a pinch (strength>0) or punch (strength<0) distortion."""
        strength = np.clip(strength, -1.0, 1.0) # -1 (punch) to 1 (pinch)
        radius_factor = np.clip(radius_factor, 0.01, 1.0)
        h, w = frame.shape[:2]
        cx, cy = int(center_x * w), int(center_y * h)
        radius = min(w, h) * radius_factor / 2

        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)

        for y in range(h):
            for x in range(w):
                dx = x - cx
                dy = y - cy
                distance = np.sqrt(dx*dx + dy*dy)

                if distance < radius:
                    # Normalized distance within radius
                    norm_dist = distance / radius
                    # Pinch/Punch distortion formula (pow controls the curve)
                    factor = pow(1.0 - norm_dist, 2.0) # Stronger effect near center
                    scale = 1.0 - strength * factor
                    new_x = cx + dx * scale
                    new_y = cy + dy * scale
                    map_x[y, x] = np.clip(new_x, 0, w - 1)
                    map_y[y, x] = np.clip(new_y, 0, h - 1)
                else:
                    map_x[y, x] = float(x)
                    map_y[y, x] = float(y)

        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)

    def _effect_distort_twirl(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                             angle: float = 45.0, radius_factor: float = 0.5,
                             center_x: float = 0.5, center_y: float = 0.5, **kwargs) -> np.ndarray:
        """Creates a twirl distortion."""
        angle_rad = np.radians(angle)
        radius_factor = np.clip(radius_factor, 0.01, 1.0)
        h, w = frame.shape[:2]
        cx, cy = int(center_x * w), int(center_y * h)
        radius = min(w, h) * radius_factor / 2

        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)

        for y in range(h):
            for x in range(w):
                dx = x - cx
                dy = y - cy
                distance = np.sqrt(dx*dx + dy*dy)

                if distance < radius:
                    # Amount of twirl depends on distance from center
                    twirl_amount = angle_rad * (1.0 - distance / radius)
                    # Rotate coordinates
                    current_angle = np.arctan2(dy, dx)
                    new_angle = current_angle + twirl_amount
                    # Calculate new coordinates
                    new_x = cx + distance * np.cos(new_angle)
                    new_y = cy + distance * np.sin(new_angle)
                    map_x[y, x] = np.clip(new_x, 0, w - 1)
                    map_y[y, x] = np.clip(new_y, 0, h - 1)
                else:
                    map_x[y, x] = float(x)
                    map_y[y, x] = float(y)

        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)

    def _effect_distort_fisheye(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                                strength: float = 0.5, **kwargs) -> np.ndarray:
        """Simulates a fisheye lens distortion."""
        strength = max(0, strength)
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)

        for y in range(h):
            for x in range(w):
                dx = x - cx
                dy = y - cy
                dist_sq = dx*dx + dy*dy
                # Fisheye distortion formula
                radius = np.sqrt(dist_sq)
                if radius == 0:
                     theta = 0
                else:
                     theta = np.arctan2(dy, dx)

                # Map radius non-linearly
                # Simple polynomial mapping for fisheye effect
                max_dist = np.sqrt((w/2)**2 + (h/2)**2)
                norm_radius = radius / max_dist
                # Adjust the power for strength: higher power means more distortion near edges
                mapped_radius = radius * (1.0 - strength * norm_radius**2) # Simple quadratic mapping

                new_x = cx + mapped_radius * np.cos(theta)
                new_y = cy + mapped_radius * np.sin(theta)

                map_x[y, x] = np.clip(new_x, 0, w - 1)
                map_y[y, x] = np.clip(new_y, 0, h - 1)

        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)

    # --- Stylistic ---

    def _effect_vignette(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                         strength: float = 0.6, radius_factor: float = 0.7, **kwargs) -> np.ndarray:
        """Applies a vignette (darkened edges) effect."""
        strength = np.clip(strength, 0.0, 1.0)
        radius_factor = np.clip(radius_factor, 0.1, 1.5)
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        # Create radial gradient mask
        y_coords, x_coords = np.ogrid[:h, :w]
        dist_sq = (x_coords - cx)**2 + (y_coords - cy)**2
        # Use radius_factor to control the size of the bright center
        # Use strength to control the darkness of the edges
        max_dist_sq = (radius_factor * min(cx, cy))**2 + 1e-6 # Avoid div by zero
        mask = dist_sq / max_dist_sq
        mask = 1.0 - mask * strength # Invert: center is 1, edges decrease
        mask = np.clip(mask, 0, 1)

        # Apply mask to all channels
        vignetted = frame.astype(np.float32) * mask[:, :, np.newaxis]
        return vignetted.astype(np.uint8)

    def _effect_film_grain(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                           amount: float = 0.05, mono: bool = False, **kwargs) -> np.ndarray:
        """Adds simulated film grain."""
        amount = np.clip(amount, 0.0, 0.5)
        h, w, c = frame.shape

        # Generate noise
        if mono:
            noise = np.random.normal(0, amount * 128, (h, w, 1)).astype(np.int16)
            noise = np.repeat(noise, c, axis=2) # Apply same noise to all channels
        else:
            noise = np.random.normal(0, amount * 128, (h, w, c)).astype(np.int16)

        grainy = np.clip(frame.astype(np.int16) + noise, 0, 255)
        return grainy.astype(np.uint8)

    def _effect_sepia(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                      intensity: float = 1.0, **kwargs) -> np.ndarray:
        """Applies a sepia tone effect."""
        intensity = np.clip(intensity, 0.0, 1.0)
        # Sepia transformation matrix
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])

        sepia = cv2.transform(frame, kernel)
        sepia = np.clip(sepia, 0, 255).astype(np.uint8)

        # Blend with original based on intensity
        return cv2.addWeighted(frame, 1.0 - intensity, sepia, intensity, 0)

    def _effect_desaturate(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                           amount: float = 1.0, **kwargs) -> np.ndarray:
        """Desaturates the image (amount=1.0 is full black & white)."""
        amount = np.clip(amount, 0.0, 1.0)
        if amount < 1e-6: return frame
        if amount > 0.999: return cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) # Efficient B&W

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= (1.0 - amount) # Reduce saturation
        hsv = np.clip(hsv, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _effect_posterize(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                          levels: int = 4, **kwargs) -> np.ndarray:
        """Reduces the number of color levels in the image."""
        levels = max(2, int(levels))
        indices = np.arange(0, 256) # Lookup table
        divider = np.linspace(0, 255, levels + 1)[1] # Find thresholds
        quantiz = np.int0(np.linspace(0, 255, levels)) # Target levels
        color_levels = np.clip(np.int0(indices / divider), 0, levels - 1)
        palette = quantiz[color_levels] # Map original values to limited palette
        posterized = cv2.LUT(frame, palette.astype(np.uint8))
        return posterized

    def _effect_pixelate(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                         block_size: int = 10, **kwargs) -> np.ndarray:
        """Applies a pixelation effect."""
        block_size = max(2, int(block_size))
        h, w = frame.shape[:2]

        # Resize down
        temp = cv2.resize(frame, (w // block_size, h // block_size), interpolation=cv2.INTER_LINEAR)
        # Resize back up
        pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
        return pixelated

    def _effect_cartoon(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                        edge_threshold: int = 50, blur_ksize: int = 7, color_levels: int = 6, **kwargs) -> np.ndarray:
        """Applies a cartoon/cel-shading effect."""
        edge_threshold = max(10, int(edge_threshold))
        blur_ksize = int(blur_ksize) * 2 + 1 # Ensure odd
        color_levels = max(2, int(color_levels))

        # 1. Edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, 3) # Reduce noise before edge detection
        edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)

        # 2. Color quantization using bilateral filter
        # Downscale for faster filtering
        small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        # Repeatedly apply bilateral filter to smooth colors while preserving edges
        smoothed = small
        for _ in range(3): # Apply filter multiple times
             smoothed = cv2.bilateralFilter(smoothed, d=blur_ksize, sigmaColor=150, sigmaSpace=150)
        # Upscale
        smoothed = cv2.resize(smoothed, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)


        # Optional: Posterize the smoothed colors further
        # smoothed = self._effect_posterize(smoothed, 0, 1, levels=color_levels)

        # 3. Combine smoothed colors and edges
        cartoon = cv2.bitwise_and(smoothed, smoothed, mask=edges)
        return cartoon


    # --- Overlays & Compositing ---

    def _effect_overlay_image(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                              overlay_path: str, x: int = 0, y: int = 0,
                              scale: float = 1.0, opacity: float = 1.0,
                              blend_mode: str = 'normal', **kwargs) -> np.ndarray:
        """Overlays another image onto the frame."""
        opacity = np.clip(opacity, 0.0, 1.0)
        if not Path(overlay_path).exists():
            logger.warning(f"Overlay image not found: {overlay_path}")
            return frame

        overlay_img = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED) # Load with alpha if present
        if overlay_img is None:
             logger.warning(f"Could not read overlay image: {overlay_path}")
             return frame

        # Resize overlay
        if abs(scale - 1.0) > 1e-6:
            new_h = int(overlay_img.shape[0] * scale)
            new_w = int(overlay_img.shape[1] * scale)
            if new_h > 0 and new_w > 0:
                 overlay_img = cv2.resize(overlay_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                 logger.warning("Invalid overlay scale resulted in zero dimensions.")
                 return frame


        oh, ow = overlay_img.shape[:2]
        fh, fw = frame.shape[:2]

        # Define region of interest (ROI) on the main frame
        roi_x1, roi_y1 = max(0, x), max(0, y)
        roi_x2, roi_y2 = min(fw, x + ow), min(fh, y + oh)

        # Define corresponding region on the overlay
        ov_x1, ov_y1 = max(0, -x), max(0, -y)
        ov_x2, ov_y2 = min(ow, fw - x), min(oh, fh - y)

        # Check if ROI is valid
        if roi_x1 >= roi_x2 or roi_y1 >= roi_y2 or ov_x1 >= ov_x2 or ov_y1 >= ov_y2:
            return frame # Overlay is completely outside the frame

        roi_h, roi_w = roi_y2 - roi_y1, roi_x2 - roi_x1
        ov_h, ov_w = ov_y2 - ov_y1, ov_x2 - ov_x1

        # Adjust overlay dimensions slightly if needed due to rounding
        overlay_cropped = overlay_img[ov_y1:ov_y1+ov_h, ov_x1:ov_x1+ov_w]
        if overlay_cropped.shape[0] != roi_h or overlay_cropped.shape[1] != roi_w:
             overlay_cropped = cv2.resize(overlay_cropped, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)


        # Handle alpha channel
        if overlay_cropped.shape[2] == 4:
            alpha_mask = overlay_cropped[:, :, 3].astype(np.float32) / 255.0 * opacity
            overlay_bgr = overlay_cropped[:, :, :3]
            alpha_mask_3ch = cv2.cvtColor(alpha_mask, cv2.COLOR_GRAY2BGR)
        else:
            alpha_mask_3ch = np.full_like(overlay_cropped, opacity, dtype=np.float32)
            overlay_bgr = overlay_cropped

        # Blend
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2].astype(np.float32)
        overlay_bgr_float = overlay_bgr.astype(np.float32)

        # Apply blend mode (simplified)
        blended_roi: np.ndarray
        if blend_mode == 'add':
            blended_roi = roi + overlay_bgr_float * alpha_mask_3ch
        elif blend_mode == 'screen':
             blended_roi = 1 - (1 - roi/255.0) * (1 - (overlay_bgr_float * alpha_mask_3ch)/255.0)
             blended_roi *= 255.0
        elif blend_mode == 'multiply':
             blended_roi = (roi/255.0) * (overlay_bgr_float * alpha_mask_3ch / 255.0)
             blended_roi *= 255.0
        elif blend_mode == 'overlay':
             # Approximation
             low = 2 * (roi/255.0) * (overlay_bgr_float * alpha_mask_3ch / 255.0)
             high = 1 - 2 * (1 - roi/255.0) * (1 - (overlay_bgr_float * alpha_mask_3ch)/255.0)
             blended_roi = np.where(roi < 128, low, high) * 255.0
        else: # Normal blending
            blended_roi = roi * (1.0 - alpha_mask_3ch) + overlay_bgr_float * alpha_mask_3ch

        frame[roi_y1:roi_y2, roi_x1:roi_x2] = np.clip(blended_roi, 0, 255).astype(np.uint8)
        return frame


    def _effect_blend_mode(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                           layer_frame: np.ndarray = None, mode: str = 'normal', opacity: float = 1.0, **kwargs) -> np.ndarray:
        """Blends the frame with another layer using a specified mode."""
        opacity = np.clip(opacity, 0.0, 1.0)
        if layer_frame is None or layer_frame.shape != frame.shape:
             logger.warning("Blend mode requires a valid 'layer_frame' of the same dimensions.")
             return frame

        base = frame.astype(np.float32) / 255.0
        layer = layer_frame.astype(np.float32) / 255.0

        blended: np.ndarray
        mode = mode.lower()

        if mode == 'add':
            blended = base + layer
        elif mode == 'screen':
            blended = 1 - (1 - base) * (1 - layer)
        elif mode == 'multiply':
            blended = base * layer
        elif mode == 'overlay':
            blended = np.where(base < 0.5, 2 * base * layer, 1 - 2 * (1 - base) * (1 - layer))
        elif mode == 'difference':
             blended = np.abs(base - layer)
        elif mode == 'lighten':
             blended = np.maximum(base, layer)
        elif mode == 'darken':
             blended = np.minimum(base, layer)
        else: # normal
            blended = layer # If blending normally, layer is just placed on top

        # Apply opacity and combine with base
        result = base * (1.0 - opacity) + blended * opacity

        return (np.clip(result * 255.0, 0, 255)).astype(np.uint8)

    def _effect_chroma_key(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                          key_color_hex: str = '#00FF00', similarity: float = 0.1, softness: float = 0.05, **kwargs) -> np.ndarray:
        """Removes a specific color (e.g., green screen). Returns frame with alpha channel."""
        similarity = np.clip(similarity, 0.01, 1.0)
        softness = np.clip(softness, 0.0, 0.5)

        try:
             # Convert hex to BGR
             hex_color = key_color_hex.lstrip('#')
             key_bgr = np.array([int(hex_color[4:6], 16), int(hex_color[2:4], 16), int(hex_color[0:2], 16)])

             # Convert frame to HSV (better for color comparison)
             hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
             key_hsv = cv2.cvtColor(np.uint8([[key_bgr]]), cv2.COLOR_BGR2HSV)[0][0]

             # Define color range based on similarity and softness
             h_range = int(similarity * 90) # Hue range depends heavily on similarity
             s_range = int(max(50, similarity * 200)) # Wider range for saturation/value
             v_range = int(max(50, similarity * 200))

             lower_bound = np.array([max(0, key_hsv[0] - h_range), max(0, key_hsv[1] - s_range), max(0, key_hsv[2] - v_range)])
             upper_bound = np.array([min(179, key_hsv[0] + h_range), min(255, key_hsv[1] + s_range), min(255, key_hsv[2] + v_range)])

             # Create mask
             mask = cv2.inRange(hsv, lower_bound, upper_bound)

             # Invert mask (we want to keep non-keyed areas)
             alpha_mask = 255 - mask

             # Apply softness (blur the mask edges)
             if softness > 0:
                 blur_k = int(softness * 50) * 2 + 1 # Kernel size based on softness
                 alpha_mask = cv2.GaussianBlur(alpha_mask, (blur_k, blur_k), 0)

             # Add alpha channel to the frame
             b, g, r = cv2.split(frame)
             bgra = cv2.merge((b, g, r, alpha_mask))
             return bgra # Return BGRA frame

        except Exception as e:
             logger.error(f"Chroma key failed: {e}")
             # Return original frame with full alpha if error occurs
             b, g, r = cv2.split(frame)
             alpha = np.full_like(b, 255)
             return cv2.merge((b, g, r, alpha))


    def _effect_mask_shape(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                          shape: str = 'rectangle', cx: int = -1, cy: int = -1,
                          width: int = 100, height: int = 100, radius: int = 50,
                          feather: int = 0, invert: bool = False, **kwargs) -> np.ndarray:
        """Applies a static shape mask (rectangle or ellipse). Returns frame with alpha."""
        fh, fw = frame.shape[:2]
        if cx == -1: cx = fw // 2
        if cy == -1: cy = fh // 2
        feather = max(0, int(feather))

        mask = np.zeros((fh, fw), dtype=np.uint8)

        if shape == 'rectangle':
             x1 = max(0, cx - width // 2)
             y1 = max(0, cy - height // 2)
             x2 = min(fw, cx + width // 2)
             y2 = min(fh, cy + height // 2)
             cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        elif shape == 'ellipse':
             # Use radius for ellipse if width/height not specified for circle
             axes = (width // 2 if width > 0 else radius, height // 2 if height > 0 else radius)
             cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 255, -1)
        else:
             logger.warning(f"Unsupported mask shape: {shape}")
             mask.fill(255) # Default to full mask

        if invert:
            mask = 255 - mask

        # Apply feathering
        if feather > 0:
            feather_k = feather * 2 + 1
            mask = cv2.GaussianBlur(mask, (feather_k, feather_k), 0)

        # Add alpha channel
        b, g, r = cv2.split(frame)
        bgra = cv2.merge((b, g, r, mask))
        return bgra


    # --- Geometric ---

    def _effect_geometric_transform(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                                   scale: float = 1.0, angle: float = 0.0,
                                   tx: float = 0.0, ty: float = 0.0,
                                   center_x: float = 0.5, center_y: float = 0.5, **kwargs) -> np.ndarray:
        """Applies scale, rotation, and translation."""
        h, w = frame.shape[:2]
        cx, cy = int(center_x * w), int(center_y * h)

        # Get rotation matrix with scaling and translation
        M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
        # Apply translation to the matrix
        M[0, 2] += tx * w # tx is relative to width
        M[1, 2] += ty * h # ty is relative to height

        # Apply affine transformation
        transformed = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        return transformed

    def _effect_geometric_crop(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                               x: int = 0, y: int = 0, width: int = -1, height: int = -1, **kwargs) -> np.ndarray:
        """Crops the frame to the specified region."""
        fh, fw = frame.shape[:2]
        if width == -1: width = fw - x
        if height == -1: height = fh - y

        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x + width), min(fh, y + height)

        if x1 >= x2 or y1 >= y2:
             logger.warning("Invalid crop dimensions, returning original frame.")
             return frame

        cropped = frame[y1:y2, x1:x2]
        # Optional: Resize back to original dimensions? For now, return cropped size.
        return cropped

    def _effect_geometric_rotate(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                                 angle: float = 0.0, **kwargs) -> np.ndarray:
        """Rotates the frame around its center."""
        return self._effect_geometric_transform(frame, frame_idx, total_frames, angle=angle, **kwargs)

    def _effect_geometric_flip(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                               direction: str = 'horizontal', **kwargs) -> np.ndarray:
        """Flips the frame horizontally, vertically, or both."""
        if direction == 'horizontal':
            return cv2.flip(frame, 1)
        elif direction == 'vertical':
            return cv2.flip(frame, 0)
        elif direction == 'both':
            return cv2.flip(frame, -1)
        return frame


    # --- Special Effects ---

    def _effect_glitch_rgb_shift(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                                amount: float = 5.0, **kwargs) -> np.ndarray:
        """Applies RGB channel shifting glitch."""
        amount = max(0, int(amount))
        if amount == 0: return frame
        h, w = frame.shape[:2]
        glitched = frame.copy()

        # Shift Red channel horizontally
        glitched[:, amount:, 2] = frame[:, :-amount, 2]
        glitched[:, :amount, 2] = frame[:, -amount:, 2] # Wrap around

        # Shift Blue channel horizontally opposite
        glitched[:, :-amount, 0] = frame[:, amount:, 0]
        glitched[:, -amount:, 0] = frame[:, :amount, 0] # Wrap around

        return glitched

    def _effect_glitch_block_corruption(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                                      block_size: int = 8, corruption_prob: float = 0.05, **kwargs) -> np.ndarray:
        """Simulates digital block corruption."""
        block_size = max(2, int(block_size))
        corruption_prob = np.clip(corruption_prob, 0.0, 1.0)
        if corruption_prob == 0: return frame
        h, w = frame.shape[:2]
        glitched = frame.copy()

        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                if np.random.rand() < corruption_prob:
                    y_end, x_end = min(y + block_size, h), min(x + block_size, w)
                    # Replace block with random color or displaced block
                    if np.random.rand() < 0.5:
                         rand_color = np.random.randint(0, 256, 3, dtype=np.uint8)
                         glitched[y:y_end, x:x_end] = rand_color
                    else:
                         # Displace block
                         src_x = np.random.randint(0, max(1, w - block_size))
                         src_y = np.random.randint(0, max(1, h - block_size))
                         src_y_end, src_x_end = min(src_y + (y_end-y), h), min(src_x + (x_end-x), w)
                         block_h, block_w = src_y_end - src_y, src_x_end - src_x

                         # Ensure block sizes match for assignment
                         if block_h == (y_end-y) and block_w == (x_end-x):
                              glitched[y:y_end, x:x_end] = frame[src_y:src_y_end, src_x:src_x_end]


        return glitched

    def _effect_scan_lines(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                           line_freq: int = 4, strength: float = 0.1, **kwargs) -> np.ndarray:
        """Adds horizontal scan lines."""
        line_freq = max(1, int(line_freq))
        strength = np.clip(strength, 0.0, 1.0)
        h, w = frame.shape[:2]
        scanlined = frame.astype(np.float32)

        for y in range(0, h, line_freq):
            scanlined[y, :] *= (1.0 - strength) # Darken scan lines

        return np.clip(scanlined, 0, 255).astype(np.uint8)

    def _effect_light_leak(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                           x: float = 0.1, y: float = 0.1, radius_factor: float = 0.5,
                           strength: float = 0.7, color_hex: str = '#FFDD88', **kwargs) -> np.ndarray:
        """Simulates a light leak overlay."""
        strength = np.clip(strength, 0.0, 1.0)
        radius_factor = np.clip(radius_factor, 0.1, 2.0)
        h, w = frame.shape[:2]
        center_x, center_y = int(x * w), int(y * h)
        radius = min(w, h) * radius_factor

        # Convert hex color to BGR
        hex_color = color_hex.lstrip('#')
        leak_color = np.array([int(hex_color[4:6], 16), int(hex_color[2:4], 16), int(hex_color[0:2], 16)], dtype=np.float32)

        # Create gradient mask
        y_coords, x_coords = np.ogrid[:h, :w]
        dist_sq = (x_coords - center_x)**2 + (y_coords - center_y)**2
        mask = np.exp(-dist_sq / (2 * (radius**2))) * strength # Gaussian falloff
        mask = mask[:, :, np.newaxis] # Add channel dimension

        # Apply using Screen blend mode
        frame_float = frame.astype(np.float32) / 255.0
        leak_layer = np.ones_like(frame_float) * leak_color / 255.0

        # Screen blend: 1 - (1 - base) * (1 - layer * mask)
        screen_blended = 1.0 - (1.0 - frame_float) * (1.0 - leak_layer * mask)

        return (np.clip(screen_blended * 255.0, 0, 255)).astype(np.uint8)

    def _effect_lens_flare(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                          x: float = 0.2, y: float = 0.2, strength: float = 0.6,
                          num_ghosts: int = 5, color_hex: str = '#FFFFDD', **kwargs) -> np.ndarray:
        """Simulates a simple lens flare effect."""
        strength = np.clip(strength, 0.0, 1.0)
        num_ghosts = max(0, int(num_ghosts))
        h, w = frame.shape[:2]
        light_x, light_y = int(x * w), int(y * h)
        center_x, center_y = w // 2, h // 2

        # Convert hex color
        hex_color = color_hex.lstrip('#')
        flare_color = np.array([int(hex_color[4:6], 16), int(hex_color[2:4], 16), int(hex_color[0:2], 16)], dtype=np.float32)

        flare_layer = np.zeros_like(frame, dtype=np.float32)

        # Draw primary flare (halo)
        cv2.circle(flare_layer, (light_x, light_y), int(min(w,h)*0.1), flare_color * 0.5, -1)
        flare_layer = cv2.GaussianBlur(flare_layer, (99, 99), 0)

        # Draw ghosts (reflections) along the line from light source through center
        dx, dy = center_x - light_x, center_y - light_y

        for i in range(1, num_ghosts + 1):
            ghost_scale = i / num_ghosts
            ghost_x = int(center_x + dx * ghost_scale * 0.8) # Position along the line
            ghost_y = int(center_y + dy * ghost_scale * 0.8)
            ghost_radius = int(min(w,h) * 0.05 * (1.0 - ghost_scale * 0.5)) # Smaller further away
            ghost_intensity = strength * 0.3 * (1.0 - ghost_scale * 0.7)

            if ghost_radius > 0:
                 ghost_color_variation = flare_color * (0.8 + np.random.rand(3)*0.4) # Vary color slightly
                 temp_ghost_layer = np.zeros_like(frame, dtype=np.float32)
                 cv2.circle(temp_ghost_layer, (ghost_x, ghost_y), ghost_radius, ghost_color_variation, -1)
                 temp_ghost_layer = cv2.GaussianBlur(temp_ghost_layer, (33, 33), 0)
                 flare_layer += temp_ghost_layer * ghost_intensity


        # Blend using Add or Screen
        result = frame.astype(np.float32) + flare_layer * strength * 1.5 # Additive blend
        # Alternative Screen blend:
        # result = 1.0 - (1.0 - frame.astype(np.float32)/255.0) * (1.0 - flare_layer * strength / 255.0)
        # result *= 255.0

        return np.clip(result, 0, 255).astype(np.uint8)


    # --- Transitions (Applicable as frame effects over a duration) ---
    # Note: These usually require context from the TimelineGenerator (previous/next frame)
    # Here they act as effects applied *to the current frame* based on progress

    def _effect_transition_fade(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                                duration: int = 30, direction: str = 'out', color_hex: str = '#000000', **kwargs) -> np.ndarray:
        """Applies fade in or fade out relative to frame_idx and duration."""
        progress = np.clip(frame_idx / max(1, duration -1), 0.0, 1.0)

        # Convert hex color
        hex_color = color_hex.lstrip('#')
        fade_color = np.array([int(hex_color[4:6], 16), int(hex_color[2:4], 16), int(hex_color[0:2], 16)], dtype=np.float32)

        if direction == 'in':
            alpha = progress
        else: # out
            alpha = 1.0 - progress

        faded = frame.astype(np.float32) * alpha + fade_color * (1.0 - alpha)
        return np.clip(faded, 0, 255).astype(np.uint8)

    def _effect_transition_dissolve(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                                    next_frame_provider: Callable[[int], Optional[np.ndarray]] = None,
                                    duration: int = 30, **kwargs) -> np.ndarray:
        """Applies dissolve based on progress. Requires a way to get the next frame."""
        if next_frame_provider is None:
             logger.warning("Dissolve effect needs 'next_frame_provider'. Fading out instead.")
             return self._effect_transition_fade(frame, frame_idx, total_frames, duration=duration, direction='out', **kwargs)

        progress = np.clip(frame_idx / max(1, duration - 1), 0.0, 1.0)
        next_frame = next_frame_provider(frame_idx) # Get the corresponding frame from the *next* clip

        if next_frame is None or next_frame.shape != frame.shape:
             logger.warning("Invalid next_frame provided for dissolve. Fading out instead.")
             return self._effect_transition_fade(frame, frame_idx, total_frames, duration=duration, direction='out', **kwargs)

        blended = cv2.addWeighted(frame, 1.0 - progress, next_frame, progress, 0)
        return blended

    def _effect_transition_wipe(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                                duration: int = 30, direction: str = 'left_to_right',
                                next_frame_provider: Callable[[int], Optional[np.ndarray]] = None,
                                **kwargs) -> np.ndarray:
        """Applies a wipe transition based on progress."""
        if next_frame_provider is None:
             logger.warning("Wipe effect needs 'next_frame_provider'. Returning original.")
             return frame

        progress = np.clip(frame_idx / max(1, duration - 1), 0.0, 1.0)
        next_frame = next_frame_provider(frame_idx)

        if next_frame is None or next_frame.shape != frame.shape:
             logger.warning("Invalid next_frame provided for wipe. Returning original.")
             return frame

        h, w = frame.shape[:2]
        result = frame.copy()

        if direction == 'left_to_right':
            split_x = int(progress * w)
            if split_x < w: result[:, split_x:] = next_frame[:, split_x:]
        elif direction == 'right_to_left':
            split_x = int((1.0 - progress) * w)
            if split_x > 0: result[:, :split_x] = next_frame[:, :split_x]
        elif direction == 'top_to_bottom':
            split_y = int(progress * h)
            if split_y < h: result[split_y:, :] = next_frame[split_y:, :]
        elif direction == 'bottom_to_top':
            split_y = int((1.0 - progress) * h)
            if split_y > 0: result[:split_y, :] = next_frame[:split_y, :]
        else:
             logger.warning(f"Unsupported wipe direction: {direction}")
             return self._effect_transition_dissolve(frame, frame_idx, total_frames, next_frame_provider, duration, **kwargs) # Fallback to dissolve


        return result


    # --- Utility ---
    def _apply_lut(self, frame: np.ndarray, lut: np.ndarray) -> np.ndarray:
        """Applies a 1D LUT to each channel."""
        # Ensure lut is uint8
        if lut.dtype != np.uint8:
             lut = np.clip(lut * 255, 0, 255).astype(np.uint8)
        # Ensure lut has correct shape (256 entries)
        if len(lut) != 256:
             # Interpolate if needed (basic linear)
             x_orig = np.linspace(0, 255, len(lut))
             x_new = np.arange(256)
             lut = np.interp(x_new, x_orig, lut).astype(np.uint8)

        # Apply LUT
        return cv2.LUT(frame, lut)

# --- END OF CLASS ---

# Helper function (if needed outside the class)
def apply_effects_to_frames(frames: List[np.ndarray], effects_plan: List[Dict]) -> List[np.ndarray]:
    """
    Applies a sequence of effects based on a plan to a list of frames.
    This function would typically live in the TimelineGenerator or a similar orchestration class.
    """
    generator = AdvancedEffectGenerator() # Use default config
    processed_frames = frames.copy()
    total_frames = len(frames)

    for frame_idx in track(range(total_frames), description="Applying Effects"):
        current_frame = processed_frames[frame_idx]
        for effect_info in effects_plan:
            # Check if effect applies to this frame (based on start/end times or indices)
            start_frame = effect_info.get('start_frame', 0)
            end_frame = effect_info.get('end_frame', total_frames - 1)
            if start_frame <= frame_idx <= end_frame:
                effect_name = effect_info.get('type')
                params = effect_info.get('params', {})
                if effect_name:
                    # Provide necessary context for keyframing/transitions
                    params['next_frame_provider'] = lambda idx: processed_frames[idx+1] if idx+1 < total_frames else None # Example provider

                    current_frame = generator.apply_effect(
                        current_frame,
                        effect_name,
                        frame_idx=frame_idx,
                        total_frames=total_frames,
                        **params
                    )
        processed_frames[frame_idx] = current_frame

    return processed_frames

if __name__ == '__main__':
    # Example usage and testing
    print("Testing AdvancedEffectGenerator...")
    generator = AdvancedEffectGenerator()
    print(f"Available Effects: {len(generator.get_available_effects())}")
    # print(generator.get_available_effects())

    # Create a dummy frame
    dummy_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    cv2.putText(dummy_frame, "Original", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Test a few effects
    cv2.imshow("Original", dummy_frame)

    # Test basic color
    colored = generator.apply_effect(dummy_frame, 'basic_color', brightness=20, contrast=1.2, saturation=1.5)
    cv2.imshow("Basic Color", colored)

    # Test Gaussian blur
    blurred = generator.apply_effect(dummy_frame, 'blur_gaussian', strength=15)
    cv2.imshow("Gaussian Blur", blurred)

    # Test Vignette
    vignetted = generator.apply_effect(dummy_frame, 'vignette', strength=0.8, radius_factor=0.6)
    cv2.imshow("Vignette", vignetted)

    # Test Pinch
    pinched = generator.apply_effect(dummy_frame, 'distort_pinch_punch', strength=0.7, radius_factor=0.4)
    cv2.imshow("Pinch", pinched)

    # Test Keyframing (simple example)
    print("\nTesting Keyframing (Brightness)...")
    keyframed_params = {
        'brightness': [
            {'frame': 0, 'value': -50},
            {'frame': 15, 'value': 50},
            {'frame': 30, 'value': 0}
        ],
        'contrast': 1.1 # Static param
    }
    total_demo_frames = 31
    for i in range(total_demo_frames):
        kf_frame = generator.apply_effect(dummy_frame, 'basic_color', frame_idx=i, total_frames=total_demo_frames, **keyframed_params)
        cv2.putText(kf_frame, f"Frame {i}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Keyframed Brightness", kf_frame)
        if cv2.waitKey(100) & 0xFF == ord('q'): # Slow down for visibility
             break
    print("Keyframing Test Complete.")


    cv2.waitKey(0)
    cv2.destroyAllWindows()