"""
Autonomous Video Editor - High-level inference interface
Provides easy-to-use interface for autonomous video editing
"""

import os
import cv2
import torch
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class AutonomousVideoEditor:
    """High-level interface for autonomous video editing"""
    
    def __init__(self, ai_model, effect_generator, config: Dict[str, Any]):
        """
        Initialize the autonomous video editor
        
        Args:
            ai_model: The trained HybridVideoAI model
            effect_generator: AdvancedEffectGenerator instance
            config: Configuration dictionary
        """
        self.ai_model = ai_model
        self.effect_generator = effect_generator
        self.config = config
        
    def edit_video(self, video_path: str, editing_prompt: str, output_path: str) -> Dict[str, Any]:
        """
        Edit a video based on a natural language prompt
        
        Args:
            video_path: Path to input video
            editing_prompt: Natural language description of desired edits
            output_path: Path for output video
            
        Returns:
            Dictionary with editing results and metadata
        """
        logger.info(f"Starting autonomous edit of {video_path}")
        logger.info(f"Prompt: {editing_prompt}")
        
        try:
            # Load video
            frames, fps, audio = self._load_video(video_path)
            original_frame_count = len(frames)
            
            # Analyze editing prompt and generate plan
            edit_plan = self._generate_edit_plan(editing_prompt, frames)
            
            # Apply edits based on the plan
            edited_frames = self._apply_edits(frames, edit_plan)
            
            # Save edited video
            self._save_video(edited_frames, output_path, fps, audio)
            
            result = {
                'success': True,
                'input_path': video_path,
                'output_path': output_path,
                'original_frames': original_frame_count,
                'edited_frames': len(edited_frames),
                'edit_plan': edit_plan,
                'editing_prompt': editing_prompt
            }
            
            logger.info(f"Successfully edited video: {output_path}")
            return result
            
        except Exception as e:
            logger.error(f"Video editing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'input_path': video_path,
                'editing_prompt': editing_prompt
            }
    
    def _load_video(self, video_path: str) -> Tuple[List[np.ndarray], float, Optional[np.ndarray]]:
        """Load video frames and metadata"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        
        logger.info(f"Loading video: {video_path} (FPS: {fps})")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            
        cap.release()
        
        logger.info(f"Loaded {len(frames)} frames")
        
        # Load audio separately using moviepy
        audio = None
        try:
            try:
                import moviepy.editor as mp
            except ImportError:
                logger.warning("moviepy not available, skipping audio extraction")
                return frames, fps, None
                
            video_clip = mp.VideoFileClip(video_path)
            if video_clip.audio is not None:
                # Extract audio as numpy array
                audio_array = video_clip.audio.to_soundarray()
                audio = {
                    'data': audio_array,
                    'fps': video_clip.audio.fps,
                    'duration': video_clip.audio.duration
                }
                logger.info(f"Loaded audio: {audio_array.shape} at {video_clip.audio.fps} Hz")
            video_clip.close()
        except Exception as e:
            logger.warning(f"Could not load audio from video: {e}")
            audio = None
        
        return frames, fps, audio
    
    def _generate_edit_plan(self, prompt: str, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Generate editing plan based on prompt and video analysis"""
        logger.info("Generating edit plan...")
        
        # Simple rule-based plan generation (can be replaced with AI model inference)
        plan = {
            'effects': [],
            'cuts': [],
            'transitions': [],
            'timing': {}
        }
        
        prompt_lower = prompt.lower()
        
        # Detect requested effects from prompt
        if 'cinematic' in prompt_lower:
            plan['effects'].extend(['color_grade_cinematic', 'dramatic_shadows'])
        if 'vintage' in prompt_lower or 'retro' in prompt_lower:
            plan['effects'].append('vintage_film')
        if 'cyberpunk' in prompt_lower or 'sci-fi' in prompt_lower:
            plan['effects'].append('cyberpunk')
        if 'bright' in prompt_lower or 'vibrant' in prompt_lower:
            plan['effects'].append('vibrant_colors')
        if 'dark' in prompt_lower or 'moody' in prompt_lower:
            plan['effects'].append('dramatic_shadows')
        if 'slow' in prompt_lower:
            plan['timing']['slow_motion'] = True
        if 'fast' in prompt_lower or 'quick' in prompt_lower:
            plan['timing']['fast_motion'] = True
            
        # Add fade in/out for professional look
        if 'professional' in prompt_lower or 'cinematic' in prompt_lower:
            plan['effects'].extend(['fade_in', 'fade_out'])
            
        # Default to basic color grading if no specific effects requested
        if not plan['effects']:
            plan['effects'].append('color_grade_cinematic')
            
        logger.info(f"Generated plan: {plan}")
        return plan
    
    def _apply_edits(self, frames: List[np.ndarray], edit_plan: Dict[str, Any]) -> List[np.ndarray]:
        """Apply edits based on the generated plan"""
        logger.info("Applying edits...")
        
        edited_frames = frames.copy()
        total_frames = len(edited_frames)
        
        # Apply effects
        for effect_name in edit_plan['effects']:
            logger.info(f"Applying effect: {effect_name}")
            
            # Handle special effects that apply to specific frame ranges
            if effect_name == 'fade_in':
                fade_length = min(30, total_frames // 4)  # First 1 second or 25% of video
                for i in range(fade_length):
                    alpha = i / fade_length
                    edited_frames[i] = self._blend_frames(
                        np.zeros_like(edited_frames[i]), 
                        edited_frames[i], 
                        alpha
                    )
            elif effect_name == 'fade_out':
                fade_length = min(30, total_frames // 4)  # Last 1 second or 25% of video
                start_idx = total_frames - fade_length
                for i in range(fade_length):
                    alpha = 1.0 - (i / fade_length)
                    frame_idx = start_idx + i
                    edited_frames[frame_idx] = self._blend_frames(
                        np.zeros_like(edited_frames[frame_idx]),
                        edited_frames[frame_idx],
                        alpha
                    )
            else:
                # Apply effect to all frames
                for i, frame in enumerate(edited_frames):
                    try:
                        edited_frames[i] = self.effect_generator.apply_effect(frame, effect_name)
                    except Exception as e:
                        logger.warning(f"Failed to apply {effect_name} to frame {i}: {e}")
                        # Keep original frame on error
                        continue
                        
        # Apply timing changes
        if edit_plan['timing'].get('slow_motion'):
            # Duplicate frames for slow motion effect
            slow_frames = []
            for frame in edited_frames:
                slow_frames.extend([frame] * 2)  # 2x slower
            edited_frames = slow_frames
            
        elif edit_plan['timing'].get('fast_motion'):
            # Skip frames for fast motion effect  
            edited_frames = edited_frames[::2]  # 2x faster
            
        logger.info(f"Applied edits, result: {len(edited_frames)} frames")
        return edited_frames
    
    def _blend_frames(self, frame1: np.ndarray, frame2: np.ndarray, alpha: float) -> np.ndarray:
        """Blend two frames with given alpha"""
        return cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
    
    def _save_video(self, frames: List[np.ndarray], output_path: str, fps: float, audio: Optional[np.ndarray] = None):
        """Save edited frames as video"""
        if not frames:
            raise ValueError("No frames to save")
            
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        logger.info(f"Saving video: {output_path} ({width}x{height} @ {fps} FPS)")
        
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
            
        out.release()
        
        # Add audio back if available
        if audio is not None:
            try:
                try:
                    import moviepy.editor as mp
                    import tempfile
                except ImportError:
                    logger.warning("moviepy not available, saving video without audio")
                    return
                
                # Create temporary video file without audio
                temp_video = tempfile.mktemp(suffix='.mp4')
                fourcc_temp = cv2.VideoWriter_fourcc(*'mp4v')
                temp_out = cv2.VideoWriter(temp_video, fourcc_temp, fps, (width, height))
                
                for frame in frames:
                    temp_out.write(frame)
                temp_out.release()
                
                # Load the temporary video and add audio
                video_clip = mp.VideoFileClip(temp_video)
                
                # Create audio clip from numpy array
                audio_clip = mp.AudioArrayClip(audio['data'], fps=audio['fps'])
                
                # Combine video and audio
                final_clip = video_clip.set_audio(audio_clip)
                final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
                
                # Cleanup
                video_clip.close()
                audio_clip.close()
                final_clip.close()
                os.unlink(temp_video)
                
                logger.info(f"Video with audio saved successfully: {output_path}")
                
            except Exception as e:
                logger.warning(f"Could not add audio to video, saving video only: {e}")
        else:
            logger.info(f"Video saved successfully: {output_path}")
    
    def get_available_effects(self) -> List[str]:
        """Get list of available effects"""
        return self.effect_generator.get_available_effects()
    
    def preview_effect(self, frame: np.ndarray, effect_name: str) -> np.ndarray:
        """Preview an effect on a single frame"""
        return self.effect_generator.apply_effect(frame, effect_name)


# Convenience functions for common operations

def quick_edit(video_path: str, prompt: str, output_path: str, config: Optional[Dict] = None):
    """Quick video editing function"""
    from ..generation.effect_generator import AdvancedEffectGenerator
    
    if config is None:
        config = {'effects': {'quality': 'high', 'gpu_acceleration': True}}
    
    effect_generator = AdvancedEffectGenerator(
        quality=config['effects']['quality'],
        gpu_acceleration=config['effects']['gpu_acceleration']
    )
    
    # Create a simple autonomous editor without AI model for basic effects
    editor = AutonomousVideoEditor(
        ai_model=None,  # Not needed for basic effects
        effect_generator=effect_generator,
        config=config
    )
    
    return editor.edit_video(video_path, prompt, output_path)


def batch_process_videos(video_paths: List[str], prompt: str, output_dir: str, config: Optional[Dict] = None):
    """Process multiple videos with the same prompt"""
    results = []
    
    for i, video_path in enumerate(video_paths):
        filename = os.path.basename(video_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{name}_edited{ext}")
        
        logger.info(f"Processing video {i+1}/{len(video_paths)}: {video_path}")
        
        result = quick_edit(video_path, prompt, output_path, config)
        results.append(result)
        
    return results
