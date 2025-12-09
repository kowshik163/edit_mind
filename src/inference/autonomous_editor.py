"""
Autonomous Video Editor - High-level inference interface
Provides easy-to-use interface for autonomous video editing
"""

import os
import cv2
import torch
import numpy as np
import logging
import json
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

try:
    from src.generation.self_coding_engine import SelfCodingVideoEditor
except ImportError:
    try:
        from ..generation.self_coding_engine import SelfCodingVideoEditor
    except ImportError:
        SelfCodingVideoEditor = None

logger = logging.getLogger(__name__)


class AutonomousVideoEditor:
    """High-level interface for autonomous video editing"""
    
    def __init__(self, ai_model, effect_generator, config: Dict[str, Any], self_coding_engine: Optional[Any] = None):
        """
        Initialize the autonomous video editor
        
        Args:
            ai_model: The trained HybridVideoAI model
            effect_generator: AdvancedEffectGenerator instance
            config: Configuration dictionary
            self_coding_engine: Optional SelfCodingVideoEditor instance
        """
        self.ai_model = ai_model
        self.effect_generator = effect_generator
        self.config = config
        self.self_coding_engine = self_coding_engine
        
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
            edit_plan = self._generate_edit_plan(editing_prompt, frames, video_path)
            
            # Apply edits based on the plan
            edited_frames = self._apply_edits(frames, edit_plan, fps)
            
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
    
    def _generate_edit_plan(self, prompt: str, frames: List[np.ndarray], video_path: str = None) -> Dict[str, Any]:
        """Generate editing plan based on prompt and video analysis using Gemini API"""
        logger.info("Generating edit plan using Gemini API...")
        
        try:
            import google.generativeai as genai
            import time
            from google.api_core import exceptions
            
            # Configure Gemini API
            api_key = os.environ.get("AIzaSyAotS0WoSyBb4jc9MxOg-joKQ3xJr5ughM")
            if not api_key:
                logger.warning("GEMINI_API_KEY not found in environment variables. Falling back to rule-based planner.")
                return self._generate_rule_based_plan(prompt)
                
            genai.configure(api_key=api_key)
            
            # Upload video to Gemini
            if not video_path or not os.path.exists(video_path):
                logger.warning("Video path not provided or invalid. Falling back to rule-based planner.")
                return self._generate_rule_based_plan(prompt)
                
            logger.info(f"Uploading video to Gemini: {video_path}")
            video_file = genai.upload_file(path=video_path)
            
            # Wait for processing
            while video_file.state.name == "PROCESSING":
                logger.info("Waiting for video processing...")
                time.sleep(2)
                video_file = genai.get_file(video_file.name)
                
            if video_file.state.name == "FAILED":
                logger.error("Video processing failed in Gemini.")
                return self._generate_rule_based_plan(prompt)
                
            logger.info("Video processing complete.")
            
            # Define available models in order of preference
            models_to_try = [
                "gemini-1.5-pro",
                "gemini-1.5-flash",
                "gemini-1.0-pro-vision"
            ]
            
            # Load memory context if available
            memory_context = ""
            memory_file = Path("data/editor_memory.json")
            if memory_file.exists():
                try:
                    with open(memory_file, 'r') as f:
                        memory_data = json.load(f)
                        memory_context = f"\n### Editor Memory & Preferences\n{json.dumps(memory_data, indent=2)}\n"
                        logger.info("Loaded editor memory context.")
                except Exception as e:
                    logger.warning(f"Failed to load editor memory: {e}")

            # Construct the comprehensive system prompt
            system_prompt = """
            You are an expert Autonomous Video Editor AI. Your goal is to analyze the provided video media and generate a highly detailed, professional editing plan based on the user's intent.
            """ + memory_context + """
            
            ### System Capabilities & Available Models
            We are equipped with an Advanced Effect Generator capable of the following operations:
            
            **1. Color & Lighting:**
            - `basic_color`: Adjust brightness, contrast, saturation.
            - `hue_shift`: Shift global hue.
            - `color_temp_tint`: Adjust temperature (warm/cool) and tint (green/magenta).
            - `vibrance`: Smart saturation boost.
            - `shadows_highlights`: Independent control of shadow and highlight recovery.
            - `curves_poly`: Polynomial curve adjustment.
            - `lut_apply`: Apply Look-Up Tables (LUTs).
            
            **2. Blur & Sharpen:**
            - `blur_gaussian`, `blur_box`, `blur_median`: Standard blurs.
            - `blur_motion`: Directional motion blur.
            - `blur_radial`: Zoom/spin blur.
            - `blur_bokeh`: Cinematic depth-of-field simulation.
            - `sharpen_basic`, `sharpen_unsharpmask`: Detail enhancement.
            
            **3. Distortion & FX:**
            - `distort_ripple`, `distort_wave`: Water/wave effects.
            - `distort_pinch_punch`, `distort_twirl`, `distort_fisheye`: Geometric distortions.
            - `vignette`: Cinematic corner darkening.
            - `glitch_analog`, `glitch_digital`: Glitch art effects.
            
            **4. Stylization:**
            - `style_pixelate`, `style_halftone`, `style_sketch`, `style_cartoon`.
            - `style_emboss`, `style_edge_detect`.
            
            **5. Transformations:**
            - `transform_zoom`, `transform_rotate`, `transform_shake`.
            
            ### Custom Code Generation
            If the user requests an effect NOT listed above (e.g., "matrix rain", "fire particles", "advanced object removal"), you must specify it in the `custom_code_requests` section.
            Provide a detailed technical description of how this effect should be implemented in Python using OpenCV/NumPy. The Self-Coding Engine will use this description to generate the code.
            
            ### Task Instructions
            1.  **Analyze the Media**: Provide a detailed breakdown of the video content, including:
                -   **Scene Description**: What is happening?
                -   **Objects/Subjects**: List key elements detected.
                -   **Mood/Atmosphere**: The emotional tone (e.g., energetic, melancholic, professional).
                -   **Technical Quality**: Lighting, stability, color balance.
            
            2.  **Interpret User Intent**: Analyze the user's prompt to understand the desired outcome (e.g., "make it cinematic", "fast-paced action", "vintage vlog").
            
            3.  **Generate Editing Plan**: Create a JSON object detailing the specific edits to apply.
            
            ### Output Format (JSON)
            Return ONLY a valid JSON object with this structure:
            {
                "analysis": {
                    "summary": "Detailed summary of the video content.",
                    "scenes": [
                        {"start": 0.0, "end": 5.0, "description": "Opening shot of a city street, daytime."},
                        {"start": 5.0, "end": 10.0, "description": "Close up of subject walking."}
                    ],
                    "objects": ["car", "building", "person", "tree"],
                    "mood": "Urban, busy, neutral lighting",
                    "technical_notes": "Slightly shaky footage, good exposure."
                },
                "editing_intent": "User wants a high-energy urban montage with a cyberpunk aesthetic.",
                "plan": {
                    "effects": [
                        {
                            "name": "color_grade_cinematic", 
                            "description": "Apply teal and orange look",
                            "parameters": {"intensity": 0.8}
                        },
                        {
                            "name": "vignette",
                            "parameters": {"intensity": 0.5}
                        }
                    ],
                    "custom_code_requests": [
                        {
                            "name": "matrix_rain_overlay",
                            "description": "Generate falling green characters similar to The Matrix. Use OpenCV to draw random characters in columns with varying speeds and fading trails. Overlay on the video with additive blending.",
                            "parameters": {"density": 0.5, "speed": 1.0}
                        }
                    ],
                    "cuts": [
                        {"start": 0.0, "end": 4.5, "description": "Trim start to remove camera setup"},
                        {"start": 5.2, "end": 9.8, "description": "Keep walking segment"}
                    ],
                    "transitions": [
                        {"type": "fade", "duration": 0.5, "position": "start"},
                        {"type": "glitch_digital", "duration": 0.3, "position": 4.5}
                    ],
                    "timing": {
                        "speed_factor": 1.0,
                        "slow_motion_segments": [{"start": 6.0, "end": 8.0, "factor": 0.5}]
                    },
                    "audio": {
                        "volume_adjust": 1.0,
                        "background_track_genre": "Synthwave"
                    }
                }
            }
            """
            
            user_message = f"User Editing Request: {prompt}"
            
            # Try models in sequence
            for model_name in models_to_try:
                try:
                    logger.info(f"Requesting editing plan from Gemini model: {model_name}...")
                    model = genai.GenerativeModel(model_name=model_name)
                    
                    # Generate content
                    response = model.generate_content([video_file, system_prompt, user_message])
                    
                    # Parse JSON response
                    response_text = response.text
                    if "```json" in response_text:
                        json_str = response_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in response_text:
                        json_str = response_text.split("```")[1].split("```")[0].strip()
                    else:
                        json_str = response_text.strip()
                        
                    plan = json.loads(json_str)
                    logger.info(f"Gemini ({model_name}) generated plan successfully.")
                    
                    # Normalize plan structure for internal use
                    # Map the detailed 'plan' structure back to the flat structure expected by _apply_edits
                    flat_plan = {
                        'effects': [e['name'] if isinstance(e, dict) else e for e in plan.get('plan', {}).get('effects', [])],
                        'custom_code_requests': plan.get('plan', {}).get('custom_code_requests', []),
                        'cuts': plan.get('plan', {}).get('cuts', []),
                        'transitions': plan.get('plan', {}).get('transitions', []),
                        'timing': plan.get('plan', {}).get('timing', {}),
                        'analysis': plan.get('analysis', {}),
                        'detailed_plan': plan # Keep the full detailed plan for reference/logging
                    }
                    
                    return flat_plan
                    
                except exceptions.ResourceExhausted:
                    logger.warning(f"Rate limit exceeded for model {model_name}. Trying next model...")
                    continue
                except Exception as e:
                    logger.error(f"Error with model {model_name}: {e}")
                    continue
            
            logger.error("All Gemini models failed or rate limited.")
            return self._generate_rule_based_plan(prompt)
                
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return self._generate_rule_based_plan(prompt)

    def _generate_rule_based_plan(self, prompt: str) -> Dict[str, Any]:
        """Fallback rule-based plan generation"""
        logger.info("Using fallback rule-based planner...")
        
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
            
        logger.info(f"Generated fallback plan: {plan}")
        return plan
    
    def _apply_edits(self, frames: List[np.ndarray], edit_plan: Dict[str, Any], fps: float = 30.0) -> List[np.ndarray]:
        """Apply edits based on the generated plan"""
        logger.info("Applying edits...")
        
        # Handle cuts/segments if specified
        if edit_plan.get('cuts'):
            logger.info(f"Applying {len(edit_plan['cuts'])} cuts...")
            segmented_frames = []
            for cut in edit_plan['cuts']:
                start_time = cut.get('start', 0.0)
                end_time = cut.get('end', len(frames) / fps)
                
                start_frame = int(start_time * fps)
                end_frame = int(end_time * fps)
                
                # Clamp to valid range
                start_frame = max(0, min(start_frame, len(frames)))
                end_frame = max(0, min(end_frame, len(frames)))
                
                if start_frame < end_frame:
                    segment = frames[start_frame:end_frame]
                    segmented_frames.extend(segment)
            
            if segmented_frames:
                edited_frames = segmented_frames
            else:
                logger.warning("Cuts resulted in empty video, using original frames.")
                edited_frames = frames.copy()
        else:
            edited_frames = frames.copy()
            
        total_frames = len(edited_frames)
        
        # Apply custom code effects if available
        if edit_plan.get('custom_code_requests') and self.self_coding_engine:
            logger.info(f"Processing {len(edit_plan['custom_code_requests'])} custom code requests...")
            for request in edit_plan['custom_code_requests']:
                try:
                    effect_name = request.get('name', 'custom_effect')
                    description = request.get('description', '')
                    parameters = request.get('parameters', {})
                    
                    logger.info(f"Generating and applying custom code for: {effect_name}")
                    
                    # Use the SelfCodingEngine's high-level method to generate and apply the effect
                    # This handles code generation, validation, testing, and application to all frames
                    edited_frames = self.self_coding_engine.apply_generated_effect(
                        frames=edited_frames,
                        effect_description=description,
                        **parameters
                    )
                        
                except Exception as e:
                    logger.error(f"Error processing custom request {request}: {e}")
        elif edit_plan.get('custom_code_requests') and not self.self_coding_engine:
            logger.warning("Custom code requests present but SelfCodingEngine not initialized.")

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
