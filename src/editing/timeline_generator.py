"""
Timeline Generator - Renders final videos from editing timelines with FFmpeg integration
"""

import torch
import numpy as np
import ffmpeg
import logging
import tempfile
import os
from typing import Dict, List, Any, Optional, Tuple
from omegaconf import DictConfig
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip
from moviepy.video.fx import resize, fadein, fadeout
from moviepy.audio.fx.audio_fadein import audio_fadein
from moviepy.audio.fx.audio_fadeout import audio_fadeout

logger = logging.getLogger(__name__)


class TimelineGenerator:
    """Advanced timeline generation and video rendering with FFmpeg and MoviePy"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        
        # Video rendering settings
        self.output_fps = config.get('output_fps', 30)
        self.output_resolution = config.get('output_resolution', (1920, 1080))
        self.video_codec = config.get('video_codec', 'libx264')
        self.audio_codec = config.get('audio_codec', 'aac')
        self.output_quality = config.get('output_quality', 'medium')
        
        # Transition settings
        self.default_transition_duration = config.get('default_transition_duration', 0.5)
        self.fade_duration = config.get('fade_duration', 0.3)
        
        logger.info("TimelineGenerator initialized")
    
    def decode_timeline(self, timeline_logits: torch.Tensor, video_duration: float) -> Dict[str, Any]:
        """Decode timeline from model logits"""
        try:
            # Convert logits to probabilities
            timeline_probs = torch.softmax(timeline_logits, dim=-1)
            
            # Extract cut points (assuming binary classification for cuts)
            cut_threshold = 0.5
            cut_probs = timeline_probs[:, 0] if timeline_logits.dim() > 1 else timeline_probs
            cut_points = torch.where(cut_probs > cut_threshold)[0].tolist()
            
            # Convert frame indices to time stamps (assuming 30 fps)
            fps = 30
            cut_times = [idx / fps for idx in cut_points]
            
            # Generate segments between cut points
            segments = []
            if cut_times:
                # Add start and end points
                all_times = [0.0] + cut_times + [video_duration]
                all_times = sorted(list(set(all_times)))  # Remove duplicates and sort
                
                for i in range(len(all_times) - 1):
                    segments.append({
                        'start': all_times[i],
                        'end': all_times[i + 1],
                        'duration': all_times[i + 1] - all_times[i]
                    })
            else:
                # No cuts found, use entire video
                segments = [{'start': 0.0, 'end': video_duration, 'duration': video_duration}]
            
            # Generate transitions (simple fade for now)
            transitions = []
            for i in range(len(segments) - 1):
                transitions.append({
                    'type': 'fade',
                    'duration': self.default_transition_duration,
                    'from_segment': i,
                    'to_segment': i + 1
                })
            
            # Generate effects (placeholder)
            effects = self._generate_effects(segments)
            
            return {
                'segments': segments,
                'cut_points': cut_times,
                'transitions': transitions,
                'effects': effects,
                'total_duration': sum(seg['duration'] for seg in segments)
            }
            
        except Exception as e:
            logger.error(f"Error decoding timeline: {e}")
            # Return simple single-segment timeline
            return {
                'segments': [{'start': 0.0, 'end': video_duration, 'duration': video_duration}],
                'cut_points': [],
                'transitions': [],
                'effects': [],
                'total_duration': video_duration
            }
    
    def _generate_effects(self, segments: List[Dict]) -> List[Dict]:
        """Generate effects for segments based on simple heuristics"""
        effects = []
        
        for i, segment in enumerate(segments):
            # Add fade in for first segment
            if i == 0:
                effects.append({
                    'type': 'fade_in',
                    'segment_idx': i,
                    'duration': self.fade_duration,
                    'start_time': segment['start']
                })
            
            # Add fade out for last segment
            if i == len(segments) - 1:
                effects.append({
                    'type': 'fade_out',
                    'segment_idx': i,
                    'duration': self.fade_duration,
                    'start_time': segment['end'] - self.fade_duration
                })
            
            # Add color correction for short segments (likely highlights)
            if segment['duration'] < 2.0:
                effects.append({
                    'type': 'color_enhance',
                    'segment_idx': i,
                    'intensity': 1.2,
                    'start_time': segment['start'],
                    'duration': segment['duration']
                })
        
        return effects
    
    def render_video(self, timeline: Dict, video_path: str, audio_path: Optional[str] = None, output_path: str = "output_video.mp4") -> str:
        """Render final video using MoviePy"""
        try:
            logger.info(f"Starting video rendering to {output_path}")
            
            # Load source video
            video = VideoFileClip(video_path)
            
            # Extract segments
            video_clips = []
            for segment in timeline['segments']:
                clip = video.subclip(segment['start'], segment['end'])
                video_clips.append(clip)
            
            # Apply transitions and effects
            processed_clips = self._apply_effects(video_clips, timeline['effects'])
            
            # Concatenate clips
            if processed_clips:
                final_video = concatenate_videoclips(processed_clips, method="compose")
            else:
                final_video = video
            
            # Handle audio
            if audio_path and os.path.exists(audio_path):
                audio = AudioFileClip(audio_path)
                final_video = final_video.set_audio(audio.subclip(0, final_video.duration))
            
            # Set output parameters
            final_video = final_video.resize(self.output_resolution)
            
            # Write final video
            final_video.write_videofile(
                output_path,
                fps=self.output_fps,
                codec=self.video_codec,
                audio_codec=self.audio_codec,
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                verbose=False,
                logger=None  # Suppress MoviePy logging
            )
            
            # Clean up
            video.close()
            if 'audio' in locals():
                audio.close()
            final_video.close()
            for clip in processed_clips:
                clip.close()
            
            logger.info(f"Video rendering completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error rendering video: {e}")
            # Return original video path as fallback
            return video_path
    
    def _apply_effects(self, clips: List, effects: List[Dict]) -> List:
        """Apply effects to video clips"""
        processed_clips = clips.copy()
        
        try:
            for effect in effects:
                segment_idx = effect.get('segment_idx', 0)
                
                if segment_idx >= len(processed_clips):
                    continue
                
                clip = processed_clips[segment_idx]
                
                if effect['type'] == 'fade_in':
                    processed_clips[segment_idx] = fadein(clip, effect['duration'])
                    
                elif effect['type'] == 'fade_out':
                    processed_clips[segment_idx] = fadeout(clip, effect['duration'])
                    
                elif effect['type'] == 'color_enhance':
                    # Simple brightness adjustment (MoviePy doesn't have color correction)
                    processed_clips[segment_idx] = clip.fx(lambda clip: clip.multiply_color(effect.get('intensity', 1.1)))
                    
                elif effect['type'] == 'resize':
                    new_size = effect.get('size', (1920, 1080))
                    processed_clips[segment_idx] = resize(clip, newsize=new_size)
                    
        except Exception as e:
            logger.warning(f"Error applying effects: {e}")
            # Return original clips if effects fail
        
        return processed_clips
    
    def render_with_ffmpeg(self, timeline: Dict, video_path: str, output_path: str = "output_video.mp4") -> str:
        """Alternative rendering using FFmpeg directly for better performance"""
        try:
            logger.info(f"Starting FFmpeg rendering to {output_path}")
            
            # Generate FFmpeg filter complex for cuts and transitions
            filter_complex = self._generate_ffmpeg_filters(timeline, video_path)
            
            # Build FFmpeg command
            input_stream = ffmpeg.input(video_path)
            
            output = (
                input_stream
                .filter_complex(filter_complex)
                .output(
                    output_path,
                    vcodec=self.video_codec,
                    acodec=self.audio_codec,
                    r=self.output_fps,
                    s=f"{self.output_resolution[0]}x{self.output_resolution[1]}"
                )
                .overwrite_output()
            )
            
            # Run FFmpeg
            ffmpeg.run(output, quiet=True)
            
            logger.info(f"FFmpeg rendering completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error with FFmpeg rendering: {e}")
            # Fallback to MoviePy
            return self.render_video(timeline, video_path, output_path=output_path)
    
    def _generate_ffmpeg_filters(self, timeline: Dict, video_path: str) -> str:
        """Generate FFmpeg filter complex string for timeline"""
        filters = []
        
        try:
            segments = timeline.get('segments', [])
            
            if not segments:
                return ""
            
            # Create trim filters for each segment
            for i, segment in enumerate(segments):
                start = segment['start']
                duration = segment['duration']
                
                trim_filter = f"[0:v]trim=start={start}:duration={duration},setpts=PTS-STARTPTS[v{i}]"
                filters.append(trim_filter)
                
                audio_filter = f"[0:a]atrim=start={start}:duration={duration},asetpts=PTS-STARTPTS[a{i}]"
                filters.append(audio_filter)
            
            # Concatenate all segments
            if len(segments) > 1:
                video_inputs = "".join([f"[v{i}]" for i in range(len(segments))])
                audio_inputs = "".join([f"[a{i}]" for i in range(len(segments))])
                
                concat_filter = f"{video_inputs}concat=n={len(segments)}:v=1:a=0[outv]; {audio_inputs}concat=n={len(segments)}:v=0:a=1[outa]"
                filters.append(concat_filter)
            else:
                filters.append("[v0][a0]")
            
            return "; ".join(filters)
            
        except Exception as e:
            logger.warning(f"Error generating FFmpeg filters: {e}")
            return ""
    
    def create_preview(self, timeline: Dict, video_path: str, preview_duration: float = 30.0, output_path: str = "preview.mp4") -> str:
        """Create a quick preview of the edited video"""
        try:
            logger.info(f"Creating preview: {output_path}")
            
            # Select key segments for preview
            segments = timeline.get('segments', [])
            total_duration = sum(seg['duration'] for seg in segments)
            
            if total_duration <= preview_duration:
                # Use all segments if total is short enough
                preview_segments = segments
            else:
                # Select representative segments
                preview_segments = self._select_preview_segments(segments, preview_duration)
            
            # Create preview timeline
            preview_timeline = {
                'segments': preview_segments,
                'transitions': [],
                'effects': []
            }
            
            # Render preview with lower quality for speed
            temp_config = self.config.copy()
            temp_config['output_resolution'] = (1280, 720)  # Lower resolution
            temp_config['output_quality'] = 'draft'
            
            temp_generator = TimelineGenerator(temp_config)
            return temp_generator.render_video(preview_timeline, video_path, output_path=output_path)
            
        except Exception as e:
            logger.error(f"Error creating preview: {e}")
            return ""
    
    def _select_preview_segments(self, segments: List[Dict], target_duration: float) -> List[Dict]:
        """Select representative segments for preview"""
        if not segments:
            return []
        
        # Simple strategy: take first few seconds of each segment
        preview_segments = []
        current_duration = 0.0
        segment_preview_duration = target_duration / len(segments)
        
        for segment in segments:
            if current_duration >= target_duration:
                break
            
            preview_length = min(segment_preview_duration, segment['duration'], target_duration - current_duration)
            
            preview_segments.append({
                'start': segment['start'],
                'end': segment['start'] + preview_length,
                'duration': preview_length
            })
            
            current_duration += preview_length
        
        return preview_segments
