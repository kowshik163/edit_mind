"""
Timeline Generator - Renders final videos from complex, multi-track editing timelines
using FFmpeg for enhanced capabilities and performance.
"""

import ffmpeg
import logging
import tempfile
import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from omegaconf import DictConfig
from dataclasses import dataclass, field
import uuid # For unique stream names

# MoviePy imports are now primarily for fallback or specific effects
# not easily done in FFmpeg, or for initial media loading if needed.
# Keep them minimal if focusing on FFmpeg.
try:
    import moviepy.editor as mp
    from moviepy.video.fx.all import resize, fadein, fadeout
    from moviepy.audio.fx.all import audio_fadein, audio_fadeout
    HAS_MOVIEPY = True
except ImportError:
    HAS_MOVIEPY = False
    mp = None # Set mp to None if import fails


logger = logging.getLogger(__name__)

# --- Enhanced Timeline Data Structures ---

@dataclass
class MediaSource:
    """Represents an input media file."""
    id: str # Unique identifier for this source file in the timeline context
    path: str # Filesystem path
    media_type: str # 'video', 'audio', 'image'
    # Optional: ffprobe info could be stored here after initial scan
    duration: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    audio_sample_rate: Optional[int] = None

@dataclass
class ClipEffect:
    """Represents an effect applied to a clip."""
    type: str # e.g., 'color_grade', 'blur', 'speed', 'mask', 'text_overlay'
    start_time: float # Relative to the clip's start on the track
    end_time: float # Relative to the clip's start on the track
    parameters: Dict[str, Any] = field(default_factory=dict)
    # Optional: keyframes for animating parameters
    keyframes: Optional[Dict[str, List[Tuple[float, Any]]]] = None # {param_name: [(time, value), ...]}

@dataclass
class Clip:
    """Represents a segment of media placed on a track."""
    source_id: str # ID of the MediaSource
    track_start_time: float # When this clip starts on the main timeline track
    source_start_time: float = 0.0 # Start time within the source media
    source_end_time: float # End time within the source media
    clip_id: str = field(default_factory=lambda: f"clip_{uuid.uuid4().hex[:8]}")
    effects: List[ClipEffect] = field(default_factory=list)

    @property
    def duration(self):
        return self.source_end_time - self.source_start_time

    @property
    def track_end_time(self):
        # Assumes speed effect is handled by adjusting source_end_time or via filter
        return self.track_start_time + self.duration

@dataclass
class Transition:
    """Represents a transition between two clips on the same track."""
    type: str # e.g., 'cut', 'fade', 'dissolve', 'wipe_left'
    duration: float
    from_clip_id: str
    to_clip_id: str

@dataclass
class Track:
    """Represents a single video or audio track."""
    id: str = field(default_factory=lambda: f"track_{uuid.uuid4().hex[:8]}")
    type: str # 'video' or 'audio'
    clips: List[Clip] = field(default_factory=list)
    transitions: List[Transition] = field(default_factory=list)
    volume: float = 1.0 # For audio tracks

@dataclass
class AdvancedTimeline:
    """Represents the complete multi-track editing timeline."""
    sources: List[MediaSource] = field(default_factory=list)
    tracks: List[Track] = field(default_factory=list)
    total_duration: float = 0.0
    output_width: int = 1920
    output_height: int = 1080
    output_fps: float = 30.0

# --- TimelineGenerator Class ---

class TimelineGenerator:
    """Advanced timeline generation and video rendering using FFmpeg primarily."""

    def __init__(self, config: DictConfig):
        self.config = config

        # Video rendering settings from config
        self.output_fps = config.get('output_fps', 30.0)
        # Ensure resolution is a tuple/list of two integers
        raw_resolution = config.get('output_resolution', [1920, 1080])
        if isinstance(raw_resolution, (list, tuple)) and len(raw_resolution) == 2:
            self.output_resolution = tuple(map(int, raw_resolution))
        else:
            logger.warning(f"Invalid output_resolution '{raw_resolution}', using 1920x1080.")
            self.output_resolution = (1920, 1080)

        self.video_codec = config.get('video_codec', 'libx264')
        self.audio_codec = config.get('audio_codec', 'aac')
        self.output_quality = config.get('output_quality', 'medium') # maps to CRF or bitrate
        self.ffmpeg_loglevel = config.get('ffmpeg_loglevel', 'warning') # FFmpeg verbosity

        # Transition settings
        self.default_transition_duration = config.get('default_transition_duration', 0.5)

        # Cache for media info
        self._media_info_cache = {}

        logger.info("Advanced TimelineGenerator initialized (FFmpeg focused)")
        if not self._check_ffmpeg():
            logger.error("FFmpeg command not found or not executable. Rendering will likely fail.")


    def _check_ffmpeg(self) -> bool:
        """Check if ffmpeg command is available."""
        try:
            subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False

    def _get_media_info(self, file_path: str) -> Dict[str, Any]:
        """Get media duration, resolution, fps using ffprobe."""
        if file_path in self._media_info_cache:
            return self._media_info_cache[file_path]
        if not Path(file_path).exists():
             logger.error(f"Media file not found for info: {file_path}")
             return {}
        try:
            logger.debug(f"Probing media file: {file_path}")
            probe = ffmpeg.probe(file_path)
            video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
            audio_stream = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)

            info = {'format_duration': float(probe['format'].get('duration', 0))}

            if video_stream:
                info['width'] = int(video_stream.get('width', 0))
                info['height'] = int(video_stream.get('height', 0))
                # Calculate FPS carefully
                avg_fps_str = video_stream.get('avg_frame_rate', '0/1')
                if '/' in avg_fps_str:
                    num, den = map(int, avg_fps_str.split('/'))
                    info['fps'] = float(num / den) if den != 0 else 0.0
                else:
                    info['fps'] = float(avg_fps_str)
                info['video_duration'] = float(video_stream.get('duration', info['format_duration']))

            if audio_stream:
                info['audio_sample_rate'] = int(audio_stream.get('sample_rate', 0))
                info['audio_channels'] = int(audio_stream.get('channels', 0))
                info['audio_duration'] = float(audio_stream.get('duration', info['format_duration']))

            # Use format duration as the primary duration unless stream duration is significantly different
            info['duration'] = info['format_duration']
            if 'video_duration' in info and abs(info['video_duration'] - info['duration']) > 0.1:
                 info['duration'] = info['video_duration'] # Prefer video stream duration if available and different
            elif 'audio_duration' in info and abs(info['audio_duration'] - info['duration']) > 0.1:
                 info['duration'] = info['audio_duration']


            self._media_info_cache[file_path] = info
            logger.debug(f"Media info for {file_path}: {info}")
            return info
        except Exception as e:
            logger.error(f"Error probing media file {file_path}: {e}")
            return {}

    def decode_timeline(self,
                        ai_planner_output: Dict[str, Any],
                        media_sources: List[Dict[str, str]]) -> AdvancedTimeline:
        """
        Decodes the AI planner's output into the AdvancedTimeline structure.

        Args:
            ai_planner_output: Dictionary output from EditingPlannerModule.
                               (Currently assumes a simpler, flat structure).
            media_sources: List of dictionaries describing input media files
                           e.g., [{'id': 'src1', 'path': 'video.mp4', 'type': 'video'}, ...]

        Returns:
            An AdvancedTimeline object.
        """
        logger.info("Decoding AI planner output into advanced timeline structure...")
        timeline = AdvancedTimeline(
            output_width=self.output_resolution[0],
            output_height=self.output_resolution[1],
            output_fps=self.output_fps
        )

        # 1. Populate Sources and get info
        source_map = {}
        for src_dict in media_sources:
            media_info = self._get_media_info(src_dict['path'])
            source = MediaSource(
                id=src_dict.get('id', Path(src_dict['path']).stem),
                path=src_dict['path'],
                media_type=src_dict.get('type', 'video'),
                duration=media_info.get('duration'),
                width=media_info.get('width'),
                height=media_info.get('height'),
                fps=media_info.get('fps'),
                audio_sample_rate=media_info.get('audio_sample_rate')
            )
            timeline.sources.append(source)
            source_map[source.id] = source

        # --- Interpretation Logic (Needs Adaptation Based on Actual AI Output) ---
        # This part heavily depends on what the AI Planner *actually* outputs.
        # Assuming a simple, flat output for now, similar to the original code.
        # We need to map this flat structure onto tracks.

        # Example: Assume AI provides cuts for the *first* video source
        if not timeline.sources:
             logger.warning("No media sources provided for timeline decoding.")
             return timeline # Return empty timeline

        primary_video_source = next((s for s in timeline.sources if s.media_type == 'video'), timeline.sources[0])
        primary_audio_source = next((s for s in timeline.sources if s.media_type == 'audio'), primary_video_source) # Use video audio if no separate audio

        # AI output interpretation (example based on original code's assumptions)
        # Using .get() extensively to handle potentially missing keys
        ai_cuts_tensor = ai_planner_output.get('cut_points') # Assumes tensor of probabilities or indices
        ai_transitions_tensor = ai_planner_output.get('transitions') # Assumes tensor of logits/IDs
        ai_effects_tensor = ai_planner_output.get('effects') # Assumes tensor of logits/IDs

        # Convert AI cut predictions to timestamps
        cut_times = []
        video_duration = primary_video_source.duration or 10.0 # Default duration if info missing
        if ai_cuts_tensor is not None and isinstance(ai_cuts_tensor, torch.Tensor):
             # This logic needs refinement based on actual tensor meaning (probs, indices?)
             # Assuming probabilities per frame/segment
             if ai_cuts_tensor.dim() > 0:
                 cut_probs = ai_cuts_tensor.squeeze().cpu().numpy()
                 # Example: find peaks above a threshold
                 threshold = 0.5
                 peaks = np.where(cut_probs > threshold)[0]
                 # Convert peaks (indices) to time - depends on how indices relate to time
                 # Assuming indices correspond roughly to segments/frames over the duration
                 num_indices = len(cut_probs)
                 cut_times = [p * video_duration / num_indices for p in peaks]
                 logger.info(f"Decoded {len(cut_times)} AI cut points.")


        # Create a basic video track (V1)
        video_track = Track(id="V1", type="video")
        last_cut = 0.0
        clip_counter = 0
        all_cut_times = sorted(list(set([0.0] + cut_times + [video_duration])))

        for i in range(len(all_cut_times) - 1):
            start = all_cut_times[i]
            end = all_cut_times[i+1]
            if end > start and start < video_duration:
                clip_duration = min(end, video_duration) - start
                if clip_duration > 0.05: # Minimum clip duration
                    clip = Clip(
                        source_id=primary_video_source.id,
                        track_start_time=start, # Place sequentially for now
                        source_start_time=start,
                        source_end_time=start + clip_duration,
                        clip_id=f"v1_clip_{clip_counter}"
                    )
                    # TODO: Add effects predicted by AI for this time range
                    # This requires mapping ai_effects_tensor indices to time ranges
                    # Example: Add a simple color grade effect
                    clip.effects.append(ClipEffect(
                         type='color_grade', # Generic type, map from AI output
                         start_time=0.0,
                         end_time=clip.duration,
                         parameters={'preset': 'cinematic'} # Parameter from AI
                    ))
                    video_track.clips.append(clip)
                    clip_counter += 1
                    last_cut = end

        # Add transitions between clips
        transition_names = ['cut', 'fade', 'dissolve', 'wipe_left'] # Example mapping
        for i in range(len(video_track.clips) - 1):
             transition_type = 'fade' # Default
             if ai_transitions_tensor is not None and i < ai_transitions_tensor.shape[1]:
                  # Assuming tensor holds transition IDs per segment/cut
                  try:
                    transition_id = torch.argmax(ai_transitions_tensor[0, i]).item() # Example decoding
                    transition_type = transition_names[transition_id % len(transition_names)]
                  except:
                       pass # Keep default

             video_track.transitions.append(Transition(
                 type=transition_type,
                 duration=self.default_transition_duration,
                 from_clip_id=video_track.clips[i].clip_id,
                 to_clip_id=video_track.clips[i+1].clip_id
             ))

        timeline.tracks.append(video_track)

        # Create a basic audio track (A1) using primary audio source
        audio_track = Track(id="A1", type="audio")
        audio_duration = primary_audio_source.duration or video_duration
        audio_track.clips.append(Clip(
            source_id=primary_audio_source.id,
            track_start_time=0.0,
            source_start_time=0.0,
            source_end_time=audio_duration
        ))
        timeline.tracks.append(audio_track)

        # Example: Add background music track (A2) if provided
        bgm_source = next((s for s in timeline.sources if s.media_type == 'audio' and s.id != primary_audio_source.id), None)
        if bgm_source:
             bgm_track = Track(id="A2", type="audio", volume=0.3) # Lower volume
             bgm_track.clips.append(Clip(
                 source_id=bgm_source.id,
                 track_start_time=0.0,
                 source_start_time=0.0,
                 source_end_time=bgm_source.duration or video_duration
             ))
             timeline.tracks.append(bgm_track)


        # Calculate total duration based on the longest track
        timeline.total_duration = max(
             (track.clips[-1].track_end_time if track.clips else 0) for track in timeline.tracks
        ) if timeline.tracks else 0


        logger.info(f"Decoded timeline: {len(timeline.tracks)} tracks, duration {timeline.total_duration:.2f}s")
        # logger.debug(f"Timeline structure: {timeline}") # Can be very verbose
        return timeline


    def render_video(self,
                     timeline: AdvancedTimeline,
                     output_path: str,
                     custom_effects: Dict[str, callable] = None) -> str:
        """
        Render the final video using FFmpeg based on the AdvancedTimeline.

        Args:
            timeline: The AdvancedTimeline object describing the edit.
            output_path: The desired path for the output video file.
            custom_effects: Dictionary mapping effect type names to callable
                            functions (like those from self-coding engine).

        Returns:
            The path to the rendered video file.
        """
        output_path = str(Path(output_path).resolve()) # Ensure absolute path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True) # Ensure dir exists

        if not timeline.tracks:
            raise ValueError("Timeline has no tracks to render.")
        if not self._check_ffmpeg():
            raise RuntimeError("FFmpeg is not available or executable.")

        logger.info(f"Starting FFmpeg rendering to {output_path}")
        logger.info(f"Timeline duration: {timeline.total_duration:.2f}s, Resolution: {timeline.output_width}x{timeline.output_height}, FPS: {timeline.output_fps}")

        try:
            # 1. Prepare FFmpeg inputs
            input_streams = {} # Map source_id to ffmpeg input object
            input_args = []
            source_map = {src.id: src for src in timeline.sources}
            for i, source in enumerate(timeline.sources):
                if not Path(source.path).exists():
                     logger.warning(f"Source file not found, skipping: {source.path}")
                     continue
                input_streams[source.id] = ffmpeg.input(source.path)
                input_args.append(source.path) # For command line logging

            # 2. Build Filter Graph
            video_streams = [] # List of final stream names for each video track
            audio_streams = [] # List of final stream names for each audio track
            filter_complex_parts = [] # List of filter strings

            stream_counter = 0
            clip_stream_map = {} # Map clip_id to its processed stream name

            for track in timeline.tracks:
                if not track.clips: continue

                processed_clip_streams = [] # Streams after effects/trim for this track

                for clip_index, clip in enumerate(track.clips):
                    if clip.source_id not in input_streams:
                         logger.warning(f"Source ID '{clip.source_id}' for clip '{clip.clip_id}' not found in inputs, skipping clip.")
                         continue

                    source_stream = input_streams[clip.source_id]
                    stream_name = f"s{stream_counter}"
                    stream_counter += 1

                    # Apply trim first
                    trim_filter = f"[{source_stream.node.label}]"
                    if track.type == 'video':
                        trim_filter += f"trim=start={clip.source_start_time:.4f}:end={clip.source_end_time:.4f},setpts=PTS-STARTPTS"
                    elif track.type == 'audio':
                         # Use atrim for audio
                        trim_filter += f"atrim=start={clip.source_start_time:.4f}:end={clip.source_end_time:.4f},asetpts=PTS-STARTPTS"

                    # Scale video clips to output resolution if necessary
                    if track.type == 'video':
                         src = source_map[clip.source_id]
                         if src.width != timeline.output_width or src.height != timeline.output_height:
                              trim_filter += f",scale={timeline.output_width}:{timeline.output_height}:force_original_aspect_ratio=decrease:eval=frame,pad={timeline.output_width}:{timeline.output_height}:(ow-iw)/2:(oh-ih)/2:black" # Scale and pad

                    trim_filter += f"[{stream_name}]"
                    filter_complex_parts.append(trim_filter)

                    # Apply effects
                    current_stream = stream_name
                    if clip.effects:
                         effect_stream, effect_filters = self._apply_ffmpeg_effects(
                             current_stream, clip, timeline.output_fps, custom_effects
                         )
                         if effect_filters:
                              filter_complex_parts.extend(effect_filters)
                              current_stream = effect_stream
                         else:
                              logger.warning(f"Could not apply effects for clip {clip.clip_id}")


                    clip_stream_map[clip.clip_id] = current_stream
                    processed_clip_streams.append(current_stream)


                # Handle Transitions and Concatenation/Overlay for the track
                if track.type == 'video':
                    final_track_stream, track_filters = self._process_video_track(
                        track, clip_stream_map, timeline.total_duration, timeline.output_fps
                    )
                    if track_filters:
                         filter_complex_parts.extend(track_filters)
                         video_streams.append(final_track_stream)
                elif track.type == 'audio':
                     final_track_stream, track_filters = self._process_audio_track(
                        track, clip_stream_map, timeline.total_duration
                    )
                     if track_filters:
                          filter_complex_parts.extend(track_filters)
                          audio_streams.append(final_track_stream)


            # 3. Combine Video Tracks (Overlay)
            if len(video_streams) > 1:
                base_video = video_streams[0] # V1 is base
                overlay_stream = base_video
                for i, overlay_video in enumerate(video_streams[1:]):
                    new_overlay_stream = f"overlayed_v{i+1}"
                    # Simple overlay at 0,0 - needs position parameters for PiP etc.
                    filter_complex_parts.append(f"[{overlay_stream}][{overlay_video}]overlay=0:0:eof_action=pass[{new_overlay_stream}]")
                    overlay_stream = new_overlay_stream
                final_video_stream = overlay_stream
            elif len(video_streams) == 1:
                final_video_stream = video_streams[0]
            else:
                 # Handle no video case - generate black screen
                 logger.warning("No video tracks found, generating black video.")
                 filter_complex_parts.append(f"color=c=black:s={timeline.output_width}x{timeline.output_height}:d={timeline.total_duration:.4f}:r={timeline.output_fps}[black_v]")
                 final_video_stream = "black_v"
                 # Need a dummy audio stream if no audio exists either
                 if not audio_streams:
                      filter_complex_parts.append(f"anullsrc=d={timeline.total_duration:.4f}[null_a]")
                      audio_streams.append("null_a")



            # 4. Combine Audio Tracks (Mix)
            if len(audio_streams) > 1:
                inputs_str = "".join([f"[{s}]" for s in audio_streams])
                filter_complex_parts.append(f"{inputs_str}amix=inputs={len(audio_streams)}:duration=first:dropout_transition=2[final_audio]")
                final_audio_stream = "final_audio"
            elif len(audio_streams) == 1:
                final_audio_stream = audio_streams[0]
                # Ensure the single audio stream is labeled for output mapping
                filter_complex_parts.append(f"[{final_audio_stream}]anull[final_audio]")
                final_audio_stream = "final_audio"
            else:
                 # Should have been handled by the 'no video' case generating null_a
                 raise ValueError("No audio streams available for final output.")


            # 5. Build and Execute FFmpeg Command
            filter_graph = ";".join(filter_complex_parts)
            logger.debug(f"FFmpeg Filter Graph:\n{filter_graph}")

            # Define output arguments based on config
            output_args = {
                 'vcodec': self.video_codec,
                 'acodec': self.audio_codec,
                 'r': timeline.output_fps,
                 'pix_fmt': 'yuv420p', # Common pixel format
                 'loglevel': self.ffmpeg_loglevel,
            }
            if self.video_codec == 'libx264':
                 quality_map = {'low': 28, 'medium': 23, 'high': 18, 'ultra': 14}
                 output_args['crf'] = quality_map.get(self.output_quality, 23)
            # Add bitrate options for other codecs if needed

            # Map final streams to output
            stream_inputs = [s for s in input_streams.values()]
            process = (
                ffmpeg
                .filter_complex(stream_inputs, filter_graph, **{f'{final_video_stream}': None, f'{final_audio_stream}': None})
                .output(output_path, **output_args)
                .overwrite_output()
            )

            cmd = process.compile()
            logger.info("Executing FFmpeg command:")
            # Log command carefully, might contain sensitive paths
            # logger.info(" ".join(cmd)) # Use cautiously

            stdout, stderr = process.run(capture_stdout=True, capture_stderr=True)

            if self.ffmpeg_loglevel in ['debug', 'verbose', 'info']:
                logger.info(f"FFmpeg stdout:\n{stdout.decode(errors='ignore')}")
            if stderr:
                 # Check stderr for errors even if process completed
                 stderr_str = stderr.decode(errors='ignore')
                 if "error" in stderr_str.lower():
                      logger.error(f"FFmpeg stderr indicates errors:\n{stderr_str}")
                      # raise RuntimeError(f"FFmpeg rendering failed. Check logs.") # Optionally raise error
                 elif self.ffmpeg_loglevel in ['debug', 'verbose', 'info']:
                      logger.info(f"FFmpeg stderr:\n{stderr_str}")


            logger.info(f"✅ FFmpeg rendering completed: {output_path}")
            return output_path

        except ffmpeg.Error as e:
            logger.error(f"❌ FFmpeg Error during rendering:")
            if hasattr(e, 'stderr') and e.stderr:
                 logger.error(f"FFmpeg stderr:\n{e.stderr.decode(errors='ignore')}")
            else:
                 logger.error(f"{e}")
            raise RuntimeError(f"FFmpeg rendering failed: {e}")
        except Exception as e:
            logger.error(f"❌ Unexpected error during rendering: {e}", exc_info=True)
            # Optionally fallback to MoviePy here if HAS_MOVIEPY
            # logger.warning("FFmpeg failed, attempting fallback with MoviePy...")
            # return self._render_with_moviepy(timeline, output_path, custom_effects)
            raise RuntimeError(f"Video rendering failed: {e}")

    def _process_video_track(self, track: Track, clip_stream_map: Dict[str, str], total_duration: float, fps: float) -> Tuple[Optional[str], List[str]]:
        """Generates FFmpeg filters for transitions and concatenation/overlay for a video track."""
        filters = []
        if not track.clips: return None, filters

        clip_streams = [clip_stream_map[clip.clip_id] for clip in track.clips if clip.clip_id in clip_stream_map]
        if not clip_streams: return None, filters

        # Handle Transitions using xfade
        if len(clip_streams) > 1 and track.transitions:
            xfade_streams = []
            current_stream = clip_streams[0]
            last_clip_end = track.clips[0].duration

            for i, transition in enumerate(track.transitions):
                 if i + 1 >= len(clip_streams): break # Should not happen if transitions are correct
                 clip1_id = transition.from_clip_id
                 clip2_id = transition.to_clip_id
                 clip1_stream = clip_stream_map.get(clip1_id)
                 clip2_stream = clip_stream_map.get(clip2_id)
                 clip1 = next((c for c in track.clips if c.clip_id == clip1_id), None)

                 if not clip1_stream or not clip2_stream or not clip1: continue

                 xfade_type = self._map_transition_to_xfade(transition.type)
                 xfade_duration = transition.duration
                 xfade_offset = last_clip_end - xfade_duration # Start fade before clip1 ends
                 fade_stream_name = f"xfade{i}_{track.id}"

                 filters.append(
                     f"[{current_stream}][{clip2_stream}]"
                     f"xfade=transition={xfade_type}:duration={xfade_duration:.4f}:offset={xfade_offset:.4f}"
                     f"[{fade_stream_name}]"
                 )
                 current_stream = fade_stream_name
                 # Update the effective end time for the next offset calculation
                 clip2 = next((c for c in track.clips if c.clip_id == clip2_id), None)
                 if clip2:
                      last_clip_end += clip2.duration - xfade_duration
                 else: # Estimate if clip2 info is missing
                      last_clip_end += self.default_transition_duration - xfade_duration # Use default duration


            final_track_stream = current_stream
        else:
            # Simple concatenation if no transitions or single clip
            inputs_str = "".join([f"[{s}]" for s in clip_streams])
            final_track_stream = f"concat_v_{track.id}"
            filters.append(f"{inputs_str}concat=n={len(clip_streams)}:v=1:a=0[{final_track_stream}]")

        # Ensure the final track stream covers the full timeline duration by padding
        padded_stream = f"padded_{track.id}"
        filters.append(
            f"[{final_track_stream}]tpad=stop_mode=clone:stop_duration={max(0, total_duration - last_clip_end):.4f}[{padded_stream}]"
        )


        return padded_stream, filters

    def _process_audio_track(self, track: Track, clip_stream_map: Dict[str, str], total_duration: float) -> Tuple[Optional[str], List[str]]:
         """Generates FFmpeg filters for concatenation and volume for an audio track."""
         filters = []
         if not track.clips: return None, filters

         clip_streams = [clip_stream_map[clip.clip_id] for clip in track.clips if clip.clip_id in clip_stream_map]
         if not clip_streams: return None, filters

         # Concatenate audio clips (transitions usually handled by video or simple crossfade in amix)
         inputs_str = "".join([f"[{s}]" for s in clip_streams])
         concat_stream = f"concat_a_{track.id}"
         filters.append(f"{inputs_str}concat=n={len(clip_streams)}:v=0:a=1[{concat_stream}]")

         # Apply volume adjustment
         volume_stream = f"vol_{track.id}"
         filters.append(f"[{concat_stream}]volume=volume={track.volume:.2f}[{volume_stream}]")

         # Pad audio to total duration
         # Calculate current duration after concat
         current_duration = sum(c.duration for c in track.clips)
         padded_stream = f"padded_a_{track.id}"
         filters.append(
             f"[{volume_stream}]apad=pad_dur={max(0, total_duration - current_duration):.4f}[{padded_stream}]"
         )


         return padded_stream, filters


    def _apply_ffmpeg_effects(self,
                             input_stream_name: str,
                             clip: Clip,
                             fps: float,
                             custom_effects: Optional[Dict[str, callable]]) -> Tuple[str, List[str]]:
        """Generates FFmpeg filter strings for effects applied to a single clip stream."""
        filters = []
        current_stream = input_stream_name
        effect_counter = 0

        for effect in clip.effects:
            effect_stream_name = f"eff_{clip.clip_id}_{effect_counter}"
            effect_filter = None
            start_eff = effect.start_time
            end_eff = effect.end_time
            enable_expr = f'enable=between(t,{start_eff:.4f},{end_eff:.4f})'

            # Map effect types to FFmpeg filters
            if effect.type == 'color_grade':
                preset = effect.parameters.get('preset', 'cinematic')
                intensity = effect.parameters.get('intensity', 0.5)
                if preset == 'cinematic':
                     # Example: slightly increase contrast, adjust saturation, maybe add slight tint
                     contrast = 1.0 + 0.2 * intensity
                     saturation = 1.0 + 0.1 * intensity
                     # Simple S-curve approximation with curves filter if needed, or eq
                     effect_filter = f"[{current_stream}]eq=contrast={contrast:.2f}:saturation={saturation:.2f}:{enable_expr}[{effect_stream_name}]"
                elif preset == 'vintage':
                     effect_filter = f"[{current_stream}]eq=saturation=0.8:contrast=1.1, frei0r=sepia:intensity={0.5*intensity}:{enable_expr}[{effect_stream_name}]"
                # Add more presets...
            elif effect.type == 'blur':
                 strength = effect.parameters.get('strength', 5) * intensity # Scale strength by intensity
                 # Use gblur with varying sigma based on strength
                 sigma = max(0.5, strength / 5.0)
                 effect_filter = f"[{current_stream}]gblur=sigma={sigma:.2f}:{enable_expr}[{effect_stream_name}]"
            elif effect.type == 'speed':
                 factor = effect.parameters.get('factor', 1.0)
                 if abs(factor - 1.0) > 0.01:
                      video_pts = f"setpts={1/factor:.4f}*PTS"
                      audio_pts = f"atempo={factor:.4f}" # Needs separate audio stream processing
                      # Note: Speed changes complicate concatenation and transitions significantly.
                      # This basic filter doesn't handle audio pitch or smooth transitions well.
                      # It's better applied BEFORE concatenation if possible, or requires complex filtergraph.
                      logger.warning("Speed effect applied simply; may affect sync and transitions.")
                      effect_filter = f"[{current_stream}]{video_pts}[{effect_stream_name}]"
                      # Audio needs separate handling
            elif effect.type == 'text_overlay':
                 text = effect.parameters.get('text', 'Sample Text')
                 pos_x = effect.parameters.get('x', '(w-text_w)/2') # Default center
                 pos_y = effect.parameters.get('y', '(h-text_h)/2')
                 font_size = effect.parameters.get('size', 24)
                 font_color = effect.parameters.get('color', 'white')
                 # Basic text overlay
                 effect_filter = f"[{current_stream}]drawtext=text='{text}':x={pos_x}:y={pos_y}:fontsize={font_size}:fontcolor={font_color}:{enable_expr}[{effect_stream_name}]"
            elif effect.type == 'mask':
                 # Basic geometric mask (e.g., rectangle) - very limited in FFmpeg filters alone
                 shape = effect.parameters.get('shape', 'rect')
                 x, y, w, h = effect.parameters.get('rect', (0, 0, 100, 100))
                 if shape == 'rect':
                      # Requires overlaying with a generated mask or complex drawbox/crop
                      logger.warning("Simple rectangular mask applied via crop (removes outside). Complex masking needs different approach.")
                      effect_filter = f"[{current_stream}]crop={w}:{h}:{x}:{y}:{enable_expr}[{effect_stream_name}]" # This crops, doesn't mask overlay
                 else:
                     logger.warning(f"Unsupported mask shape '{shape}' in FFmpeg filter generation.")

            # --- Placeholder for Custom Effects ---
            elif custom_effects and effect.type in custom_effects:
                 # Custom effects require pre-processing frames outside FFmpeg usually
                 logger.warning(f"Custom effect '{effect.type}' requested. FFmpeg generator cannot directly apply arbitrary Python functions. Pre-processing needed.")
                 # No filter added, relies on external processing or self-coding integration before rendering

            # --- Add more built-in effect mappings ---

            if effect_filter:
                filters.append(effect_filter)
                current_stream = effect_stream_name
                effect_counter += 1
            else:
                 logger.debug(f"No FFmpeg filter generated for effect: {effect.type}")


        return current_stream, filters


    def _map_transition_to_xfade(self, transition_type: str) -> str:
        """Maps descriptive transition names to FFmpeg xfade filter types."""
        mapping = {
            'cut': 'custom', # No transition, handled by concat/overlay timing
            'fade': 'fade',
            'dissolve': 'dissolve', # Often same as fade in ffmpeg xfade
            'wipe_left': 'wipeleft',
            'wipe_right': 'wiperight',
            'wipe_up': 'wipeup',
            'wipe_down': 'wipedown',
            'slide_left': 'slideleft',
            'slide_right': 'slideright',
            'slide_up': 'slideup',
            'slide_down': 'slidedown',
            'circleopen': 'circleopen',
            'circleclose': 'circleclose',
            'rectcrop': 'rectcrop',
            'diagtl': 'diagtl', # Diagonal top-left
            'diagtr': 'diagtr', # Diagonal top-right
            'diagbl': 'diagbl', # Diagonal bottom-left
            'diagbr': 'diagbr', # Diagonal bottom-right
            'hlslice': 'hlslice', # Horizontal slice
            'hrslice': 'hrslice',
            'vuslice': 'vuslice', # Vertical slice
            'vdslice': 'vdslice',
            # Add more mappings as needed
        }
        # Use 'fade' as default if type not found
        return mapping.get(transition_type.lower().replace(" ", "_"), 'fade')

    # --- Fallback Rendering (Optional) ---
    def _render_with_moviepy(self, timeline: AdvancedTimeline, output_path: str, custom_effects: Optional[Dict[str, callable]]) -> str:
        """Fallback rendering using MoviePy (less capable but simpler)."""
        if not HAS_MOVIEPY:
            raise RuntimeError("MoviePy is not installed. Cannot use fallback renderer.")

        logger.warning("Using MoviePy fallback renderer. Advanced features may be limited.")
        try:
             video_clips_on_tracks = {} # track_id: [moviepy_clips]
             audio_clips_on_tracks = {} # track_id: [moviepy_clips]
             source_map = {src.id: src.path for src in timeline.sources}

             max_end_time = 0

             # Load clips and place them on conceptual tracks
             for track in timeline.tracks:
                 if track.type == 'video': track_dict = video_clips_on_tracks
                 elif track.type == 'audio': track_dict = audio_clips_on_tracks
                 else: continue

                 track_dict[track.id] = []
                 for clip_data in track.clips:
                     if clip_data.source_id not in source_map: continue
                     source_path = source_map[clip_data.source_id]

                     if track.type == 'video':
                         try:
                              clip = mp.VideoFileClip(source_path).subclip(
                                  clip_data.source_start_time, clip_data.source_end_time
                              )
                              # Resize clip
                              clip = clip.resize(height=timeline.output_height) # Resize maintaining aspect
                              # TODO: Apply MoviePy effects from clip_data.effects
                              clip = clip.set_start(clip_data.track_start_time).set_duration(clip_data.duration)
                              track_dict[track.id].append(clip)
                              max_end_time = max(max_end_time, clip_data.track_end_time)
                         except Exception as e:
                              logger.warning(f"MoviePy failed to load/process video clip {clip_data.clip_id}: {e}")
                     elif track.type == 'audio':
                           try:
                                clip = mp.AudioFileClip(source_path).subclip(
                                     clip_data.source_start_time, clip_data.source_end_time
                                )
                                clip = clip.volumex(track.volume) # Apply track volume
                                # TODO: Apply audio effects
                                clip = clip.set_start(clip_data.track_start_time).set_duration(clip_data.duration)
                                track_dict[track.id].append(clip)
                                max_end_time = max(max_end_time, clip_data.track_end_time)
                           except Exception as e:
                                logger.warning(f"MoviePy failed to load/process audio clip {clip_data.clip_id}: {e}")

             # Composite video tracks (simple overlay)
             final_video_clips = []
             video_track_ids = sorted(video_clips_on_tracks.keys()) # Render V1 first, then overlay others
             for track_id in video_track_ids:
                  final_video_clips.extend(video_clips_on_tracks[track_id])

             # Composite audio tracks
             final_audio_clips = []
             for track_id in audio_clips_on_tracks:
                  final_audio_clips.extend(audio_clips_on_tracks[track_id])

             # Create final composition
             if not final_video_clips:
                  # Create black video if no video clips
                  logger.warning("No video clips for MoviePy rendering, creating black video.")
                  final_video = mp.ColorClip(size=(timeline.output_width, timeline.output_height),
                                             color=(0,0,0), duration=max_end_time).set_fps(timeline.output_fps)
             else:
                  final_video = mp.CompositeVideoClip(final_video_clips, size=(timeline.output_width, timeline.output_height)).set_duration(max_end_time).set_fps(timeline.output_fps)


             if final_audio_clips:
                  final_audio = mp.CompositeAudioClip(final_audio_clips).set_duration(max_end_time)
                  final_video = final_video.set_audio(final_audio)

             # Write file
             final_video.write_videofile(
                 output_path,
                 fps=timeline.output_fps,
                 codec=self.video_codec,
                 audio_codec=self.audio_codec,
                 temp_audiofile=f'temp-audio_{uuid.uuid4().hex[:8]}.m4a',
                 remove_temp=True,
                 verbose=False,
                 logger=None # Suppress MoviePy logging
             )

             # Cleanup MoviePy objects explicitly
             if 'final_audio' in locals() and final_audio: final_audio.close()
             if final_video: final_video.close()
             for track_clips in video_clips_on_tracks.values():
                  for clip in track_clips: clip.close()
             for track_clips in audio_clips_on_tracks.values():
                  for clip in track_clips: clip.close()


             logger.info(f"✅ MoviePy fallback rendering completed: {output_path}")
             return output_path

        except Exception as e:
            logger.error(f"❌ MoviePy fallback rendering also failed: {e}", exc_info=True)
            raise RuntimeError(f"MoviePy fallback failed: {e}")


    # --- Preview Generation (can keep using MoviePy for speed/simplicity) ---
    def create_preview(self, timeline: AdvancedTimeline, output_path: str = "preview.mp4", preview_duration: float = 15.0) -> str:
        """Create a quick preview using MoviePy."""
        if not HAS_MOVIEPY:
             logger.warning("MoviePy not installed, cannot create preview.")
             return ""
        try:
             logger.info(f"Creating preview (max {preview_duration}s): {output_path}")
             # Render a shorter version using MoviePy
             temp_timeline = timeline # Need a way to shorten the AdvancedTimeline structure easily
             # For simplicity, just render the first `preview_duration` seconds
             # This requires modifying the MoviePy render logic slightly

             # Use MoviePy fallback render but limit duration
             # (Simplified - actual implementation would need careful clipping)
             preview_path = self._render_with_moviepy(temp_timeline, output_path, None) # Pass None for custom effects
             # Clip the rendered preview if it's too long
             clip = mp.VideoFileClip(preview_path)
             if clip.duration > preview_duration:
                 clip = clip.subclip(0, preview_duration)
                 clip.write_videofile(output_path, codec=self.video_codec, audio_codec=self.audio_codec, logger=None)
             clip.close()

             return output_path

        except Exception as e:
             logger.error(f"Error creating preview: {e}")
             return ""
