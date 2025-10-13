"""
Synthetic Data Generator - Uses teacher models to generate training data
Combines templates, stock footage, and teacher models to create synthetic edited videos
"""

import os
import json
import logging
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import cv2
import librosa
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    CLIPModel, CLIPProcessor,
    WhisperProcessor, WhisperForConditionalGeneration
)
from diffusers import StableDiffusionPipeline, DiffusionPipeline
import moviepy.editor as mp
from datetime import datetime
import hashlib
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class EditingInstruction:
    """Structure for editing instructions"""
    action: str  # "cut", "transition", "effect", "beat_sync", "text_overlay"
    timestamp: float
    duration: float
    parameters: Dict[str, Any]
    confidence: float


@dataclass
class SyntheticDataSample:
    """Structure for generated synthetic data"""
    id: str
    raw_footage_path: str
    template_path: Optional[str]
    edited_video_path: str
    editing_instructions: List[EditingInstruction]
    metadata: Dict[str, Any]
    teacher_model_used: str
    generation_timestamp: float


class SyntheticDataGenerator:
    """
    Generates synthetic training data using teacher models
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize teacher models
        self.teacher_models = {}
        self.load_teacher_models()
        
        # Data paths
        self.raw_footage_dir = Path("data/templates/stock_footage")
        self.templates_dir = Path("data/templates")
        self.synthetic_output_dir = Path("data/synthetic")
        self.synthetic_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generation statistics
        self.generation_stats = {
            'total_generated': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'template_based': 0,
            'llm_generated': 0
        }
        
    def load_teacher_models(self):
        """Load all teacher models for synthetic data generation"""
        logger.info("🤖 Loading teacher models for synthetic data generation...")
        
        try:
            # Text generation model for editing instructions
            text_model_name = self.config.get('teachers', {}).get('text_model', 'microsoft/DialoGPT-small')
            logger.info(f"Loading text model: {text_model_name}")
            self.teacher_models['text_tokenizer'] = AutoTokenizer.from_pretrained(text_model_name)
            self.teacher_models['text_model'] = AutoModelForCausalLM.from_pretrained(text_model_name)
            
            # Vision model for content analysis
            vision_model_name = "openai/clip-vit-base-patch32"
            logger.info(f"Loading vision model: {vision_model_name}")
            self.teacher_models['vision_processor'] = CLIPProcessor.from_pretrained(vision_model_name)
            self.teacher_models['vision_model'] = CLIPModel.from_pretrained(vision_model_name)
            
            # Audio analysis model
            audio_model_name = "openai/whisper-base"
            logger.info(f"Loading audio model: {audio_model_name}")
            self.teacher_models['audio_processor'] = WhisperProcessor.from_pretrained(audio_model_name)
            self.teacher_models['audio_model'] = WhisperForConditionalGeneration.from_pretrained(audio_model_name)
            
            # Video generation models (if available)
            video_models = self.config.get('teachers', {}).get('video_generation', {})
            for model_key, model_name in video_models.items():
                try:
                    logger.info(f"Attempting to load video model: {model_name}")
                    # Note: These would be loaded if available on HuggingFace
                    # For now, we'll use placeholder implementations
                    self.teacher_models[f'video_{model_key}'] = f"placeholder_{model_name}"
                except Exception as e:
                    logger.warning(f"Could not load video model {model_name}: {e}")
            
            # Move models to device
            for key, model in self.teacher_models.items():
                if hasattr(model, 'to'):
                    self.teacher_models[key] = model.to(self.device)
                    
            logger.info(f"✅ Loaded {len(self.teacher_models)} teacher models")
            
        except Exception as e:
            logger.error(f"❌ Error loading teacher models: {e}")
            
    def generate_synthetic_dataset(self, num_samples: int = 1000) -> Dict[str, Any]:
        """
        Generate comprehensive synthetic dataset
        """
        logger.info(f"🎬 Starting synthetic data generation ({num_samples} samples)...")
        
        generation_results = {
            'template_based_samples': [],
            'llm_generated_samples': [],
            'beat_sync_samples': [],
            'transition_samples': [],
            'effect_samples': []
        }
        
        # 1. Generate template-based samples
        template_samples = self._generate_template_based_samples(num_samples // 2)
        generation_results['template_based_samples'] = template_samples
        
        # 2. Generate LLM-guided samples  
        llm_samples = self._generate_llm_guided_samples(num_samples // 4)
        generation_results['llm_generated_samples'] = llm_samples
        
        # 3. Generate beat-synchronized samples
        beat_samples = self._generate_beat_sync_samples(num_samples // 8)
        generation_results['beat_sync_samples'] = beat_samples
        
        # 4. Generate transition-focused samples
        transition_samples = self._generate_transition_samples(num_samples // 8)
        generation_results['transition_samples'] = transition_samples
        
        # 5. Generate effect-focused samples
        effect_samples = self._generate_effect_samples(num_samples // 8)
        generation_results['effect_samples'] = effect_samples
        
        # 6. Save comprehensive metadata
        self._save_synthetic_metadata(generation_results)
        
        logger.info(f"🎉 Synthetic data generation complete: {self.generation_stats}")
        return generation_results
        
    def _generate_template_based_samples(self, num_samples: int) -> List[SyntheticDataSample]:
        """Generate samples by applying templates to raw footage"""
        logger.info(f"📽️ Generating {num_samples} template-based samples...")
        
        samples = []
        
        # Load template metadata
        template_metadata_file = self.templates_dir / "metadata" / "templates_metadata.json"
        if not template_metadata_file.exists():
            logger.warning("No template metadata found, skipping template-based generation")
            return samples
            
        with open(template_metadata_file, 'r') as f:
            template_data = json.load(f)
            
        templates = template_data.get('templates', [])
        
        # Load stock footage
        stock_footage_files = list(self.raw_footage_dir.glob("*.mp4"))
        
        if not stock_footage_files:
            logger.warning("No stock footage found, skipping template-based generation")
            return samples
            
        # Generate samples
        for i in range(min(num_samples, len(templates) * len(stock_footage_files))):
            try:
                # Select template and footage
                template = templates[i % len(templates)]
                footage_file = stock_footage_files[i % len(stock_footage_files)]
                
                # Generate editing instructions using template
                editing_instructions = self._generate_template_instructions(template, footage_file)
                
                # Apply template to footage
                edited_video_path = self._apply_template_to_footage(
                    template, footage_file, editing_instructions, i
                )
                
                if edited_video_path:
                    sample = SyntheticDataSample(
                        id=f"template_{i:06d}",
                        raw_footage_path=str(footage_file),
                        template_path=template.get('file_path'),
                        edited_video_path=edited_video_path,
                        editing_instructions=editing_instructions,
                        metadata={
                            'template_id': template['id'],
                            'template_category': template['category'],
                            'template_source': template['source'],
                            'generation_method': 'template_based'
                        },
                        teacher_model_used='template_engine',
                        generation_timestamp=datetime.now().timestamp()
                    )
                    
                    samples.append(sample)
                    self.generation_stats['template_based'] += 1
                    
            except Exception as e:
                logger.error(f"Error generating template sample {i}: {e}")
                self.generation_stats['failed_generations'] += 1
                
        logger.info(f"✅ Generated {len(samples)} template-based samples")
        return samples
        
    def _generate_llm_guided_samples(self, num_samples: int) -> List[SyntheticDataSample]:
        """Generate samples using LLM to create editing instructions"""
        logger.info(f"🧠 Generating {num_samples} LLM-guided samples...")
        
        samples = []
        stock_footage_files = list(self.raw_footage_dir.glob("*.mp4"))
        
        if not stock_footage_files:
            logger.warning("No stock footage found for LLM generation")
            return samples
            
        for i in range(num_samples):
            try:
                # Select random footage
                footage_file = stock_footage_files[i % len(stock_footage_files)]
                
                # Analyze footage content
                content_analysis = self._analyze_footage_content(footage_file)
                
                # Generate editing instructions using LLM
                editing_instructions = self._generate_llm_instructions(content_analysis, footage_file)
                
                # Apply instructions to create edited video
                edited_video_path = self._apply_llm_instructions(
                    footage_file, editing_instructions, i
                )
                
                if edited_video_path:
                    sample = SyntheticDataSample(
                        id=f"llm_{i:06d}",
                        raw_footage_path=str(footage_file),
                        template_path=None,
                        edited_video_path=edited_video_path,
                        editing_instructions=editing_instructions,
                        metadata={
                            'content_analysis': content_analysis,
                            'generation_method': 'llm_guided',
                            'style': 'creative'
                        },
                        teacher_model_used='text_model',
                        generation_timestamp=datetime.now().timestamp()
                    )
                    
                    samples.append(sample)
                    self.generation_stats['llm_generated'] += 1
                    
            except Exception as e:
                logger.error(f"Error generating LLM sample {i}: {e}")
                self.generation_stats['failed_generations'] += 1
                
        logger.info(f"✅ Generated {len(samples)} LLM-guided samples")
        return samples
        
    def _generate_beat_sync_samples(self, num_samples: int) -> List[SyntheticDataSample]:
        """Generate beat-synchronized editing samples"""
        logger.info(f"🎵 Generating {num_samples} beat-sync samples...")
        
        samples = []
        
        # Find footage with audio
        footage_files = [f for f in self.raw_footage_dir.glob("*.mp4")]
        
        for i in range(min(num_samples, len(footage_files))):
            try:
                footage_file = footage_files[i]
                
                # Extract beat information
                beat_info = self._extract_beat_information(footage_file)
                
                if beat_info and beat_info['beats']:
                    # Generate beat-synchronized editing instructions
                    editing_instructions = self._generate_beat_sync_instructions(beat_info)
                    
                    # Apply beat-sync editing
                    edited_video_path = self._apply_beat_sync_editing(
                        footage_file, editing_instructions, beat_info, i
                    )
                    
                    if edited_video_path:
                        sample = SyntheticDataSample(
                            id=f"beatsync_{i:06d}",
                            raw_footage_path=str(footage_file),
                            template_path=None,
                            edited_video_path=edited_video_path,
                            editing_instructions=editing_instructions,
                            metadata={
                                'beat_info': beat_info,
                                'generation_method': 'beat_sync',
                                'style': 'rhythmic'
                            },
                            teacher_model_used='audio_model',
                            generation_timestamp=datetime.now().timestamp()
                        )
                        
                        samples.append(sample)
                        
            except Exception as e:
                logger.error(f"Error generating beat-sync sample {i}: {e}")
                
        logger.info(f"✅ Generated {len(samples)} beat-sync samples")
        return samples
        
    def _generate_transition_samples(self, num_samples: int) -> List[SyntheticDataSample]:
        """Generate samples focused on transitions"""
        logger.info(f"🔄 Generating {num_samples} transition-focused samples...")
        
        samples = []
        footage_files = list(self.raw_footage_dir.glob("*.mp4"))
        
        transition_types = [
            "fade_in_out", "crossfade", "slide_left", "slide_right", 
            "zoom_in", "zoom_out", "spin", "wipe", "dissolve"
        ]
        
        for i in range(min(num_samples, len(footage_files))):
            try:
                footage_file = footage_files[i]
                
                # Generate transition-focused instructions
                transition_type = transition_types[i % len(transition_types)]
                editing_instructions = self._generate_transition_instructions(
                    footage_file, transition_type
                )
                
                # Apply transition editing
                edited_video_path = self._apply_transition_editing(
                    footage_file, editing_instructions, transition_type, i
                )
                
                if edited_video_path:
                    sample = SyntheticDataSample(
                        id=f"transition_{i:06d}",
                        raw_footage_path=str(footage_file),
                        template_path=None,
                        edited_video_path=edited_video_path,
                        editing_instructions=editing_instructions,
                        metadata={
                            'transition_type': transition_type,
                            'generation_method': 'transition_focused',
                            'style': 'cinematic'
                        },
                        teacher_model_used='vision_model',
                        generation_timestamp=datetime.now().timestamp()
                    )
                    
                    samples.append(sample)
                    
            except Exception as e:
                logger.error(f"Error generating transition sample {i}: {e}")
                
        logger.info(f"✅ Generated {len(samples)} transition samples")
        return samples
        
    def _generate_effect_samples(self, num_samples: int) -> List[SyntheticDataSample]:
        """Generate samples with various effects"""
        logger.info(f"✨ Generating {num_samples} effect-focused samples...")
        
        samples = []
        footage_files = list(self.raw_footage_dir.glob("*.mp4"))
        
        effect_types = [
            "color_grading", "speed_ramp", "slow_motion", "time_lapse",
            "blur_effect", "sharpen", "vignette", "film_grain", "glitch"
        ]
        
        for i in range(min(num_samples, len(footage_files))):
            try:
                footage_file = footage_files[i]
                
                # Generate effect-focused instructions
                effect_type = effect_types[i % len(effect_types)]
                editing_instructions = self._generate_effect_instructions(
                    footage_file, effect_type
                )
                
                # Apply effect editing
                edited_video_path = self._apply_effect_editing(
                    footage_file, editing_instructions, effect_type, i
                )
                
                if edited_video_path:
                    sample = SyntheticDataSample(
                        id=f"effect_{i:06d}",
                        raw_footage_path=str(footage_file),
                        template_path=None,
                        edited_video_path=edited_video_path,
                        editing_instructions=editing_instructions,
                        metadata={
                            'effect_type': effect_type,
                            'generation_method': 'effect_focused',
                            'style': 'stylized'
                        },
                        teacher_model_used='vision_model',
                        generation_timestamp=datetime.now().timestamp()
                    )
                    
                    samples.append(sample)
                    
            except Exception as e:
                logger.error(f"Error generating effect sample {i}: {e}")
                
        logger.info(f"✅ Generated {len(samples)} effect samples")
        return samples
        
    def _generate_template_instructions(self, template: Dict, footage_file: Path) -> List[EditingInstruction]:
        """Generate editing instructions based on template"""
        instructions = []
        
        try:
            # Get template metadata
            duration = template.get('duration', 10.0)
            beat_markers = template.get('beat_markers', [])
            transition_points = template.get('transition_points', [])
            
            # Generate cuts at transition points
            for i, point in enumerate(transition_points[:5]):  # Limit to 5 cuts
                instructions.append(EditingInstruction(
                    action="cut",
                    timestamp=point,
                    duration=0.1,
                    parameters={'cut_type': 'hard_cut'},
                    confidence=0.9
                ))
                
            # Generate beat-sync effects
            for i, beat in enumerate(beat_markers[:10]):  # Limit to 10 beats
                if i % 2 == 0:  # Every other beat
                    instructions.append(EditingInstruction(
                        action="beat_sync",
                        timestamp=beat,
                        duration=0.2,
                        parameters={'effect': 'flash', 'intensity': 0.5},
                        confidence=0.8
                    ))
                    
            # Add template-specific effects
            category = template.get('category', 'general')
            if category == 'phonk':
                # Add phonk-style editing
                instructions.extend(self._generate_phonk_instructions(duration))
            elif category == 'transitions':
                # Add transition effects
                instructions.extend(self._generate_transition_effects())
                
        except Exception as e:
            logger.error(f"Error generating template instructions: {e}")
            
        return instructions
        
    def _generate_phonk_instructions(self, duration: float) -> List[EditingInstruction]:
        """Generate phonk-style editing instructions"""
        instructions = []
        
        # Typical phonk editing: fast cuts, bass drops, color shifts
        num_cuts = int(duration * 2)  # 2 cuts per second
        
        for i in range(num_cuts):
            timestamp = i * (duration / num_cuts)
            
            instructions.append(EditingInstruction(
                action="effect",
                timestamp=timestamp,
                duration=0.1,
                parameters={
                    'effect_type': 'color_grade',
                    'saturation': 1.5,
                    'contrast': 1.2,
                    'shadows': -0.3
                },
                confidence=0.9
            ))
            
            if i % 4 == 0:  # Every 4th cut
                instructions.append(EditingInstruction(
                    action="speed_change",
                    timestamp=timestamp,
                    duration=0.25,
                    parameters={'speed': 1.5},
                    confidence=0.8
                ))
                
        return instructions
        
    def _analyze_footage_content(self, footage_file: Path) -> Dict[str, Any]:
        """Analyze footage content using vision model"""
        try:
            # Load video and extract frame
            cap = cv2.VideoCapture(str(footage_file))
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return {'error': 'Could not read video'}
                
            # Use CLIP to analyze content
            processor = self.teacher_models.get('vision_processor')
            model = self.teacher_models.get('vision_model')
            
            if processor and model:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with CLIP
                inputs = processor(images=frame_rgb, return_tensors="pt")
                
                # Generate text prompts for analysis
                text_prompts = [
                    "a person", "landscape", "city", "nature", "indoor scene",
                    "outdoor scene", "movement", "static scene", "bright", "dark"
                ]
                
                text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True)
                
                with torch.no_grad():
                    image_features = model.get_image_features(**inputs)
                    text_features = model.get_text_features(**text_inputs)
                    
                    # Calculate similarities
                    similarities = torch.cosine_similarity(
                        image_features, text_features, dim=1
                    )
                    
                    # Get top matches
                    top_indices = similarities.argsort(descending=True)[:3]
                    top_matches = [text_prompts[i] for i in top_indices]
                    
                return {
                    'content_type': top_matches[0],
                    'secondary_content': top_matches[1:],
                    'brightness': float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()),
                    'has_motion': True,  # Simplified
                    'scene_complexity': 'medium'  # Simplified
                }
                
        except Exception as e:
            logger.error(f"Error analyzing footage content: {e}")
            
        return {'content_type': 'unknown', 'brightness': 128}
        
    def _generate_llm_instructions(self, content_analysis: Dict, footage_file: Path) -> List[EditingInstruction]:
        """Generate editing instructions using LLM"""
        instructions = []
        
        try:
            tokenizer = self.teacher_models.get('text_tokenizer')
            model = self.teacher_models.get('text_model')
            
            if not tokenizer or not model:
                return instructions
                
            # Create prompt based on content analysis
            content_type = content_analysis.get('content_type', 'unknown')
            brightness = content_analysis.get('brightness', 128)
            
            prompt = f"Edit a video with {content_type} content, brightness {brightness}. Create editing instructions:"
            
            # Generate text
            inputs = tokenizer.encode(prompt, return_tensors='pt')
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 50,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse generated text into instructions (simplified)
            # In practice, you'd have a more sophisticated parser
            if "cut" in generated_text.lower():
                instructions.append(EditingInstruction(
                    action="cut",
                    timestamp=2.0,
                    duration=0.1,
                    parameters={'cut_type': 'jump_cut'},
                    confidence=0.7
                ))
                
            if "fade" in generated_text.lower():
                instructions.append(EditingInstruction(
                    action="transition",
                    timestamp=5.0,
                    duration=1.0,
                    parameters={'transition_type': 'fade'},
                    confidence=0.8
                ))
                
        except Exception as e:
            logger.error(f"Error generating LLM instructions: {e}")
            
        return instructions
        
    def _extract_beat_information(self, footage_file: Path) -> Optional[Dict[str, Any]]:
        """Extract beat and rhythm information from video audio"""
        try:
            # Load audio from video
            video = mp.VideoFileClip(str(footage_file))
            if video.audio is None:
                return None
                
            # Extract audio
            audio_path = self.synthetic_output_dir / f"temp_audio_{footage_file.stem}.wav"
            video.audio.write_audiofile(str(audio_path), verbose=False, logger=None)
            
            # Use librosa to detect beats
            y, sr = librosa.load(str(audio_path))
            
            # Beat tracking
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')
            
            # Onset detection
            onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
            
            # Cleanup temp file
            audio_path.unlink(missing_ok=True)
            video.close()
            
            return {
                'tempo': float(tempo),
                'beats': beats.tolist(),
                'onsets': onsets.tolist(),
                'duration': float(len(y) / sr)
            }
            
        except Exception as e:
            logger.error(f"Error extracting beat information: {e}")
            return None
            
    def _generate_beat_sync_instructions(self, beat_info: Dict) -> List[EditingInstruction]:
        """Generate beat-synchronized editing instructions"""
        instructions = []
        
        beats = beat_info.get('beats', [])
        tempo = beat_info.get('tempo', 120)
        
        for i, beat_time in enumerate(beats[:20]):  # Limit to 20 beats
            if i % 2 == 0:  # Every other beat
                instructions.append(EditingInstruction(
                    action="cut",
                    timestamp=float(beat_time),
                    duration=0.05,
                    parameters={'sync_to_beat': True, 'beat_index': i},
                    confidence=0.95
                ))
            else:
                instructions.append(EditingInstruction(
                    action="effect",
                    timestamp=float(beat_time),
                    duration=0.1,
                    parameters={
                        'effect_type': 'zoom_pulse',
                        'intensity': 0.1,
                        'sync_to_beat': True
                    },
                    confidence=0.9
                ))
                
        return instructions
        
    def _apply_template_to_footage(self, template: Dict, footage_file: Path, 
                                 instructions: List[EditingInstruction], sample_id: int) -> Optional[str]:
        """Apply template to footage to create edited video"""
        try:
            # Load video
            video = mp.VideoFileClip(str(footage_file))
            
            # Apply editing instructions
            edited_clips = []
            
            # Sort instructions by timestamp
            instructions.sort(key=lambda x: x.timestamp)
            
            current_time = 0.0
            
            for instruction in instructions:
                # Add segment before instruction
                if instruction.timestamp > current_time:
                    segment = video.subclip(current_time, instruction.timestamp)
                    edited_clips.append(segment)
                
                # Apply instruction
                if instruction.action == "cut":
                    # For cuts, we just update the current time
                    current_time = instruction.timestamp + instruction.duration
                elif instruction.action == "beat_sync":
                    # Add a flash effect
                    segment = video.subclip(
                        instruction.timestamp, 
                        instruction.timestamp + instruction.duration
                    )
                    # Apply brightness increase for flash effect
                    segment = segment.fx(mp.vfx.colorx, 1.5)
                    edited_clips.append(segment)
                    current_time = instruction.timestamp + instruction.duration
                    
            # Add remaining video
            if current_time < video.duration:
                remaining = video.subclip(current_time, video.duration)
                edited_clips.append(remaining)
                
            # Concatenate clips
            if edited_clips:
                final_video = mp.concatenate_videoclips(edited_clips)
                
                # Save edited video
                output_path = self.synthetic_output_dir / f"template_edit_{sample_id:06d}.mp4"
                final_video.write_videofile(
                    str(output_path), 
                    verbose=False, 
                    logger=None,
                    codec='libx264',
                    audio_codec='aac'
                )
                
                # Cleanup
                video.close()
                final_video.close()
                for clip in edited_clips:
                    clip.close()
                    
                return str(output_path)
                
        except Exception as e:
            logger.error(f"Error applying template to footage: {e}")
            
        return None
        
    def _apply_llm_instructions(self, footage_file: Path, 
                              instructions: List[EditingInstruction], sample_id: int) -> Optional[str]:
        """Apply LLM-generated instructions to footage"""
        try:
            video = mp.VideoFileClip(str(footage_file))
            edited_video = video
            
            # Apply each instruction
            for instruction in instructions:
                if instruction.action == "cut":
                    # Simple cut implementation
                    if instruction.timestamp < video.duration:
                        edited_video = edited_video.subclip(0, instruction.timestamp)
                        
                elif instruction.action == "transition" and instruction.parameters.get('transition_type') == 'fade':
                    # Apply fade transition
                    edited_video = edited_video.fx(mp.vfx.fadein, 1.0).fx(mp.vfx.fadeout, 1.0)
                    
            # Save edited video
            output_path = self.synthetic_output_dir / f"llm_edit_{sample_id:06d}.mp4"
            edited_video.write_videofile(
                str(output_path),
                verbose=False,
                logger=None,
                codec='libx264',
                audio_codec='aac'
            )
            
            # Cleanup
            video.close()
            edited_video.close()
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error applying LLM instructions: {e}")
            
        return None
        
    def _apply_beat_sync_editing(self, footage_file: Path, instructions: List[EditingInstruction],
                               beat_info: Dict, sample_id: int) -> Optional[str]:
        """Apply beat-synchronized editing"""
        try:
            video = mp.VideoFileClip(str(footage_file))
            clips = []
            
            beats = beat_info.get('beats', [])
            
            # Create cuts at beat intervals
            for i in range(len(beats) - 1):
                start_time = beats[i]
                end_time = beats[i + 1]
                
                if start_time < video.duration and end_time <= video.duration:
                    clip = video.subclip(start_time, end_time)
                    
                    # Apply beat-sync effect (speed variation)
                    if i % 4 == 0:  # Every 4th beat
                        clip = clip.fx(mp.vfx.speedx, 1.2)
                    elif i % 2 == 0:  # Every other beat
                        clip = clip.fx(mp.vfx.colorx, 1.1)
                        
                    clips.append(clip)
                    
            if clips:
                final_video = mp.concatenate_videoclips(clips)
                
                output_path = self.synthetic_output_dir / f"beatsync_edit_{sample_id:06d}.mp4"
                final_video.write_videofile(
                    str(output_path),
                    verbose=False,
                    logger=None,
                    codec='libx264',
                    audio_codec='aac'
                )
                
                # Cleanup
                video.close()
                final_video.close()
                for clip in clips:
                    clip.close()
                    
                return str(output_path)
                
        except Exception as e:
            logger.error(f"Error applying beat-sync editing: {e}")
            
        return None
        
    def _generate_transition_instructions(self, footage_file: Path, transition_type: str) -> List[EditingInstruction]:
        """Generate transition-focused instructions"""
        return [
            EditingInstruction(
                action="transition",
                timestamp=1.0,
                duration=2.0,
                parameters={'transition_type': transition_type, 'intensity': 0.8},
                confidence=0.9
            )
        ]
        
    def _apply_transition_editing(self, footage_file: Path, instructions: List[EditingInstruction],
                                transition_type: str, sample_id: int) -> Optional[str]:
        """Apply transition editing"""
        try:
            video = mp.VideoFileClip(str(footage_file))
            
            # Apply transition effect based on type
            if transition_type == "fade_in_out":
                edited_video = video.fx(mp.vfx.fadein, 1.0).fx(mp.vfx.fadeout, 1.0)
            elif transition_type == "zoom_in":
                edited_video = video.fx(mp.vfx.resize, lambda t: 1 + 0.02 * t)
            else:
                # Default: simple crossfade effect
                edited_video = video.fx(mp.vfx.fadein, 0.5).fx(mp.vfx.fadeout, 0.5)
                
            output_path = self.synthetic_output_dir / f"transition_edit_{sample_id:06d}.mp4"
            edited_video.write_videofile(
                str(output_path),
                verbose=False,
                logger=None,
                codec='libx264',
                audio_codec='aac'
            )
            
            video.close()
            edited_video.close()
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error applying transition editing: {e}")
            
        return None
        
    def _generate_effect_instructions(self, footage_file: Path, effect_type: str) -> List[EditingInstruction]:
        """Generate effect-focused instructions"""
        return [
            EditingInstruction(
                action="effect",
                timestamp=0.0,
                duration=10.0,
                parameters={'effect_type': effect_type, 'intensity': 0.7},
                confidence=0.85
            )
        ]
        
    def _apply_effect_editing(self, footage_file: Path, instructions: List[EditingInstruction],
                            effect_type: str, sample_id: int) -> Optional[str]:
        """Apply effect editing"""
        try:
            video = mp.VideoFileClip(str(footage_file))
            
            # Apply effect based on type
            if effect_type == "slow_motion":
                edited_video = video.fx(mp.vfx.speedx, 0.5)
            elif effect_type == "time_lapse":
                edited_video = video.fx(mp.vfx.speedx, 2.0)
            elif effect_type == "color_grading":
                edited_video = video.fx(mp.vfx.colorx, 1.2)
            elif effect_type == "blur_effect":
                edited_video = video.fx(mp.vfx.blur, 2)
            else:
                # Default: brightness adjustment
                edited_video = video.fx(mp.vfx.colorx, 1.1)
                
            output_path = self.synthetic_output_dir / f"effect_edit_{sample_id:06d}.mp4"
            edited_video.write_videofile(
                str(output_path),
                verbose=False,
                logger=None,
                codec='libx264',
                audio_codec='aac'
            )
            
            video.close()
            edited_video.close()
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error applying effect editing: {e}")
            
        return None
        
    def _save_synthetic_metadata(self, generation_results: Dict[str, Any]):
        """Save comprehensive metadata for synthetic dataset"""
        metadata_file = self.synthetic_output_dir / "synthetic_dataset_metadata.json"
        
        # Collect all samples
        all_samples = []
        for sample_type, samples in generation_results.items():
            for sample in samples:
                sample_dict = {
                    'id': sample.id,
                    'raw_footage_path': sample.raw_footage_path,
                    'template_path': sample.template_path,
                    'edited_video_path': sample.edited_video_path,
                    'editing_instructions': [
                        {
                            'action': inst.action,
                            'timestamp': inst.timestamp,
                            'duration': inst.duration,
                            'parameters': inst.parameters,
                            'confidence': inst.confidence
                        }
                        for inst in sample.editing_instructions
                    ],
                    'metadata': sample.metadata,
                    'teacher_model_used': sample.teacher_model_used,
                    'generation_timestamp': sample.generation_timestamp,
                    'sample_type': sample_type
                }
                all_samples.append(sample_dict)
                
        # Create comprehensive metadata
        metadata = {
            'dataset_info': {
                'name': 'Synthetic Video Editing Dataset',
                'version': '1.0',
                'created_at': datetime.now().isoformat(),
                'total_samples': len(all_samples)
            },
            'generation_stats': self.generation_stats,
            'sample_types': {
                sample_type: len(samples) 
                for sample_type, samples in generation_results.items()
            },
            'teacher_models': list(self.teacher_models.keys()),
            'samples': all_samples
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"💾 Saved synthetic dataset metadata: {metadata_file}")
        logger.info(f"📊 Total synthetic samples: {len(all_samples)}")
        
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get generation statistics"""
        return {
            'generation_stats': self.generation_stats,
            'output_directory': str(self.synthetic_output_dir),
            'teacher_models_loaded': len(self.teacher_models)
        }