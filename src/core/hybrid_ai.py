"""
Hybrid Video AI - The core autonomous video editing AI system
Combines reasoning, perception, and editing capabilities in a unified model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import tempfile
from typing import Dict, List, Optional, Tuple, Any
from transformers import (
    AutoTokenizer, 
    AutoModel,
    LlamaForCausalLM,
    CLIPVisionModel,
    WhisperModel,
    WhisperProcessor
)

from models.multimodal_fusion import MultiModalFusionModule
from models.video_understanding import VideoUnderstandingModule
from models.editing_planner import EditingPlannerModule
from perception.vision_processor import VisionProcessor
from utils.model_downloader import ModelDownloader

# Setup logging
logger = logging.getLogger(__name__)
from audio.audio_processor import AudioProcessor
from editing.timeline_generator import TimelineGenerator


class HybridVideoAI(nn.Module):
    """
    Main Hybrid AI model that fuses multiple capabilities:
    - Language reasoning (CodeLLaMA/Mixtral)
    - Vision understanding (SigLIP + RT-DETR + SAM)
    - Audio analysis (Whisper + BeatNet)
    - Video editing logic (Custom transformer)
    - Code generation for effects
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Auto-download models if not available
        self._ensure_models_available()
        
        # Initialize tokenizer - Use teacher model for better capabilities
        model_name = config.get('teachers', {}).get('text_model', 'meta-llama/Llama-2-7b-hf')
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                padding_side="left",
                cache_dir=config.get('model_cache_dir', 'models/cache')
            )
        except Exception as e:
            logger.warning(f"Failed to load tokenizer for {model_name}, using fallback: {e}")
            # Fallback to a more compatible model
            fallback_model = 'microsoft/DialoGPT-small'
            self.tokenizer = AutoTokenizer.from_pretrained(
                fallback_model,
                padding_side="left", 
                cache_dir=config.get('model_cache_dir', 'models/cache')
            )
            model_name = fallback_model
            
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Determine dtype based on device availability and config
        use_bf16 = torch.cuda.is_available() and config.get('system', {}).get('mixed_precision', False)
        model_dtype = torch.bfloat16 if use_bf16 else torch.float32
        
        # Core reasoning backbone - Use advanced teacher model
        try:
            self.language_model = LlamaForCausalLM.from_pretrained(
                model_name,
                torch_dtype=model_dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                load_in_8bit=config.get('quantization', {}).get('load_in_8bit', False) and torch.cuda.is_available(),  # Only 8-bit on GPU
                cache_dir=config.get('model_cache_dir', 'models/cache')
            )
            logger.info(f"✅ Loaded advanced language model: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load advanced language model {model_name}, using fallback: {e}")
            fallback_model = 'microsoft/DialoGPT-small'
            self.language_model = AutoModel.from_pretrained(
                fallback_model,
                cache_dir=config.get('model_cache_dir', 'models/cache')
            )
        
        # Multi-model ensemble for robust reasoning
        self.ensemble_models = {}
        self.ensemble_tokenizers = {}
        self._initialize_ensemble_models(config)
        
        # Vision encoder - Use SigLIP for better visual understanding
        vision_model = config.get('teachers', {}).get('vision_encoder', 'google/siglip-large-patch16-384')
        try:
            from transformers import SiglipVisionModel
            self.vision_encoder = SiglipVisionModel.from_pretrained(
                vision_model,
                cache_dir=config.get('model_cache_dir', 'models/cache')
            )
            logger.info(f"✅ Loaded advanced vision model: {vision_model}")
        except Exception as e:
            logger.warning(f"Failed to load SigLIP, using CLIP fallback: {e}")
            vision_model = config.get('teachers', {}).get('vision_encoder', 'openai/clip-vit-base-patch32')
            self.vision_encoder = CLIPVisionModel.from_pretrained(
                vision_model,
                cache_dir=config.get('model_cache_dir', 'models/cache')
            )
        
        # Audio encoder - Use Whisper Large for better audio understanding
        audio_model = config.get('teachers', {}).get('audio_models', ['openai/whisper-large-v3'])[0]
        try:
            self.audio_encoder = WhisperModel.from_pretrained(
                audio_model,
                cache_dir=config.get('model_cache_dir', 'models/cache'),
                torch_dtype=model_dtype
            )
            self.whisper_processor = WhisperProcessor.from_pretrained(
                audio_model,
                cache_dir=config.get('model_cache_dir', 'models/cache')
            )
            logger.info(f"✅ Loaded advanced audio model: {audio_model}")
        except Exception as e:
            logger.warning(f"Failed to load Whisper Large, using base model: {e}")
            audio_model = 'openai/whisper-base'
            self.audio_encoder = WhisperModel.from_pretrained(
                audio_model,
                cache_dir=config.get('model_cache_dir', 'models/cache')
            )
            self.whisper_processor = WhisperProcessor.from_pretrained(
                audio_model,
                cache_dir=config.get('model_cache_dir', 'models/cache')
            )
        
        # Initialize model downloader
        self.model_downloader = ModelDownloader(
            cache_dir=config.get('model_cache_dir', 'models/cache')
        )
        
        # Detect actual embedding dimensions from loaded models
        text_dim = self.language_model.get_input_embeddings().embedding_dim
        vision_dim = self.vision_encoder.config.hidden_size
        audio_dim = self.audio_encoder.config.d_model
        
        logger.info(f"Detected embedding dimensions:")
        logger.info(f"  text_dim: {text_dim}")
        logger.info(f"  vision_dim: {vision_dim}")
        logger.info(f"  audio_dim: {audio_dim}")
        
        # Multimodal fusion layer with detected dimensions
        model_config = config.get('model', {})
        self.fusion_module = MultiModalFusionModule(
            text_dim=text_dim,
            vision_dim=vision_dim, 
            audio_dim=audio_dim,
            fusion_dim=model_config.get('fusion_dim', 1024),
            num_heads=model_config.get('num_attention_heads', 16)
        )
        
        # Self-coding engine for dynamic effect generation
        try:
            from ..generation.self_coding_engine import SelfCodingVideoEditor
            self.self_coding_engine = SelfCodingVideoEditor(config)
            self.has_self_coding = True
            logger.info("✅ Self-coding engine enabled")
        except ImportError as e:
            logger.warning(f"Self-coding engine not available: {e}")
            self.self_coding_engine = None
            self.has_self_coding = False
        
        # Video understanding module
        self.video_understanding = VideoUnderstandingModule(
            fusion_dim=model_config.get('fusion_dim', 1024),
            hidden_dim=model_config.get('hidden_dim', 2048)
        )
        
        # Editing planner 
        self.editing_planner = EditingPlannerModule(
            hidden_dim=model_config.get('hidden_dim', 2048),
            vocab_size=len(self.tokenizer)
        )
        
        # Specialized processors
        self.vision_processor = VisionProcessor(config)
        self.audio_processor = AudioProcessor(config)
        self.timeline_generator = TimelineGenerator(config)
        
        # Training phase tracker
        self.current_phase = "pretraining"
        
        # Analysis memory for learning from successful prompts
        self.analysis_memory = {
            'successful_prompts': [],
            'successful_effects': [],
            'failed_attempts': [],
            'model_performance': {}
        }
        
    def _ensure_models_available(self):
        """Ensure all required models are downloaded and available"""
        try:
            # Get model names from config
            backbone_model = self.config.get('teachers', {}).get('text_model', 'microsoft/DialoGPT-small')
            vision_model = self.config.get('teachers', {}).get('vision_encoder', 'openai/clip-vit-base-patch32') 
            audio_model = self.config.get('teachers', {}).get('audio_models', ['openai/whisper-tiny'])[0]
            
            # Auto-download models using the model downloader
            model_downloader = ModelDownloader(
                cache_dir=self.config.get('model_cache_dir', 'models/cache')
            )
            
            logger.info("Ensuring required models are available...")
            
            # Download all models with fallback handling
            model_downloader.download_all_models()
            
            logger.info("All required models are available")
            
        except Exception as e:
            logger.warning(f"Model download failed, will attempt to load from cache: {e}")
    
    def _initialize_ensemble_models(self, config: Dict[str, Any]):
        """Initialize ensemble of language models for robust consensus"""
        
        # Define ensemble models with fallback chain
        ensemble_candidates = [
            {
                'name': 'microsoft/DialoGPT-medium',
                'type': 'dialog',
                'priority': 1
            },
            {
                'name': 'microsoft/DialoGPT-small', 
                'type': 'dialog',
                'priority': 2
            },
            {
                'name': 'distilbert-base-uncased',
                'type': 'encoder',
                'priority': 3
            }
        ]
        
        # Try to load ensemble models
        for model_config in ensemble_candidates:
            try:
                model_name = model_config['name']
                
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=config.get('model_cache_dir', 'models/cache')
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Load model based on type
                if model_config['type'] == 'dialog':
                    try:
                        from transformers import AutoModelForCausalLM
                        model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            cache_dir=config.get('model_cache_dir', 'models/cache')
                        )
                    except Exception:
                        model = AutoModel.from_pretrained(
                            model_name,
                            cache_dir=config.get('model_cache_dir', 'models/cache')
                        )
                else:
                    model = AutoModel.from_pretrained(
                        model_name,
                        cache_dir=config.get('model_cache_dir', 'models/cache')
                    )
                
                # Store in ensemble
                self.ensemble_models[model_name] = {
                    'model': model,
                    'type': model_config['type'],
                    'priority': model_config['priority'],
                    'success_rate': 0.5,  # Initial success rate
                    'total_attempts': 0,
                    'successful_attempts': 0
                }
                self.ensemble_tokenizers[model_name] = tokenizer
                
                logger.info(f"✅ Loaded ensemble model: {model_name}")
                
            except Exception as e:
                logger.warning(f"Failed to load ensemble model {model_config['name']}: {e}")
        
        logger.info(f"🎯 Ensemble initialized with {len(self.ensemble_models)} models")
    
    def forward(self, 
                video_frames: Optional[torch.Tensor] = None,
                audio_features: Optional[torch.Tensor] = None, 
                text_input_ids: Optional[torch.Tensor] = None,
                text_attention_mask: Optional[torch.Tensor] = None,
                editing_prompt: Optional[str] = None,
                return_timeline: bool = False,
                custom_effects: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the hybrid AI system
        
        Args:
            video_frames: (B, T, C, H, W) video tensor
            audio_features: (B, T, F) audio features
            text_input_ids: (B, L) tokenized text
            text_attention_mask: (B, L) attention mask
            editing_prompt: Natural language editing instruction
            return_timeline: Whether to generate editing timeline
            
        Returns:
            Dictionary with model outputs
        """
        logger.info("=" * 80)
        logger.info("FORWARD PASS STARTED")
        logger.info("Input shapes:")
        if video_frames is not None:
            logger.info(f"  video_frames: {video_frames.shape}")
        if audio_features is not None:
            logger.info(f"  audio_features: {audio_features.shape}")
        if text_input_ids is not None:
            logger.info(f"  text_input_ids: {text_input_ids.shape}")
        logger.info("=" * 80)
        
        outputs = {}
        
        # Process text input
        if text_input_ids is not None:
            text_embeddings = self.language_model.get_input_embeddings()(text_input_ids)
            outputs['text_embeddings'] = text_embeddings
        
        # Process vision input
        if video_frames is not None:
            B, T, C, H, W = video_frames.shape
            # Flatten time dimension for vision encoder
            frames_flat = video_frames.view(B * T, C, H, W)
            vision_outputs = self.vision_encoder(pixel_values=frames_flat)
            
            # IMPORTANT: Use pooler_output if available (single vector per image),
            # otherwise use CLS token (first token) or mean pooling
            if hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
                # This gives us (B*T, vision_dim)
                vision_embeddings = vision_outputs.pooler_output
            elif hasattr(vision_outputs, 'last_hidden_state'):
                # Extract CLS token (first token) instead of mean pooling
                # This gives consistent dimensionality
                vision_embeddings = vision_outputs.last_hidden_state[:, 0, :]  # Take CLS token
            else:
                raise ValueError("Vision encoder output format not recognized")
            
            # Reshape to (B, T, vision_dim)
            vision_embeddings = vision_embeddings.view(B, T, -1)
            outputs['vision_embeddings'] = vision_embeddings
            
        # Process audio input
        if audio_features is not None:
            # Convert raw audio to mel-spectrogram features using WhisperProcessor
            # audio_features should be raw audio waveform (batch_size, audio_length) or (batch_size, channels, audio_length)
            if audio_features.dim() == 3:
                # If (B, C, L), take first channel
                audio_features = audio_features[:, 0, :]
            
            # Process audio to mel-spectrogram using the processor
            # The processor expects a list of numpy arrays
            audio_list = [audio.cpu().numpy() for audio in audio_features]
            processed_audio = self.audio_processor.whisper_processor(
                audio_list,
                sampling_rate=16000,
                return_tensors="pt",
                padding="max_length"
            )
            
            # Move to correct device and dtype
            mel_features = processed_audio.input_features.to(audio_features.device)
            
            # Use Whisper encoder for audio embeddings
            audio_embeddings = self.audio_encoder.encoder(mel_features).last_hidden_state
            B, audio_seq_len, audio_dim = audio_embeddings.shape
            
            # Pool audio embeddings to match video frame count if available
            if video_frames is not None and 'vision_embeddings' in outputs:
                T = outputs['vision_embeddings'].shape[1]  # Get actual number of video frames
                # Use adaptive average pooling to match video frames
                # Transpose to (B, audio_dim, audio_seq_len) for pooling
                audio_transposed = audio_embeddings.transpose(1, 2)
                # Pool to target sequence length T
                audio_pooled = F.adaptive_avg_pool1d(audio_transposed, T)
                # Transpose back to (B, T, audio_dim)
                audio_embeddings = audio_pooled.transpose(1, 2)
            else:
                # Use a default target sequence length
                target_len = 8
                if audio_seq_len > target_len:
                    # Use adaptive pooling
                    audio_transposed = audio_embeddings.transpose(1, 2)
                    audio_pooled = F.adaptive_avg_pool1d(audio_transposed, target_len)
                    audio_embeddings = audio_pooled.transpose(1, 2)
            outputs['audio_embeddings'] = audio_embeddings
            
        # Multimodal fusion
        if any(k in outputs for k in ['text_embeddings', 'vision_embeddings', 'audio_embeddings']):
            # Log shapes before fusion
            logger.info("="*60)
            logger.info("MULTIMODAL FUSION - Input Shapes:")
            if 'text_embeddings' in outputs:
                logger.info(f"  text_embeddings: {outputs['text_embeddings'].shape}")
            else:
                logger.info(f"  text_embeddings: None")
            if 'vision_embeddings' in outputs:
                logger.info(f"  vision_embeddings: {outputs['vision_embeddings'].shape}")
            else:
                logger.info(f"  vision_embeddings: None")
            if 'audio_embeddings' in outputs:
                logger.info(f"  audio_embeddings: {outputs['audio_embeddings'].shape}")
            else:
                logger.info(f"  audio_embeddings: None")
            logger.info("="*60)
            
            try:
                fused_embeddings = self.fusion_module(
                    text_emb=outputs.get('text_embeddings'),
                    vision_emb=outputs.get('vision_embeddings'), 
                    audio_emb=outputs.get('audio_embeddings')
                )
                outputs['fused_embeddings'] = fused_embeddings
                logger.info(f"✓ Fusion successful - output shape: {fused_embeddings.shape}")
                
                # Video understanding
                logger.info("Starting video understanding...")
                video_understanding = self.video_understanding(fused_embeddings)
                outputs['video_understanding'] = video_understanding
                logger.info("✓ Video understanding successful")
            except RuntimeError as e:
                # Log shapes for debugging
                logger.error("="*60)
                logger.error(f"ERROR in multimodal fusion or video understanding!")
                logger.error(f"Error message: {e}")
                if 'text_embeddings' in outputs:
                    logger.error(f"  text_embeddings shape: {outputs['text_embeddings'].shape}")
                if 'vision_embeddings' in outputs:
                    logger.error(f"  vision_embeddings shape: {outputs['vision_embeddings'].shape}")
                if 'audio_embeddings' in outputs:
                    logger.error(f"  audio_embeddings shape: {outputs['audio_embeddings'].shape}")
                if 'fused_embeddings' in outputs:
                    logger.error(f"  fused_embeddings shape: {outputs['fused_embeddings'].shape}")
                logger.error(f"  Error message: {str(e)}")
                raise
            
        # Generate editing plan if requested
        if return_timeline and editing_prompt:
            timeline = self.generate_editing_timeline(editing_prompt, outputs)
            outputs['timeline'] = timeline
            
        return outputs
    
    def generate_editing_timeline(self, prompt: str, context: Dict[str, torch.Tensor]) -> Dict:
        """Generate editing timeline from natural language prompt"""
        
        # Encode prompt
        prompt_tokens = self.tokenizer.encode(prompt, return_tensors='pt')
        if torch.cuda.is_available():
            prompt_tokens = prompt_tokens.cuda()
            
        # Use editing planner to generate timeline tokens
        with torch.no_grad():
            timeline_logits = self.editing_planner(
                input_ids=prompt_tokens,
                context_embeddings=context.get('fused_embeddings')
            )
            
        # Convert to editing timeline
        timeline = self.timeline_generator.decode_timeline(timeline_logits)
        return timeline
    
    def autonomous_edit(self, media_files=None, prompt: str = None, video_paths=None, video_path: str = None) -> str:
        """
        Fully autonomous video editing based on natural language prompt
        Enhanced with self-coding capabilities for custom effects
        Supports multiple input media types: videos, images, and audio
        
        Args:
            media_files: Dict with keys 'videos', 'images', 'audio' containing file paths (new parameter)
            video_paths: List of paths to input videos (for backward compatibility)
            video_path: Single video path (for backward compatibility)
            prompt: Natural language editing instruction
            
        Returns:
            Path to edited video
        """
        # Handle backward compatibility and parameter validation
        if media_files is None and video_paths is None and video_path is None:
            raise ValueError("Either media_files, video_paths, or video_path must be provided")
        
        if prompt is None:
            raise ValueError("prompt parameter is required")
        
        # Handle backwards compatibility
        if video_path is not None:
            media_files = {'videos': [video_path], 'images': [], 'audio': []}
        elif video_paths is not None:
            if isinstance(video_paths, str):
                video_paths = [video_paths]
            media_files = {'videos': video_paths, 'images': [], 'audio': []}
        elif media_files is None:
            raise ValueError("media_files parameter is required")
        
        # Ensure media_files has the expected structure
        if not isinstance(media_files, dict):
            raise ValueError("media_files must be a dictionary with 'videos', 'images', 'audio' keys")
        
        videos = media_files.get('videos', [])
        images = media_files.get('images', [])
        audios = media_files.get('audio', [])
            
        logger.info(f"🎬 Starting autonomous multimedia edit: {len(videos)} video(s), {len(images)} image(s), {len(audios)} audio(s)")
        logger.info(f"� Prompt: {prompt}")
        
        # Process multimedia inputs to create comprehensive video data
        multimedia_data = self._process_multimedia_inputs(videos, images, audios)
        
        # Extract video and audio data from multimedia composition
        video_data = multimedia_data['video_data']
        audio_data = multimedia_data['audio_data']
        
        # Analyze prompt for custom effect requirements
        custom_effects = self._analyze_prompt_for_custom_effects(prompt)
        
        # Generate custom effects if needed using self-coding
        generated_effects = {}
        if custom_effects and self.has_self_coding:
            logger.info(f"🔧 Generating {len(custom_effects)} custom effects...")
            for effect_desc in custom_effects:
                try:
                    effect_func = self.self_coding_engine.create_custom_effect(
                        effect_desc, 
                        test_frame=video_data['frames'][0] if len(video_data['frames']) > 0 else None
                    )
                    if effect_func:
                        generated_effects[effect_desc] = effect_func
                        logger.info(f"✅ Generated effect: {effect_desc}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to generate effect '{effect_desc}': {e}")
        
        # Run inference with enhanced editing capabilities
        outputs = self.forward(
            video_frames=video_data['frames'],
            audio_features=audio_data['features'],
            editing_prompt=prompt,
            return_timeline=True,
            custom_effects=generated_effects
        )
        
        # Generate final video with custom effects applied
        output_path = self.timeline_generator.render_video(
            timeline=outputs['timeline'],
            video_data=video_data,
            audio_data=audio_data,
            custom_effects=generated_effects
        )
        
        logger.info(f"🎉 Autonomous edit complete: {output_path}")
        return output_path
    
    def _concatenate_videos(self, video_paths: List[str]) -> str:
        """
        Concatenate multiple videos into a single video file
        
        Args:
            video_paths: List of paths to video files to concatenate
            
        Returns:
            Path to concatenated video file
        """
        try:
            from moviepy.editor import VideoFileClip, concatenate_videoclips
            
            logger.info(f"🔗 Loading {len(video_paths)} videos for concatenation...")
            clips = []
            
            for i, video_path in enumerate(video_paths):
                if not os.path.exists(video_path):
                    logger.error(f"Video file not found: {video_path}")
                    continue
                    
                try:
                    clip = VideoFileClip(video_path)
                    clips.append(clip)
                    logger.info(f"✅ Loaded video {i+1}/{len(video_paths)}: {video_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load video {video_path}: {e}")
                    continue
            
            if not clips:
                raise ValueError("No valid video clips could be loaded")
            
            if len(clips) == 1:
                logger.info("Only one valid video found, using it directly")
                concatenated_clip = clips[0]
            else:
                logger.info(f"🎬 Concatenating {len(clips)} video clips...")
                concatenated_clip = concatenate_videoclips(clips, method="compose")
            
            # Create temporary file for concatenated video
            temp_dir = tempfile.gettempdir()
            concatenated_path = os.path.join(temp_dir, f"concatenated_video_{os.getpid()}.mp4")
            
            # Write concatenated video
            logger.info(f"💾 Writing concatenated video to: {concatenated_path}")
            concatenated_clip.write_videofile(
                concatenated_path,
                codec='libx264',
                audio_codec='aac',
                verbose=False,
                logger=None  # Suppress moviepy logs
            )
            
            # Clean up clips
            for clip in clips:
                clip.close()
            concatenated_clip.close()
            
            logger.info(f"✅ Video concatenation complete: {concatenated_path}")
            return concatenated_path
            
        except Exception as e:
            logger.error(f"❌ Failed to concatenate videos: {e}")
            # Fallback: use first valid video
            for video_path in video_paths:
                if os.path.exists(video_path):
                    logger.warning(f"⚠️ Falling back to first valid video: {video_path}")
                    return video_path
            raise ValueError(f"No valid video files found in: {video_paths}")
    
    def _process_multimedia_inputs(self, videos: List[str], images: List[str], audios: List[str]) -> Dict[str, Any]:
        """
        Process mixed multimedia inputs and create a unified video composition
        
        Args:
            videos: List of video file paths
            images: List of image file paths  
            audios: List of audio file paths
            
        Returns:
            Dict containing processed video and audio data
        """
        logger.info(f"🎯 Processing multimedia inputs: {len(videos)} videos, {len(images)} images, {len(audios)} audios")
        
        try:
            from moviepy.editor import VideoFileClip, ImageClip, AudioFileClip, concatenate_videoclips, CompositeAudioClip
            import cv2
            import numpy as np
            
            # Collect all video clips
            video_clips = []
            
            # Process video files
            if videos:
                logger.info(f"📹 Processing {len(videos)} video files...")
                for i, video_path in enumerate(videos):
                    if os.path.exists(video_path):
                        try:
                            clip = VideoFileClip(video_path)
                            video_clips.append(clip)
                            logger.info(f"✅ Loaded video {i+1}: {video_path}")
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to load video {video_path}: {e}")
            
            # Process image files (convert to video clips)
            if images:
                logger.info(f"🖼️ Processing {len(images)} image files...")
                default_duration = 3.0  # Default 3 seconds per image
                for i, image_path in enumerate(images):
                    if os.path.exists(image_path):
                        try:
                            # Create video clip from image
                            img_clip = ImageClip(image_path, duration=default_duration)
                            video_clips.append(img_clip)
                            logger.info(f"✅ Loaded image {i+1}: {image_path} (duration: {default_duration}s)")
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to load image {image_path}: {e}")
            
            # Create base video composition
            if not video_clips:
                logger.warning("⚠️ No valid video or image content found, creating blank video")
                # Create a 5-second blank video as fallback
                blank_clip = ImageClip(np.zeros((720, 1280, 3), dtype=np.uint8), duration=5.0)
                video_clips.append(blank_clip)
            
            # Concatenate all video content
            if len(video_clips) == 1:
                final_video_clip = video_clips[0]
            else:
                logger.info(f"🔗 Concatenating {len(video_clips)} video segments...")
                final_video_clip = concatenate_videoclips(video_clips, method="compose")
            
            # Process audio files
            audio_clips = []
            
            # Add existing audio from videos (already included in video clips)
            if final_video_clip.audio is not None:
                audio_clips.append(final_video_clip.audio)
            
            # Process standalone audio files
            if audios:
                logger.info(f"🎵 Processing {len(audios)} audio files...")
                for i, audio_path in enumerate(audios):
                    if os.path.exists(audio_path):
                        try:
                            audio_clip = AudioFileClip(audio_path)
                            audio_clips.append(audio_clip)
                            logger.info(f"✅ Loaded audio {i+1}: {audio_path}")
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to load audio {audio_path}: {e}")
            
            # Combine all audio
            if len(audio_clips) > 1:
                logger.info(f"🎵 Mixing {len(audio_clips)} audio tracks...")
                final_audio = CompositeAudioClip(audio_clips)
            elif len(audio_clips) == 1:
                final_audio = audio_clips[0]
            else:
                final_audio = None
            
            # Apply audio to video
            if final_audio is not None:
                final_video_clip = final_video_clip.set_audio(final_audio)
            
            # Save temporary composed video
            temp_dir = tempfile.gettempdir()
            temp_video_path = os.path.join(temp_dir, f"multimedia_composition_{os.getpid()}.mp4")
            
            logger.info(f"💾 Writing multimedia composition to: {temp_video_path}")
            final_video_clip.write_videofile(
                temp_video_path,
                codec='libx264',
                audio_codec='aac',
                verbose=False,
                logger=None
            )
            
            # Clean up clips
            for clip in video_clips:
                clip.close()
            for clip in audio_clips:
                clip.close()
            final_video_clip.close()
            
            # Now process the composed video through our standard pipeline
            logger.info("🔍 Analyzing composed multimedia content...")
            video_data = self.vision_processor.load_video(temp_video_path)
            audio_data = self.audio_processor.load_audio(temp_video_path)
            
            # Add metadata about source files
            multimedia_metadata = {
                'source_videos': len(videos),
                'source_images': len(images), 
                'source_audios': len(audios),
                'composition_path': temp_video_path,
                'total_duration': video_data.get('duration', 0),
                'mixed_media': True
            }
            
            video_data['multimedia_metadata'] = multimedia_metadata
            audio_data['multimedia_metadata'] = multimedia_metadata
            
            logger.info(f"✅ Multimedia processing complete - Duration: {multimedia_metadata['total_duration']:.2f}s")
            
            return {
                'video_data': video_data,
                'audio_data': audio_data,
                'temp_composition_path': temp_video_path
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to process multimedia inputs: {e}")
            # Fallback: try to use first available media file
            all_media = videos + images + audios
            if all_media:
                fallback_path = all_media[0]
                logger.warning(f"⚠️ Falling back to single media file: {fallback_path}")
                
                if fallback_path in videos:
                    video_data = self.vision_processor.load_video(fallback_path)
                    audio_data = self.audio_processor.load_audio(fallback_path)
                else:
                    # For images/audio, create minimal video data
                    video_data = self._create_minimal_video_data()
                    audio_data = self._create_minimal_audio_data()
                
                return {
                    'video_data': video_data,
                    'audio_data': audio_data,
                    'temp_composition_path': None
                }
            else:
                raise ValueError("No valid media files provided")
    
    def _create_minimal_video_data(self) -> Dict[str, Any]:
        """Create minimal video data for fallback scenarios"""
        return {
            'frames': torch.zeros((1, 3, 224, 224)),  # Single black frame
            'fps': 30.0,
            'duration': 1.0,
            'num_frames': 1,
            'width': 224,
            'height': 224,
            'embeddings': None,
            'detections': [],
            'scene_stats': {},
            'temporal_features': None
        }
    
    def _create_minimal_audio_data(self) -> Dict[str, Any]:
        """Create minimal audio data for fallback scenarios"""
        return {
            'features': {
                'mfcc': torch.zeros((1, 13)),
                'spectral_centroid': torch.zeros(1),
                'spectral_rolloff': torch.zeros(1),
                'zero_crossing_rate': torch.zeros(1),
                'tempo': torch.tensor([120.0]),
                'chroma': torch.zeros((1, 12)),
                'mel_spectrogram': torch.zeros((1, 128, 100))
            },
            'transcription': {'text': '', 'confidence': 0.0, 'language': 'unknown'},
            'content_analysis': {},
            'events': [],
            'audio_path': None
        }
    
    def _analyze_prompt_for_custom_effects(self, prompt: str) -> List[str]:
        """
        Sophisticated LLM-powered analysis of editing prompt using advanced reasoning capabilities.
        Employs multi-stage analysis with context understanding, retry mechanisms, and robust error handling.
        """
        
        # Try multiple analysis approaches for maximum robustness
        analysis_results = []
        
        # Approach 1: Multi-model ensemble consensus
        ensemble_results = self._perform_ensemble_consensus_analysis(prompt)
        if ensemble_results:
            analysis_results.extend(ensemble_results)
            logger.info(f"🎯 Ensemble consensus succeeded: {len(ensemble_results)} effects")
        
        # Approach 2: Structured analysis with explicit formatting
        structured_results = self._perform_structured_analysis(prompt)
        if structured_results:
            analysis_results.extend(structured_results)
            logger.info(f"✅ Structured analysis succeeded: {len(structured_results)} effects")
        
        # Approach 3: Creative interpretation with broader context
        creative_results = self._perform_creative_analysis(prompt)
        if creative_results:
            analysis_results.extend(creative_results)
            logger.info(f"✅ Creative analysis succeeded: {len(creative_results)} effects")
        
        # Approach 4: Technical requirements extraction
        technical_results = self._perform_technical_analysis(prompt)
        if technical_results:
            analysis_results.extend(technical_results)
            logger.info(f"✅ Technical analysis succeeded: {len(technical_results)} effects")
        
        # Combine and validate results
        if analysis_results:
            final_effects = self._combine_and_validate_analysis_results(analysis_results, prompt)
            if self._validate_analysis_quality(final_effects, prompt):
                # Learn from successful analysis
                self._learn_from_successful_analysis(prompt, final_effects, analysis_results)
                logger.info(f"🎯 High-quality LLM analysis completed: {len(final_effects)} effects identified")
                return final_effects
            else:
                logger.warning("⚠️ LLM analysis quality below threshold, attempting refinement")
                # Try memory-enhanced refined approach
                refined_results = self._perform_memory_enhanced_analysis(prompt, analysis_results)
                if refined_results and self._validate_analysis_quality(refined_results, prompt):
                    self._learn_from_successful_analysis(prompt, refined_results, [refined_results])
                    logger.info(f"✅ Memory-enhanced analysis successful: {len(refined_results)} effects")
                    return refined_results
        
        # Record failed attempt for learning
        self._record_failed_analysis_attempt(prompt, analysis_results)
        
        # Only fall back if all LLM approaches fail or produce low-quality results
        logger.warning("🔄 All LLM analysis approaches failed or produced low-quality results, using enhanced fallback")
        return self._enhanced_contextual_fallback(prompt)
    
    def _learn_from_successful_analysis(self, prompt: str, effects: List[str], raw_results: List[List[str]]):
        """Learn from successful analysis to improve future performance"""
        
        try:
            # Store successful prompt pattern
            prompt_pattern = self._extract_prompt_pattern(prompt)
            
            # Store in memory
            success_entry = {
                'prompt': prompt,
                'prompt_pattern': prompt_pattern,
                'effects': effects,
                'raw_results': raw_results,
                'timestamp': torch.tensor(float(hash(prompt) % 1000000)),  # Simple timestamp
                'quality_score': len(effects)
            }
            
            self.analysis_memory['successful_prompts'].append(success_entry)
            self.analysis_memory['successful_effects'].extend(effects)
            
            # Maintain memory size limit
            if len(self.analysis_memory['successful_prompts']) > 100:
                # Keep most recent and highest quality
                sorted_memories = sorted(
                    self.analysis_memory['successful_prompts'],
                    key=lambda x: (x['quality_score'], x['timestamp']),
                    reverse=True
                )
                self.analysis_memory['successful_prompts'] = sorted_memories[:50]
            
            logger.info(f"📚 Learned from successful analysis. Memory size: {len(self.analysis_memory['successful_prompts'])}")
            
        except Exception as e:
            logger.warning(f"Failed to learn from successful analysis: {e}")
    
    def _extract_prompt_pattern(self, prompt: str) -> Dict[str, Any]:
        """Extract patterns from prompt for memory matching"""
        
        pattern = {
            'length': len(prompt),
            'keywords': [],
            'style_indicators': [],
            'technical_terms': []
        }
        
        try:
            prompt_lower = prompt.lower()
            
            # Extract keywords
            import re
            
            # Style indicators
            style_words = re.findall(r'\b(?:cinematic|vintage|modern|artistic|dramatic|subtle|bold|professional|creative)\b', prompt_lower)
            pattern['style_indicators'] = list(set(style_words))
            
            # Technical terms
            tech_words = re.findall(r'\b(?:color|grade|grading|effect|filter|transition|composite|motion|track|stabiliz)\w*\b', prompt_lower)
            pattern['technical_terms'] = list(set(tech_words))
            
            # General keywords (remove common words)
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            words = re.findall(r'\b\w+\b', prompt_lower)
            keywords = [word for word in words if len(word) > 3 and word not in stop_words]
            pattern['keywords'] = list(set(keywords))[:10]  # Limit to 10 most unique
            
        except Exception as e:
            logger.warning(f"Failed to extract prompt pattern: {e}")
        
        return pattern
    
    def _record_failed_analysis_attempt(self, prompt: str, attempted_results: List[List[str]]):
        """Record failed analysis attempts for learning"""
        
        try:
            failure_entry = {
                'prompt': prompt,
                'attempted_results': attempted_results,
                'timestamp': torch.tensor(float(hash(prompt) % 1000000)),
                'failure_reasons': []
            }
            
            self.analysis_memory['failed_attempts'].append(failure_entry)
            
            # Maintain memory size limit
            if len(self.analysis_memory['failed_attempts']) > 50:
                # Keep most recent
                self.analysis_memory['failed_attempts'] = self.analysis_memory['failed_attempts'][-25:]
            
        except Exception as e:
            logger.warning(f"Failed to record failure: {e}")
    
    def _perform_memory_enhanced_analysis(self, prompt: str, previous_results: List[List[str]]) -> List[str]:
        """Perform analysis enhanced by memory of previous successful patterns"""
        
        # Find similar successful prompts in memory
        similar_memories = self._find_similar_memories(prompt)
        
        if not similar_memories:
            logger.info("No similar memories found, using standard refined analysis")
            return self._perform_refined_analysis(prompt, previous_results)
        
        logger.info(f"🧠 Found {len(similar_memories)} similar memories, enhancing analysis")
        
        # Create memory-enhanced prompt
        memory_context = self._build_memory_context(similar_memories)
        
        system_prompt = f"""You are an expert video editor with access to successful analysis patterns from similar requests.

MEMORY CONTEXT - Similar successful analyses:
{memory_context}

Use these patterns to guide your analysis but adapt them to the current request. Focus on what worked well in similar cases."""

        user_prompt = f"""Based on the memory context of similar successful analyses, provide targeted effects for:

"{prompt}"

Learn from the patterns in the memory context and provide 3-5 specific, actionable effects."""

        return self._execute_llm_analysis_with_retry(
            system_prompt, user_prompt, "memory_enhanced", max_retries=2
        )
    
    def _find_similar_memories(self, prompt: str) -> List[Dict]:
        """Find similar successful analyses in memory"""
        
        current_pattern = self._extract_prompt_pattern(prompt)
        similar_memories = []
        
        try:
            for memory in self.analysis_memory['successful_prompts']:
                similarity_score = self._calculate_pattern_similarity(
                    current_pattern, memory['prompt_pattern']
                )
                
                if similarity_score > 0.3:  # Threshold for similarity
                    similar_memories.append({
                        'memory': memory,
                        'similarity': similarity_score
                    })
            
            # Sort by similarity and return top matches
            similar_memories.sort(key=lambda x: x['similarity'], reverse=True)
            return similar_memories[:3]  # Top 3 similar memories
            
        except Exception as e:
            logger.warning(f"Failed to find similar memories: {e}")
            return []
    
    def _calculate_pattern_similarity(self, pattern1: Dict, pattern2: Dict) -> float:
        """Calculate similarity between two prompt patterns"""
        
        try:
            similarity = 0.0
            
            # Keyword overlap
            keywords1 = set(pattern1.get('keywords', []))
            keywords2 = set(pattern2.get('keywords', []))
            if keywords1 and keywords2:
                keyword_sim = len(keywords1.intersection(keywords2)) / len(keywords1.union(keywords2))
                similarity += keyword_sim * 0.4
            
            # Style indicators overlap
            styles1 = set(pattern1.get('style_indicators', []))
            styles2 = set(pattern2.get('style_indicators', []))
            if styles1 and styles2:
                style_sim = len(styles1.intersection(styles2)) / len(styles1.union(styles2))
                similarity += style_sim * 0.3
            
            # Technical terms overlap
            tech1 = set(pattern1.get('technical_terms', []))
            tech2 = set(pattern2.get('technical_terms', []))
            if tech1 and tech2:
                tech_sim = len(tech1.intersection(tech2)) / len(tech1.union(tech2))
                similarity += tech_sim * 0.3
            
            return similarity
            
        except Exception as e:
            logger.warning(f"Failed to calculate similarity: {e}")
            return 0.0
    
    def _build_memory_context(self, similar_memories: List[Dict]) -> str:
        """Build context string from similar memories"""
        
        context_parts = []
        
        for i, memory_data in enumerate(similar_memories, 1):
            memory = memory_data['memory']
            similarity = memory_data['similarity']
            
            context_part = f"""
Example {i} (similarity: {similarity:.2f}):
Prompt: "{memory['prompt']}"
Successful Effects: {', '.join(memory['effects'][:3])}
"""
            context_parts.append(context_part)
        
        return '\n'.join(context_parts)
    
    def _perform_structured_analysis(self, prompt: str) -> List[str]:
        """Perform structured analysis with explicit JSON-like formatting"""
        
        system_prompt = """You are a professional video editing AI assistant. Your task is to analyze video editing requests and extract specific effects, treatments, and enhancements needed.

Always respond in this exact JSON-like format:
{
  "explicit_effects": ["effect1", "effect2"],
  "style_enhancements": ["style1", "style2"], 
  "technical_requirements": ["requirement1", "requirement2"],
  "creative_opportunities": ["opportunity1", "opportunity2"]
}

Be specific and actionable. Each item should be a clear, implementable effect or technique."""

        user_prompt = f"""Analyze this video editing request and extract all effects and requirements:

"{prompt}"

Provide your analysis in the specified JSON format. Focus on practical, implementable effects."""

        return self._execute_llm_analysis_with_retry(
            system_prompt, user_prompt, "structured", max_retries=3
        )
    
    def _perform_creative_analysis(self, prompt: str) -> List[str]:
        """Perform creative analysis for artistic and stylistic opportunities"""
        
        system_prompt = """You are a creative director and visual effects artist. Analyze video editing requests to identify creative opportunities and artistic enhancements that would elevate the final result.

Focus on:
- Artistic style opportunities
- Creative transitions and effects
- Mood and atmosphere enhancements
- Innovative techniques
- Visual storytelling elements

List your suggestions as specific, implementable effects (one per line, starting with '-')."""

        user_prompt = f"""What creative opportunities do you see in this video editing request?

"{prompt}"

Suggest specific effects and techniques that would make this video more engaging and visually compelling."""

        return self._execute_llm_analysis_with_retry(
            system_prompt, user_prompt, "creative", max_retries=2
        )
    
    def _perform_technical_analysis(self, prompt: str) -> List[str]:
        """Perform technical analysis for professional video editing requirements"""
        
        system_prompt = """You are a technical video editor specializing in professional post-production workflows. Analyze requests to identify technical requirements, color grading needs, and professional techniques.

Focus on:
- Color correction and grading requirements
- Compositing and masking needs
- Motion tracking and stabilization
- Audio synchronization requirements
- Format and delivery specifications

List technical requirements as specific, actionable items (one per line, starting with '-')."""

        user_prompt = f"""What technical video editing requirements do you identify in this request?

"{prompt}"

Focus on professional techniques, color work, compositing needs, and technical implementation details."""

        return self._execute_llm_analysis_with_retry(
            system_prompt, user_prompt, "technical", max_retries=2
        )
    
    def _execute_llm_analysis_with_retry(self, system_prompt: str, user_prompt: str, 
                                       analysis_type: str, max_retries: int = 3) -> List[str]:
        """Execute LLM analysis with retry mechanism and error handling"""
        
        for attempt in range(max_retries):
            try:
                # Format conversation
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                formatted_prompt = self._format_conversation_for_inference(messages)
                
                # Tokenize with error handling
                try:
                    inputs = self.tokenizer(
                        formatted_prompt,
                        return_tensors='pt',
                        max_length=min(2048, self.tokenizer.model_max_length or 2048),
                        truncation=True,
                        padding=True
                    )
                except Exception as e:
                    logger.warning(f"Tokenization failed for {analysis_type} analysis: {e}")
                    continue
                
                # Move to appropriate device
                try:
                    if torch.cuda.is_available() and hasattr(self.language_model, 'device') and self.language_model.device != torch.device('cpu'):
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                except Exception as e:
                    logger.warning(f"Device placement failed: {e}")
                
                # Generate with sophisticated parameters
                with torch.no_grad():
                    if hasattr(self.language_model, 'generate'):
                        try:
                            # Adjust generation parameters based on analysis type
                            if analysis_type == "structured":
                                temperature = 0.1  # Very low for structured output
                                top_p = 0.8
                                max_new_tokens = 300
                            elif analysis_type == "creative":
                                temperature = 0.6  # Higher for creativity
                                top_p = 0.95
                                max_new_tokens = 400
                            else:  # technical
                                temperature = 0.3  # Moderate for technical precision
                                top_p = 0.9
                                max_new_tokens = 350
                            
                            outputs = self.language_model.generate(
                                **inputs,
                                max_new_tokens=max_new_tokens,
                                temperature=temperature,
                                top_p=top_p,
                                do_sample=True,
                                num_beams=1,
                                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                                eos_token_id=self.tokenizer.eos_token_id,
                                repetition_penalty=1.1,
                                no_repeat_ngram_size=3,
                                early_stopping=True
                            )
                            
                            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                            effects = self._extract_effects_from_response(response, analysis_type)
                            
                            if effects:  # Success
                                logger.info(f"✅ {analysis_type.title()} analysis attempt {attempt + 1} succeeded")
                                return effects
                            else:
                                logger.warning(f"⚠️ {analysis_type.title()} analysis attempt {attempt + 1} produced no effects")
                                
                        except torch.cuda.OutOfMemoryError:
                            logger.warning(f"CUDA OOM in {analysis_type} analysis, trying CPU")
                            if torch.cuda.is_available():
                                inputs = {k: v.cpu() for k, v in inputs.items()}
                                continue
                        except Exception as e:
                            logger.warning(f"Generation failed in {analysis_type} analysis attempt {attempt + 1}: {e}")
                            continue
                    else:
                        logger.warning(f"Language model generation not available for {analysis_type} analysis")
                        break
                        
            except Exception as e:
                logger.warning(f"{analysis_type.title()} analysis attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying {analysis_type} analysis ({attempt + 2}/{max_retries})")
                continue
        
        logger.warning(f"❌ All {max_retries} attempts for {analysis_type} analysis failed")
        return []
    
    def _perform_ensemble_consensus_analysis(self, prompt: str) -> List[str]:
        """Perform analysis using multiple models and combine results via consensus"""
        
        if not self.ensemble_models:
            logger.warning("No ensemble models available for consensus")
            return []
        
        logger.info(f"🤖 Running ensemble consensus with {len(self.ensemble_models)} models")
        
        # Collect results from all available models
        all_model_results = []
        
        # Sort models by success rate and priority
        sorted_models = sorted(
            self.ensemble_models.items(),
            key=lambda x: (x[1]['success_rate'], -x[1]['priority']),
            reverse=True
        )
        
        for model_name, model_info in sorted_models[:3]:  # Use top 3 models
            try:
                model_results = self._run_analysis_on_model(
                    prompt, model_name, model_info
                )
                
                if model_results:
                    all_model_results.append({
                        'model': model_name,
                        'results': model_results,
                        'confidence': model_info['success_rate']
                    })
                    logger.info(f"✅ {model_name} contributed {len(model_results)} effects")
                    
                    # Update success stats
                    model_info['successful_attempts'] += 1
                else:
                    logger.warning(f"⚠️ {model_name} produced no results")
                
                model_info['total_attempts'] += 1
                
                # Update success rate
                if model_info['total_attempts'] > 0:
                    model_info['success_rate'] = (
                        model_info['successful_attempts'] / model_info['total_attempts']
                    )
                    
            except Exception as e:
                logger.warning(f"❌ {model_name} failed: {e}")
                model_info['total_attempts'] += 1
                if model_info['total_attempts'] > 0:
                    model_info['success_rate'] = (
                        model_info['successful_attempts'] / model_info['total_attempts']
                    )
        
        # Generate consensus from all model results
        if all_model_results:
            consensus_effects = self._generate_ensemble_consensus(all_model_results, prompt)
            logger.info(f"🎯 Ensemble consensus generated {len(consensus_effects)} effects")
            return consensus_effects
        
        return []
    
    def _run_analysis_on_model(self, prompt: str, model_name: str, model_info: Dict) -> List[str]:
        """Run analysis on a specific ensemble model"""
        
        try:
            model = model_info['model']
            tokenizer = self.ensemble_tokenizers[model_name]
            
            # Create a focused prompt for ensemble analysis
            analysis_prompt = f"""Analyze this video editing request and identify the key effects needed:

"{prompt}"

List the 3 most important effects or treatments, one per line starting with '-'."""

            # Tokenize
            inputs = tokenizer(
                analysis_prompt,
                return_tensors='pt',
                max_length=512,  # Smaller for ensemble efficiency
                truncation=True,
                padding=True
            )
            
            # Move to device if needed
            if torch.cuda.is_available() and hasattr(model, 'device') and model.device != torch.device('cpu'):
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                if hasattr(model, 'generate') and model_info['type'] == 'dialog':
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=150,
                        temperature=0.4,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.1
                    )
                    
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Remove original prompt from response
                    if analysis_prompt in response:
                        response = response.replace(analysis_prompt, "").strip()
                    
                    # Extract effects
                    effects = self._extract_effects_from_ensemble_response(response, model_name)
                    return effects
                    
                else:
                    # For encoder models, use embeddings to find similar concepts
                    outputs = model(**inputs)
                    # This is a simplified approach for encoder models
                    return [f"[{model_name}] Context-based effect analysis"]
                    
        except Exception as e:
            logger.warning(f"Model {model_name} analysis failed: {e}")
            return []
    
    def _extract_effects_from_ensemble_response(self, response: str, model_name: str) -> List[str]:
        """Extract effects from ensemble model response"""
        effects = []
        
        try:
            import re
            
            # Look for list items
            list_items = re.findall(r'[-*•]\s*([^\n]+)', response)
            for item in list_items:
                item = item.strip()
                if len(item) > 3 and len(item) < 100:
                    effects.append(f"[{model_name}] {item}")
            
            # If no list items, look for sentences with effect keywords
            if not effects:
                sentences = response.split('.')
                for sentence in sentences:
                    sentence = sentence.strip()
                    if any(keyword in sentence.lower() for keyword in ['effect', 'filter', 'style', 'color', 'transition']):
                        if len(sentence) > 5 and len(sentence) < 150:
                            effects.append(f"[{model_name}] {sentence}")
            
        except Exception as e:
            logger.warning(f"Failed to extract from {model_name} response: {e}")
        
        return effects[:3]  # Limit per model
    
    def _generate_ensemble_consensus(self, model_results: List[Dict], prompt: str) -> List[str]:
        """Generate consensus effects from multiple model results"""
        
        # Collect all effects with their sources
        effect_votes = {}
        
        for result_data in model_results:
            model_name = result_data['model']
            effects = result_data['results']
            confidence = result_data['confidence']
            
            for effect in effects:
                # Extract core effect (remove model prefix)
                if ']' in effect:
                    core_effect = effect.split(']', 1)[-1].strip().lower()
                else:
                    core_effect = effect.lower()
                
                # Vote with confidence weighting
                if core_effect not in effect_votes:
                    effect_votes[core_effect] = {
                        'votes': 0,
                        'models': [],
                        'original_text': effect,
                        'total_confidence': 0
                    }
                
                effect_votes[core_effect]['votes'] += 1
                effect_votes[core_effect]['models'].append(model_name)
                effect_votes[core_effect]['total_confidence'] += confidence
        
        # Rank effects by consensus strength
        consensus_effects = []
        
        for core_effect, vote_data in effect_votes.items():
            # Calculate consensus score
            consensus_score = (
                vote_data['votes'] * 2 +  # Number of models agreeing
                vote_data['total_confidence']  # Sum of model confidence
            )
            
            consensus_effects.append({
                'effect': vote_data['original_text'],
                'score': consensus_score,
                'votes': vote_data['votes'],
                'models': vote_data['models']
            })
        
        # Sort by consensus score
        consensus_effects.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top consensus effects
        final_effects = []
        for effect_data in consensus_effects[:8]:  # Top 8 consensus effects
            effect_text = effect_data['effect']
            
            # Add consensus information
            if effect_data['votes'] > 1:
                model_list = ', '.join(effect_data['models'][:2])  # Show first 2 models
                effect_text = f"[Ensemble Consensus] {effect_text} (agreed by {effect_data['votes']} models: {model_list})"
            else:
                effect_text = f"[Ensemble] {effect_text}"
            
            final_effects.append(effect_text)
        
        return final_effects
    
    def _extract_effects_from_response(self, response: str, analysis_type: str) -> List[str]:
        """Extract effects from LLM response based on analysis type"""
        effects = []
        
        try:
            # Remove the original prompt from response
            if "Assistant:" in response:
                analysis = response.split("Assistant:")[-1].strip()
            elif "User:" in response:
                # Find the last user message and take everything after it
                parts = response.split("User:")
                if len(parts) > 1:
                    analysis = parts[-1].split("System:")[-1].strip()
                else:
                    analysis = response
            else:
                analysis = response
            
            if analysis_type == "structured":
                effects = self._extract_from_structured_response(analysis)
            elif analysis_type == "creative":
                effects = self._extract_from_creative_response(analysis)
            elif analysis_type == "technical":
                effects = self._extract_from_technical_response(analysis)
            
            # Clean and validate effects
            effects = [effect.strip() for effect in effects if effect.strip()]
            effects = [effect for effect in effects if len(effect) > 3 and len(effect) < 200]
            
        except Exception as e:
            logger.warning(f"Failed to extract effects from {analysis_type} response: {e}")
        
        return effects
    
    def _extract_from_structured_response(self, response: str) -> List[str]:
        """Extract effects from structured JSON-like response"""
        effects = []
        
        try:
            import re
            import json
            
            # Try to parse as JSON first
            try:
                # Look for JSON-like structure
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    data = json.loads(json_str)
                    
                    # Extract from all categories
                    for category in ['explicit_effects', 'style_enhancements', 'technical_requirements', 'creative_opportunities']:
                        if category in data and isinstance(data[category], list):
                            for effect in data[category]:
                                if isinstance(effect, str):
                                    effects.append(f"[{category.replace('_', ' ').title()}] {effect}")
            except json.JSONDecodeError:
                pass
            
            # Fallback: extract from structured text
            if not effects:
                # Look for quoted items
                quoted_items = re.findall(r'"([^"]+)"', response)
                effects.extend([f"[Structured] {item}" for item in quoted_items])
                
                # Look for list items
                list_items = re.findall(r'[-*•]\s*([^\n]+)', response)
                effects.extend([f"[Structured] {item}" for item in list_items])
                
        except Exception as e:
            logger.warning(f"Failed to extract from structured response: {e}")
        
        return effects
    
    def _extract_from_creative_response(self, response: str) -> List[str]:
        """Extract effects from creative analysis response"""
        effects = []
        
        try:
            import re
            
            # Look for list items (-, *, •)
            list_items = re.findall(r'[-*•]\s*([^\n]+)', response)
            effects.extend([f"[Creative] {item.strip()}" for item in list_items])
            
            # Look for numbered items
            numbered_items = re.findall(r'\d+\.\s*([^\n]+)', response)
            effects.extend([f"[Creative] {item.strip()}" for item in numbered_items])
            
            # Look for effect-related sentences
            effect_sentences = re.findall(r'[^.!?]*(?:effect|style|transition|treatment|enhancement)[^.!?]*[.!?]', response.lower())
            for sentence in effect_sentences:
                sentence = sentence.strip()
                if len(sentence) > 10:
                    effects.append(f"[Creative] {sentence}")
            
        except Exception as e:
            logger.warning(f"Failed to extract from creative response: {e}")
        
        return effects
    
    def _extract_from_technical_response(self, response: str) -> List[str]:
        """Extract effects from technical analysis response"""
        effects = []
        
        try:
            import re
            
            # Look for technical terms and requirements
            technical_patterns = [
                r'color\s+(?:grading|correction|grade)[^.!?]*[.!?]',
                r'compositing[^.!?]*[.!?]',
                r'motion\s+tracking[^.!?]*[.!?]',
                r'stabilization[^.!?]*[.!?]',
                r'masking[^.!?]*[.!?]'
            ]
            
            for pattern in technical_patterns:
                matches = re.findall(pattern, response.lower())
                for match in matches:
                    effects.append(f"[Technical] {match.strip()}")
            
            # Look for list items
            list_items = re.findall(r'[-*•]\s*([^\n]+)', response)
            effects.extend([f"[Technical] {item.strip()}" for item in list_items])
            
            # Look for numbered items
            numbered_items = re.findall(r'\d+\.\s*([^\n]+)', response)
            effects.extend([f"[Technical] {item.strip()}" for item in numbered_items])
            
        except Exception as e:
            logger.warning(f"Failed to extract from technical response: {e}")
        
        return effects
    
    def _combine_and_validate_analysis_results(self, all_results: List[List[str]], prompt: str) -> List[str]:
        """Combine results from multiple analysis approaches and validate"""
        
        # Flatten all results
        all_effects = []
        for result_list in all_results:
            all_effects.extend(result_list)
        
        if not all_effects:
            return []
        
        # Remove duplicates while preserving categories
        unique_effects = []
        seen_effects = set()
        
        for effect in all_effects:
            # Extract the core effect (without category prefix)
            if ']' in effect:
                core_effect = effect.split(']', 1)[-1].strip().lower()
            else:
                core_effect = effect.strip().lower()
            
            if core_effect not in seen_effects and len(core_effect) > 3:
                seen_effects.add(core_effect)
                unique_effects.append(effect)
        
        # Rank by relevance to original prompt
        ranked_effects = self._rank_effects_by_relevance(unique_effects, prompt)
        
        # Limit to most relevant effects
        return ranked_effects[:12]
    
    def _rank_effects_by_relevance(self, effects: List[str], prompt: str) -> List[str]:
        """Rank effects by relevance to the original prompt"""
        
        prompt_words = set(prompt.lower().split())
        ranked_effects = []
        
        # Calculate relevance score for each effect
        effect_scores = []
        
        for effect in effects:
            # Extract core effect text
            if ']' in effect:
                core_text = effect.split(']', 1)[-1].strip()
            else:
                core_text = effect
            
            effect_words = set(core_text.lower().split())
            
            # Calculate word overlap score
            overlap_score = len(prompt_words.intersection(effect_words))
            
            # Bonus for category types
            category_bonus = 0
            if '[explicit' in effect.lower() or '[structured' in effect.lower():
                category_bonus = 3  # Highest priority for explicit effects
            elif '[technical' in effect.lower():
                category_bonus = 2
            elif '[creative' in effect.lower():
                category_bonus = 1
            
            total_score = overlap_score + category_bonus
            effect_scores.append((effect, total_score))
        
        # Sort by score (descending) and return effects
        effect_scores.sort(key=lambda x: x[1], reverse=True)
        return [effect for effect, _ in effect_scores]
    
    def _validate_analysis_quality(self, effects: List[str], prompt: str) -> bool:
        """Advanced validation of LLM analysis quality with multiple quality metrics"""
        
        if not effects:
            logger.warning("❌ Validation failed: No effects generated")
            return False
        
        logger.info(f"🔍 Validating analysis quality for {len(effects)} effects")
        
        # Multi-dimensional quality assessment
        quality_metrics = {
            'category_diversity': 0,
            'prompt_relevance': 0, 
            'effect_specificity': 0,
            'technical_depth': 0,
            'actionability': 0,
            'uniqueness': 0
        }
        
        # Metric 1: Category diversity (multiple analysis types)
        categories = set()
        for effect in effects:
            if '[' in effect and ']' in effect:
                category = effect.split('[')[1].split(']')[0].lower()
                categories.add(category)
        
        if len(categories) >= 3:
            quality_metrics['category_diversity'] = 3
        elif len(categories) >= 2:
            quality_metrics['category_diversity'] = 2
        elif len(categories) >= 1:
            quality_metrics['category_diversity'] = 1
        
        # Metric 2: Enhanced prompt relevance analysis
        prompt_words = set(prompt.lower().split())
        prompt_concepts = self._extract_concepts_from_text(prompt.lower())
        
        relevant_effects = 0
        high_relevance_effects = 0
        
        for effect in effects:
            effect_words = set(effect.lower().split())
            effect_concepts = self._extract_concepts_from_text(effect.lower())
            
            # Word-level relevance
            word_overlap = len(prompt_words.intersection(effect_words))
            # Concept-level relevance
            concept_overlap = len(prompt_concepts.intersection(effect_concepts))
            
            if word_overlap > 0 or concept_overlap > 0:
                relevant_effects += 1
                if word_overlap >= 2 or concept_overlap >= 1:
                    high_relevance_effects += 1
        
        relevance_ratio = relevant_effects / len(effects)
        high_relevance_ratio = high_relevance_effects / len(effects)
        
        if high_relevance_ratio >= 0.4:
            quality_metrics['prompt_relevance'] = 3
        elif relevance_ratio >= 0.6:
            quality_metrics['prompt_relevance'] = 2
        elif relevance_ratio >= 0.3:
            quality_metrics['prompt_relevance'] = 1
        
        # Metric 3: Effect specificity (detailed vs vague)
        specific_effects = 0
        for effect in effects:
            # Remove category prefix
            core_effect = effect.split(']', 1)[-1].strip() if ']' in effect else effect
            specificity_score = self._calculate_effect_specificity(core_effect)
            if specificity_score >= 0.7:
                specific_effects += 1
        
        specificity_ratio = specific_effects / len(effects)
        if specificity_ratio >= 0.6:
            quality_metrics['effect_specificity'] = 3
        elif specificity_ratio >= 0.4:
            quality_metrics['effect_specificity'] = 2
        elif specificity_ratio >= 0.2:
            quality_metrics['effect_specificity'] = 1
        
        # Metric 4: Technical depth assessment
        technical_terms = [
            'color grading', 'color correction', 'compositing', 'motion tracking',
            'stabilization', 'masking', 'keying', 'rotoscoping', 'temporal',
            'spatial', 'luminance', 'chrominance', 'gamma', 'contrast'
        ]
        
        technical_effects = 0
        for effect in effects:
            effect_lower = effect.lower()
            if any(term in effect_lower for term in technical_terms):
                technical_effects += 1
        
        if technical_effects >= len(effects) * 0.4:
            quality_metrics['technical_depth'] = 3
        elif technical_effects >= len(effects) * 0.2:
            quality_metrics['technical_depth'] = 2
        elif technical_effects > 0:
            quality_metrics['technical_depth'] = 1
        
        # Metric 5: Actionability (can be implemented)
        actionable_effects = 0
        for effect in effects:
            if self._is_effect_actionable(effect):
                actionable_effects += 1
        
        actionability_ratio = actionable_effects / len(effects)
        if actionability_ratio >= 0.8:
            quality_metrics['actionability'] = 3
        elif actionability_ratio >= 0.6:
            quality_metrics['actionability'] = 2
        elif actionability_ratio >= 0.4:
            quality_metrics['actionability'] = 1
        
        # Metric 6: Uniqueness (avoid repetition)
        unique_concepts = set()
        for effect in effects:
            concepts = self._extract_concepts_from_text(effect.lower())
            unique_concepts.update(concepts)
        
        uniqueness_ratio = len(unique_concepts) / max(len(effects), 1)
        if uniqueness_ratio >= 0.8:
            quality_metrics['uniqueness'] = 3
        elif uniqueness_ratio >= 0.6:
            quality_metrics['uniqueness'] = 2
        elif uniqueness_ratio >= 0.4:
            quality_metrics['uniqueness'] = 1
        
        # Calculate overall quality score
        total_score = sum(quality_metrics.values())
        max_possible_score = len(quality_metrics) * 3
        quality_percentage = total_score / max_possible_score
        
        # Log detailed metrics
        logger.info(f"📊 Quality Metrics:")
        for metric, score in quality_metrics.items():
            logger.info(f"  {metric.replace('_', ' ').title()}: {score}/3")
        logger.info(f"  Overall Quality: {quality_percentage:.2%} ({total_score}/{max_possible_score})")
        
        # Advanced quality thresholds
        if quality_percentage >= 0.7:
            logger.info("✅ High quality analysis - exceeds standards")
            return True
        elif quality_percentage >= 0.5:
            logger.info("✅ Good quality analysis - meets standards")
            return True
        elif quality_percentage >= 0.35:
            logger.warning("⚠️ Moderate quality analysis - borderline")
            # Additional check: if ensemble consensus is strong, accept moderate quality
            ensemble_effects = [e for e in effects if 'ensemble' in e.lower()]
            if len(ensemble_effects) >= len(effects) * 0.3:
                logger.info("✅ Accepting due to strong ensemble consensus")
                return True
            return False
        else:
            logger.warning("❌ Low quality analysis - below standards")
            return False
    
    def _extract_concepts_from_text(self, text: str) -> set:
        """Extract key concepts from text for relevance analysis"""
        
        import re
        concepts = set()
        
        # Video editing concepts
        editing_concepts = re.findall(r'\b(?:edit|cut|trim|splice|montage|sequence)\w*\b', text)
        concepts.update(editing_concepts)
        
        # Visual effects concepts
        vfx_concepts = re.findall(r'\b(?:effect|filter|transition|fade|dissolve|wipe)\w*\b', text)
        concepts.update(vfx_concepts)
        
        # Color and grading concepts
        color_concepts = re.findall(r'\b(?:color|colour|grade|grading|correct|correction|lut|gamma)\w*\b', text)
        concepts.update(color_concepts)
        
        # Motion and tracking concepts
        motion_concepts = re.findall(r'\b(?:motion|track|tracking|stabiliz|shake|smooth)\w*\b', text)
        concepts.update(motion_concepts)
        
        # Style and aesthetic concepts
        style_concepts = re.findall(r'\b(?:style|aesthetic|cinematic|vintage|modern|artistic|dramatic)\w*\b', text)
        concepts.update(style_concepts)
        
        return concepts
    
    def _calculate_effect_specificity(self, effect_text: str) -> float:
        """Calculate how specific/detailed an effect description is"""
        
        specificity_score = 0.0
        
        # Length bonus (more detailed descriptions tend to be longer)
        if len(effect_text) > 50:
            specificity_score += 0.3
        elif len(effect_text) > 20:
            specificity_score += 0.2
        
        # Technical term bonus
        technical_indicators = [
            'color grading', 'color correction', 'motion tracking', 'compositing',
            'masking', 'keying', 'gamma', 'contrast', 'saturation', 'luminance',
            'temporal', 'spatial', 'frequency', 'amplitude'
        ]
        
        tech_count = sum(1 for term in technical_indicators if term in effect_text.lower())
        specificity_score += min(tech_count * 0.1, 0.3)
        
        # Numerical parameters bonus
        import re
        numbers = re.findall(r'\d+(?:\.\d+)?', effect_text)
        if numbers:
            specificity_score += 0.2
        
        # Specific tool/software mentions
        tools = ['davinci', 'premiere', 'after effects', 'ffmpeg', 'opencv']
        if any(tool in effect_text.lower() for tool in tools):
            specificity_score += 0.2
        
        return min(specificity_score, 1.0)
    
    def _is_effect_actionable(self, effect_text: str) -> bool:
        """Determine if an effect description is actionable/implementable"""
        
        # Remove category prefix
        core_effect = effect_text.split(']', 1)[-1].strip() if ']' in effect_text else effect_text
        core_effect_lower = core_effect.lower()
        
        # Too vague indicators
        vague_indicators = [
            'make it better', 'improve quality', 'enhance video', 'add effects',
            'make it look good', 'professional look', 'nice effect'
        ]
        
        if any(vague in core_effect_lower for vague in vague_indicators):
            return False
        
        # Too short or too generic
        if len(core_effect.strip()) < 5:
            return False
        
        # Actionable indicators
        actionable_indicators = [
            'color grade', 'color correct', 'stabilize', 'track motion', 'add transition',
            'apply filter', 'adjust', 'composite', 'mask', 'key out', 'fade in',
            'fade out', 'crop', 'scale', 'rotate', 'blur', 'sharpen'
        ]
        
        if any(action in core_effect_lower for action in actionable_indicators):
            return True
        
        # Has specific parameters or values
        import re
        if re.search(r'\d+(?:\.\d+)?(?:px|%|sec|ms|db|hz)', core_effect_lower):
            return True
        
        # Default: moderately actionable if it's reasonably specific
        return len(core_effect.strip()) >= 10
    
    def _perform_refined_analysis(self, prompt: str, previous_results: List[List[str]]) -> List[str]:
        """Perform refined analysis using insights from previous attempts"""
        
        # Analyze what worked in previous attempts
        successful_categories = set()
        for result_list in previous_results:
            for effect in result_list:
                if '[' in effect and ']' in effect:
                    category = effect.split('[')[1].split(']')[0].lower()
                    successful_categories.add(category)
        
        # Create a focused refined prompt
        system_prompt = f"""You are an expert video editor. Previous analysis identified categories: {', '.join(successful_categories)}. 

Provide a concise, focused analysis that builds on these insights. Be specific and actionable."""

        user_prompt = f"""Building on previous analysis, provide the most important and actionable video editing effects for:

"{prompt}"

Focus on the 3-5 most critical effects that would have the biggest impact. Be specific about implementation."""

        return self._execute_llm_analysis_with_retry(
            system_prompt, user_prompt, "refined", max_retries=2
        )
    
    def _enhanced_contextual_fallback(self, prompt: str) -> List[str]:
        """Enhanced fallback that uses contextual understanding instead of simple regex"""
        
        logger.info("🔄 Using enhanced contextual fallback analysis")
        
        # Use the existing sophisticated fallback but with better logging
        fallback_effects = self._sophisticated_fallback_analysis(prompt)
        
        if fallback_effects:
            logger.info(f"📝 Contextual fallback identified {len(fallback_effects)} effects")
            # Mark as fallback for transparency
            enhanced_effects = [f"[Fallback Analysis] {effect}" for effect in fallback_effects]
            return enhanced_effects
        else:
            logger.warning("❌ Even enhanced fallback failed to identify effects")
            return [f"[Fallback Analysis] Basic video editing with standard transitions and effects"]
    
    def _format_conversation_for_inference(self, messages: List[Dict[str, str]]) -> str:
        """Format conversation messages for model inference"""
        formatted = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                formatted += f"System: {content}\n\n"
            elif role == "user":
                formatted += f"User: {content}\n\nAssistant: "
        return formatted
    
    def _extract_effects_from_advanced_analysis(self, response: str, analysis_type: str) -> List[str]:
        """Extract effects from sophisticated LLM analysis response"""
        effects = []
        
        try:
            # Remove the original prompt from response
            if "Assistant:" in response:
                analysis = response.split("Assistant:")[-1].strip()
            else:
                analysis = response
            
            # Look for structured sections
            sections = {
                'explicit': ['explicit effects', 'direct requests', 'mentioned effects', '1.'],
                'implied': ['implied', 'creative opportunities', 'suggestions', '2.'], 
                'technical': ['technical', 'requirements', 'technical aspects', '3.'],
                'style': ['style', 'aesthetic', 'mood', '4.'],
                'advanced': ['advanced', 'professional', 'techniques', '5.']
            }
            
            analysis_lower = analysis.lower()
            
            # Extract from structured sections
            for section_type, keywords in sections.items():
                for keyword in keywords:
                    if keyword in analysis_lower:
                        # Find the section content
                        start_idx = analysis_lower.find(keyword)
                        
                        # Find next section or end
                        next_sections = []
                        for other_type, other_keywords in sections.items():
                            if other_type != section_type:
                                for other_keyword in other_keywords:
                                    next_idx = analysis_lower.find(other_keyword, start_idx + len(keyword))
                                    if next_idx > start_idx:
                                        next_sections.append(next_idx)
                        
                        end_idx = min(next_sections) if next_sections else len(analysis)
                        section_content = analysis[start_idx:end_idx]
                        
                        # Extract effects from section
                        section_effects = self._parse_section_for_effects(section_content, section_type)
                        effects.extend(section_effects)
                        break  # Only process first match per section
            
            # If no structured sections found, use general parsing
            if not effects:
                effects = self._parse_general_effects(analysis)
            
            # Add analysis type prefix for context
            effects = [f"[{analysis_type}] {effect}" for effect in effects if effect]
            
        except Exception as e:
            logger.warning(f"Failed to extract effects from {analysis_type} analysis: {e}")
        
        return effects
    
    def _parse_section_for_effects(self, section_content: str, section_type: str) -> List[str]:
        """Parse a specific section for effect descriptions"""
        effects = []
        
        # Look for bullet points, numbered lists, and comma-separated items
        import re
        
        # Bullet points and numbered lists
        list_items = re.findall(r'[-*•]\s*([^\n]+)', section_content)
        list_items.extend(re.findall(r'\d+\.\s*([^\n]+)', section_content))
        
        # Quoted items
        quoted_items = re.findall(r'"([^"]+)"', section_content)
        quoted_items.extend(re.findall(r"'([^']+)'", section_content))
        
        # Technical terms and effect names
        if section_type in ['technical', 'advanced']:
            tech_patterns = re.findall(r'\b(color\s+\w+|motion\s+\w+|\w+\s+effect|\w+\s+filter|\w+\s+transition)\b', section_content.lower())
            effects.extend(tech_patterns)
        
        # Combine all found items
        all_items = list_items + quoted_items
        
        # Clean and filter
        for item in all_items:
            item = item.strip().strip('.,!?')
            if len(item) > 5 and len(item) < 100:  # Reasonable length
                effects.append(item)
        
        return effects[:5]  # Limit per section
    
    def _parse_general_effects(self, analysis: str) -> List[str]:
        """General parsing for effect descriptions when no structure is found"""
        effects = []
        
        # Look for effect-related sentences
        import re
        
        effect_sentences = re.findall(r'[^.!?]*(?:effect|filter|transition|style|treatment|technique)[^.!?]*[.!?]', analysis.lower())
        
        for sentence in effect_sentences:
            # Extract the key effect description
            sentence = sentence.strip()
            if len(sentence) > 10 and len(sentence) < 150:
                effects.append(sentence)
        
        return effects[:3]  # Limit general parsing
    
    def _deduplicate_and_rank_effects(self, effects: List[str], original_prompt: str) -> List[str]:
        """Remove duplicates and rank effects by relevance"""
        if not effects:
            return []
        
        # Remove duplicates while preserving order
        seen = set()
        unique_effects = []
        
        for effect in effects:
            # Clean the effect string
            cleaned = effect.lower().strip()
            # Remove analysis type prefix for comparison
            if '] ' in cleaned:
                cleaned = cleaned.split('] ', 1)[-1]
            
            if cleaned not in seen and len(cleaned) > 3:
                seen.add(cleaned)
                unique_effects.append(effect)
        
        # Rank by relevance to original prompt
        prompt_lower = original_prompt.lower()
        ranked_effects = []
        
        # High relevance: directly mentioned in prompt
        for effect in unique_effects:
            effect_words = effect.lower().split()
            if any(word in prompt_lower for word in effect_words if len(word) > 3):
                ranked_effects.append(effect)
        
        # Medium relevance: related concepts
        for effect in unique_effects:
            if effect not in ranked_effects:
                ranked_effects.append(effect)
        
        return ranked_effects[:8]  # Return top 8 effects
    
    def _parse_llm_analysis(self, response: str) -> List[str]:
        """Parse structured LLM response to extract custom effects"""
        custom_effects = []
        
        try:
            # Look for CUSTOM_EFFECTS section
            if "CUSTOM_EFFECTS:" in response:
                effects_section = response.split("CUSTOM_EFFECTS:")[1].split("STYLE_REQUESTS:")[0]
                # Extract list items
                import re
                effects = re.findall(r'\[([^\]]+)\]', effects_section)
                if effects:
                    custom_effects.extend([effect.strip() for effect in effects[0].split(',')])
            
            # Look for STYLE_REQUESTS section  
            if "STYLE_REQUESTS:" in response:
                style_section = response.split("STYLE_REQUESTS:")[1].split("TECHNICAL_REQUIREMENTS:")[0] if "TECHNICAL_REQUIREMENTS:" in response else response.split("STYLE_REQUESTS:")[1]
                import re
                styles = re.findall(r'\[([^\]]+)\]', style_section)
                if styles:
                    custom_effects.extend([f"{style.strip()} style effect" for style in styles[0].split(',')])
                    
        except Exception as e:
            logger.warning(f"Failed to parse LLM analysis: {e}")
            
        # Clean up effects list
        custom_effects = [effect.strip().strip('"\'') for effect in custom_effects if effect.strip()]
        custom_effects = [effect for effect in custom_effects if len(effect) > 3]  # Remove very short effects
        
        return custom_effects
    
    def _sophisticated_fallback_analysis(self, prompt: str) -> List[str]:
        """
        Sophisticated fallback analysis using advanced natural language processing and context understanding.
        Uses multiple analytical approaches when LLM generation is not available.
        """
        
        custom_effects = []
        prompt_lower = prompt.lower()
        
        # Advanced contextual indicators for custom effect requests
        custom_effect_patterns = {
            'creation_requests': [
                r'create (?:a |an )?(\w+(?:\s+\w+){0,3})\s+(?:effect|filter|style)',
                r'generate (?:a |an )?(\w+(?:\s+\w+){0,3})\s+(?:effect|transition)',
                r'add (?:a |an )?(?:custom|unique|special)\s+(\w+(?:\s+\w+){0,2})',
                r'make (?:it |this )?(?:look|appear)\s+(\w+(?:\s+\w+){0,2})',
                r'apply (?:a |an )?(\w+(?:\s+\w+){0,2})\s+(?:treatment|style|look)'
            ],
            'style_descriptors': [
                r'(?:with|using|in)\s+(?:a |an )?(\w+(?:\s+\w+){0,2})\s+style',
                r'(?:make|give|add)\s+(?:it |this )?(?:a |an )?(\w+(?:\s+\w+){0,2})\s+(?:feel|vibe|aesthetic)',
                r'(?:cinematic|artistic|stylized|creative)\s+(\w+(?:\s+\w+){0,2})'
            ],
            'technical_requests': [
                r'(?:color|colour)\s+(grade|grading|correct|correction)\s+(?:to|for|with)\s+(\w+(?:\s+\w+){0,2})',
                r'(?:composite|blend|layer|mask)\s+(?:with|using|for)\s+(\w+(?:\s+\w+){0,2})',
                r'(?:motion|camera)\s+(track|tracking|stabiliz\w+)\s+(\w+(?:\s+\w+){0,2})'
            ]
        }
        
        # Extract effects using pattern matching
        import re
        
        for category, patterns in custom_effect_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, prompt_lower)
                for match in matches:
                    if isinstance(match, tuple):
                        effect_desc = ' '.join(match).strip()
                    else:
                        effect_desc = match.strip()
                    
                    if effect_desc and len(effect_desc) > 2:
                        category_prefix = category.replace('_', ' ').title()
                        custom_effects.append(f"{category_prefix}: {effect_desc}")
        
        # Advanced semantic analysis using professional editing vocabulary
        professional_techniques = {
            'cinematography': {
                'keywords': ['cinematic', 'film', 'movie', 'shot', 'frame', 'composition'],
                'effects': ['cinematic color grading', 'film grain', 'letterboxing', 'depth of field']
            },
            'motion_graphics': {
                'keywords': ['motion', 'graphics', 'animation', 'animated', 'dynamic', 'kinetic'],
                'effects': ['animated text', 'motion tracking', 'kinetic typography', 'logo animation']
            },
            'visual_effects': {
                'keywords': ['vfx', 'effects', 'magical', 'surreal', 'fantasy', 'sci-fi'],
                'effects': ['particle effects', 'light rays', 'energy beams', 'magical glow']
            },
            'audio_visual': {
                'keywords': ['music', 'beat', 'rhythm', 'sync', 'audio'],
                'effects': ['audio visualization', 'rhythm-based cuts', 'music sync effects']
            },
            'stylistic': {
                'keywords': ['vintage', 'retro', 'modern', 'futuristic', 'artistic', 'abstract'],
                'effects': ['vintage film look', 'retro color scheme', 'modern transitions', 'artistic filters']
            }
        }
        
        # Analyze prompt for professional technique indicators
        for technique_category, technique_data in professional_techniques.items():
            keyword_matches = sum(1 for keyword in technique_data['keywords'] if keyword in prompt_lower)
            
            if keyword_matches > 0:
                # Select most relevant effects based on keyword density
                relevance_score = keyword_matches / len(technique_data['keywords'])
                num_effects = min(len(technique_data['effects']), max(1, int(relevance_score * 3)))
                
                selected_effects = technique_data['effects'][:num_effects]
                for effect in selected_effects:
                    custom_effects.append(f"Professional {technique_category.replace('_', ' ').title()}: {effect}")
        
        # Context-aware intensity and complexity analysis
        intensity_indicators = {
            'subtle': ['subtle', 'light', 'gentle', 'soft', 'mild'],
            'moderate': ['noticeable', 'clear', 'visible', 'apparent'],
            'dramatic': ['dramatic', 'bold', 'strong', 'intense', 'heavy', 'extreme']
        }
        
        detected_intensity = 'moderate'  # default
        for intensity, keywords in intensity_indicators.items():
            if any(keyword in prompt_lower for keyword in keywords):
                detected_intensity = intensity
                break
        
        # Enhance effects with intensity information
        if custom_effects:
            intensity_enhanced_effects = []
            for effect in custom_effects:
                if detected_intensity != 'moderate':
                    enhanced_effect = f"{effect} ({detected_intensity} intensity)"
                else:
                    enhanced_effect = effect
                intensity_enhanced_effects.append(enhanced_effect)
            custom_effects = intensity_enhanced_effects
        
        # Advanced content type detection for contextual effects
        content_type_indicators = {
            'social_media': ['instagram', 'tiktok', 'youtube', 'social', 'viral', 'trending'],
            'corporate': ['business', 'corporate', 'professional', 'presentation'],
            'creative': ['art', 'artistic', 'creative', 'experimental', 'unique'],
            'entertainment': ['fun', 'entertaining', 'playful', 'exciting', 'dynamic']
        }
        
        detected_content_type = None
        for content_type, keywords in content_type_indicators.items():
            if any(keyword in prompt_lower for keyword in keywords):
                detected_content_type = content_type
                break
        
        # Add content-type specific effect suggestions
        if detected_content_type:
            content_specific_effects = {
                'social_media': ['trendy transitions', 'social media optimized colors', 'engagement-focused cuts'],
                'corporate': ['professional transitions', 'corporate branding effects', 'clean typography'],
                'creative': ['experimental effects', 'artistic filters', 'creative compositions'],
                'entertainment': ['dynamic transitions', 'fun effects', 'energetic pacing']
            }
            
            if detected_content_type in content_specific_effects:
                for effect in content_specific_effects[detected_content_type]:
                    custom_effects.append(f"Content-Type Enhancement: {effect}")
        
        # Remove duplicates and clean up
        unique_effects = []
        seen_effects = set()
        
        for effect in custom_effects:
            effect_clean = effect.lower().strip()
            if effect_clean not in seen_effects and len(effect) > 5:
                seen_effects.add(effect_clean)
                unique_effects.append(effect)
        
        # Limit to most relevant effects
        unique_effects = unique_effects[:10]
        
        logger.info(f"📝 Sophisticated fallback analysis identified {len(unique_effects)} effects")
        for i, effect in enumerate(unique_effects[:3], 1):
            logger.info(f"  {i}. {effect}")
        
        return unique_effects
        
    def set_training_phase(self, phase: str):
        """Set current training phase for different optimization strategies"""
        self.current_phase = phase
        
        if phase == "distillation":
            # Freeze certain layers during distillation
            for param in self.language_model.parameters():
                param.requires_grad = False
        elif phase == "editing_finetuning":
            # Use LoRA for efficient fine-tuning
            self._setup_lora()
        elif phase == "self_improvement":
            # Enable all parameters for RLHF
            for param in self.parameters():
                param.requires_grad = True
                
    def _setup_lora(self):
        """Setup LoRA for parameter-efficient fine-tuning"""
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            logger.warning("PEFT not available, skipping LoRA setup")
            return
        
        lora_config = LoraConfig(
            r=self.config['training']['phase3']['lora_r'],
            lora_alpha=self.config['training']['phase3']['lora_alpha'], 
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.language_model = get_peft_model(self.language_model, lora_config)
        
    def save_checkpoint(self, path: str, epoch: int, optimizer_state: dict = None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'current_phase': self.current_phase
        }
        if optimizer_state:
            checkpoint['optimizer_state_dict'] = optimizer_state
            
        torch.save(checkpoint, path)
        
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.current_phase = checkpoint.get('current_phase', 'pretraining')
        return model
