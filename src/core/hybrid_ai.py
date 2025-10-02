"""
Hybrid Video AI - The core autonomous video editing AI system
Combines reasoning, perception, and editing capabilities in a unified model
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional, Tuple, Any
from transformers import (
    AutoTokenizer, 
    AutoModel,
    LlamaForCausalLM,
    CLIPVisionModel,
    WhisperModel
)

from ..models.multimodal_fusion import MultiModalFusionModule
from ..models.video_understanding import VideoUnderstandingModule
from ..models.editing_planner import EditingPlannerModule
from ..perception.vision_processor import VisionProcessor
from ..utils.model_downloader import ModelDownloader

# Setup logging
logger = logging.getLogger(__name__)
from ..audio.audio_processor import AudioProcessor
from ..editing.timeline_generator import TimelineGenerator


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
        
        # Auto-download models if not available
        self._ensure_models_available()
        
        # Initialize tokenizer - Use teacher model for better capabilities
        model_name = config.get('teachers', {}).get('text_model', config['model'].get('backbone', 'meta-llama/Llama-2-7b-hf'))
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
            
        # Core reasoning backbone - Use advanced teacher model
        try:
            self.language_model = LlamaForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto" if torch.cuda.is_available() else None,
                load_in_8bit=config.get('quantization', {}).get('load_in_8bit', True),  # Enable 8-bit by default for large models
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
            vision_model = config['model'].get('vision_encoder', 'openai/clip-vit-base-patch32')
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
                torch_dtype=torch.bfloat16
            )
            logger.info(f"✅ Loaded advanced audio model: {audio_model}")
        except Exception as e:
            logger.warning(f"Failed to load Whisper Large, using base model: {e}")
            audio_model = 'openai/whisper-base'
            self.audio_encoder = WhisperModel.from_pretrained(
                audio_model,
                cache_dir=config.get('model_cache_dir', 'models/cache')
            )
        
        # Initialize model downloader
        self.model_downloader = ModelDownloader(
            cache_dir=config.get('model_cache_dir', 'models/cache')
        )
        
        # Multimodal fusion layer
        self.fusion_module = MultiModalFusionModule(
            text_dim=config['model']['text_dim'],
            vision_dim=config['model']['vision_dim'], 
            audio_dim=config['model']['audio_dim'],
            fusion_dim=config['model']['fusion_dim'],
            num_heads=config['model']['num_attention_heads']
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
            fusion_dim=config['model']['fusion_dim'],
            hidden_dim=config['model']['hidden_dim']
        )
        
        # Editing planner 
        self.editing_planner = EditingPlannerModule(
            hidden_dim=config['model']['hidden_dim'],
            vocab_size=len(self.tokenizer)
        )
        
        # Specialized processors
        self.vision_processor = VisionProcessor(config)
        self.audio_processor = AudioProcessor(config)
        self.timeline_generator = TimelineGenerator(config)
        
        # Training phase tracker
        self.current_phase = "pretraining"
        
    def _ensure_models_available(self):
        """Ensure all required models are downloaded and available"""
        try:
            # Get model names from config
            backbone_model = self.config['model'].get('backbone', 'microsoft/DialoGPT-small')
            vision_model = self.config['model'].get('vision_encoder', 'openai/clip-vit-base-patch32') 
            audio_model = self.config['model'].get('audio_encoder', 'openai/whisper-tiny')
            
            # Auto-download models using the model downloader
            model_downloader = ModelDownloader(
                cache_dir=self.config.get('model_cache_dir', 'models/cache')
            )
            
            logger.info("Ensuring required models are available...")
            
            # Download backbone model
            model_downloader.download_model(backbone_model, model_type='language')
            
            # Download vision model  
            model_downloader.download_model(vision_model, model_type='vision')
            
            # Download audio model
            model_downloader.download_model(audio_model, model_type='audio')
            
            logger.info("All required models are available")
            
        except Exception as e:
            logger.warning(f"Model download failed, will attempt to load from cache: {e}")
    
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
            vision_embeddings = vision_outputs.last_hidden_state.mean(dim=1)  # Pool
            vision_embeddings = vision_embeddings.view(B, T, -1)
            outputs['vision_embeddings'] = vision_embeddings
            
        # Process audio input
        if audio_features is not None:
            # Use Whisper encoder for audio embeddings
            audio_embeddings = self.audio_encoder.encoder(audio_features).last_hidden_state
            outputs['audio_embeddings'] = audio_embeddings
            
        # Multimodal fusion
        if any(k in outputs for k in ['text_embeddings', 'vision_embeddings', 'audio_embeddings']):
            fused_embeddings = self.fusion_module(
                text_emb=outputs.get('text_embeddings'),
                vision_emb=outputs.get('vision_embeddings'), 
                audio_emb=outputs.get('audio_embeddings')
            )
            outputs['fused_embeddings'] = fused_embeddings
            
            # Video understanding
            video_understanding = self.video_understanding(fused_embeddings)
            outputs['video_understanding'] = video_understanding
            
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
    
    def autonomous_edit(self, video_path: str, prompt: str) -> str:
        """
        Fully autonomous video editing based on natural language prompt
        Enhanced with self-coding capabilities for custom effects
        
        Args:
            video_path: Path to input video
            prompt: Natural language editing instruction
            
        Returns:
            Path to edited video
        """
        logger.info(f"🎬 Starting autonomous edit: {prompt}")
        
        # Load and preprocess video
        video_data = self.vision_processor.load_video(video_path)
        audio_data = self.audio_processor.load_audio(video_path)
        
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
    
    def _analyze_prompt_for_custom_effects(self, prompt: str) -> List[str]:
        """
        Advanced LLM-powered analysis of editing prompt to identify and extract custom effect requirements.
        Uses the language model to understand nuanced effect descriptions and generate structured editing plans.
        """
        
        # Enhanced prompt analysis using the language model
        analysis_prompt = f"""
        Analyze the following video editing request and extract any custom effects, transitions, or special visual treatments that need to be created:

        USER REQUEST: "{prompt}"

        Instructions:
        1. Identify any requests for custom visual effects (glitch, distortion, particle systems, etc.)
        2. Look for unique transitions or motion graphics requests  
        3. Detect style requests (cyberpunk, vintage, neon, etc.)
        4. Extract any specific technical requirements (color grading, compositing, etc.)
        5. Return a structured list of effect descriptions

        Response format:
        CUSTOM_EFFECTS: [effect1, effect2, effect3]
        STYLE_REQUESTS: [style1, style2]
        TECHNICAL_REQUIREMENTS: [req1, req2]

        Analysis:
        """
        
        try:
            # Tokenize the analysis prompt
            inputs = self.tokenizer.encode(analysis_prompt, return_tensors='pt', truncation=True, max_length=1024)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            
            # Generate LLM analysis with controlled generation
            with torch.no_grad():
                if hasattr(self.language_model, 'generate'):
                    outputs = self.language_model.generate(
                        inputs,
                        max_new_tokens=300,
                        temperature=0.3,  # Low temperature for more focused analysis
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    
                    # Decode the response
                    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Extract custom effects from structured response
                    custom_effects = self._parse_llm_analysis(response)
                    
                    if custom_effects:
                        logger.info(f"🧠 LLM identified {len(custom_effects)} custom effects: {custom_effects}")
                        return custom_effects
                        
                else:
                    logger.warning("Language model doesn't support generation, falling back to heuristic analysis")
                    
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}, using fallback keyword analysis")
        
        # Fallback to enhanced keyword-based analysis if LLM fails
        return self._fallback_effect_analysis(prompt)
    
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
    
    def _fallback_effect_analysis(self, prompt: str) -> List[str]:
        """Enhanced fallback analysis using improved keyword detection"""
        
        custom_effect_indicators = [
            "create a", "generate a", "add a custom", "make a unique",
            "apply a special", "create custom", "generate unique",
            "unusual effect", "special transition", "unique filter",
            "dynamic effect", "animated", "procedural", "generative"
        ]
        
        custom_effects = []
        prompt_lower = prompt.lower()
        
        # Advanced effect keywords with categories
        effect_categories = {
            'glitch': ['glitch', 'corruption', 'digital noise', 'pixelation', 'datamosh'],
            'distortion': ['distortion', 'warp', 'ripple', 'wave', 'bend', 'twist'],
            'particle': ['particle', 'sparks', 'dust', 'smoke', 'fire', 'energy'],
            'style': ['cyberpunk', 'vintage', 'retro', 'neon', '80s', 'synthwave', 'vaporwave'],
            'motion': ['zoom', 'pan', 'tilt', 'rotate', 'shake', 'stabilize'],
            'color': ['color grade', 'tint', 'saturation', 'contrast', 'brightness'],
            'composite': ['green screen', 'chroma key', 'masking', 'rotoscope']
        }
        
        # Look for custom effect requests with context
        for indicator in custom_effect_indicators:
            if indicator in prompt_lower:
                start_idx = prompt_lower.find(indicator)
                end_idx = prompt_lower.find('.', start_idx)
                if end_idx == -1:
                    end_idx = len(prompt)
                
                effect_desc = prompt[start_idx:end_idx].strip()
                if effect_desc and len(effect_desc) > 10:
                    custom_effects.append(effect_desc)
        
        # Categorized effect detection
        for category, keywords in effect_categories.items():
            for keyword in keywords:
                if keyword in prompt_lower:
                    effect_name = f"{keyword} effect"
                    if effect_name not in [e.lower() for e in custom_effects]:
                        custom_effects.append(effect_name)
        
        logger.info(f"📝 Keyword analysis identified {len(custom_effects)} effects: {custom_effects}")
        return custom_effects
        
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
