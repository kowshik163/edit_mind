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
        Sophisticated LLM-powered analysis of editing prompt using advanced reasoning capabilities.
        Employs multi-stage analysis with context understanding and creative interpretation.
        """
        
        # Multi-stage sophisticated prompt for comprehensive analysis
        analysis_system_prompt = """You are an expert video editor and visual effects artist with deep knowledge of:
- Modern video editing techniques and workflows
- Visual effects creation and compositing
- Motion graphics and animation
- Color theory and cinematography  
- Creative storytelling through visual media

Your task is to analyze video editing requests and identify opportunities for custom effects, creative enhancements, and technical implementations."""

        user_analysis_prompt = f"""Analyze this video editing request with expert-level understanding:

"{prompt}"

Provide a comprehensive analysis covering:

1. EXPLICIT EFFECTS: Any directly mentioned effects, filters, or treatments
2. IMPLIED CREATIVE OPPORTUNITIES: Subtle suggestions for enhancements based on context
3. TECHNICAL REQUIREMENTS: Technical aspects needed (color grading, compositing, etc.)
4. STYLE INTERPRETATION: Aesthetic direction and mood implications
5. ADVANCED TECHNIQUES: Professional techniques that would elevate the result

Format your response as a structured analysis with clear categories and actionable items.

Focus on identifying both obvious and nuanced requirements that would create a professional, engaging result."""
        
        try:
            # Use sophisticated conversation-style inference
            messages = [
                {"role": "system", "content": analysis_system_prompt},
                {"role": "user", "content": user_analysis_prompt}
            ]
            
            # Create a more sophisticated prompt for analysis
            formatted_prompt = self._format_conversation_for_inference(messages)
            
            # Advanced tokenization with proper attention
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors='pt',
                max_length=2048,  # Larger context for sophisticated analysis
                truncation=True,
                padding=True
            )
            
            # Move to device if available
            if torch.cuda.is_available() and self.language_model.device != torch.device('cpu'):
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Sophisticated generation with multiple sampling strategies
            with torch.no_grad():
                if hasattr(self.language_model, 'generate'):
                    
                    # Strategy 1: High-quality reasoning (low temperature)
                    reasoning_outputs = self.language_model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.2,  # Low temperature for analytical reasoning
                        top_p=0.9,
                        do_sample=True,
                        num_beams=1,
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                        no_repeat_ngram_size=3
                    )
                    
                    reasoning_response = self.tokenizer.decode(reasoning_outputs[0], skip_special_tokens=True)
                    reasoning_effects = self._extract_effects_from_advanced_analysis(reasoning_response, "reasoning")
                    
                    # Strategy 2: Creative ideation (higher temperature)
                    creative_outputs = self.language_model.generate(
                        **inputs,
                        max_new_tokens=384,
                        temperature=0.7,  # Higher temperature for creative suggestions
                        top_p=0.95,
                        do_sample=True,
                        num_beams=1,
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.15
                    )
                    
                    creative_response = self.tokenizer.decode(creative_outputs[0], skip_special_tokens=True)
                    creative_effects = self._extract_effects_from_advanced_analysis(creative_response, "creative")
                    
                    # Combine and deduplicate effects
                    all_effects = reasoning_effects + creative_effects
                    unique_effects = self._deduplicate_and_rank_effects(all_effects, prompt)
                    
                    if unique_effects:
                        logger.info(f"🧠 Advanced LLM analysis identified {len(unique_effects)} effects: {unique_effects[:3]}...")
                        return unique_effects
                        
                else:
                    logger.warning("Language model generation not available")
                    
        except Exception as e:
            logger.warning(f"Advanced LLM analysis failed: {e}")
        
        # Enhanced fallback with context understanding
        return self._sophisticated_fallback_analysis(prompt)
    
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
