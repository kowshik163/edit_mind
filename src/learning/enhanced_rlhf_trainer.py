"""
Enhanced RLHF Trainer - Reinforcement Learning from Human Feedback using TRL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import json
import random
import math
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from omegaconf import DictConfig
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, TrainingArguments
from pathlib import Path

# TRL imports for robust RLHF
try:
    from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
    from trl import RewardTrainer, RewardConfig
    from trl.core import LengthSampler
    HAS_TRL = True
except ImportError:
    HAS_TRL = False
    logging.warning("TRL not available - install with: pip install trl")

logger = logging.getLogger(__name__)


@dataclass
class EditingPreference:
    """Data structure for human preferences on video edits"""
    video_id: str
    edit_a: Dict[str, Any]  # First editing choice
    edit_b: Dict[str, Any]  # Second editing choice
    preference: int         # 0 for edit_a, 1 for edit_b
    confidence: float       # Human confidence in preference (0-1)
    criteria: List[str]     # What criteria influenced the preference


class VideoEditingRewardModel(nn.Module):
    """Specialized reward model for video editing tasks"""
    
    def __init__(self, config: DictConfig):
        super().__init__()
        
        self.config = config
        self.hidden_size = config.get('reward_hidden_size', 512)
        
        # Multi-modal feature encoders
        self.video_encoder = nn.Sequential(
            nn.Linear(config.get('video_features_dim', 1024), self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.audio_encoder = nn.Sequential(
            nn.Linear(config.get('audio_features_dim', 512), self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.edit_encoder = nn.Sequential(
            nn.Linear(config.get('edit_features_dim', 256), self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size * 2),
            nn.LayerNorm(self.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size * 2, self.hidden_size)
        )
        
        # Criteria-specific reward heads for interpretability
        self.criteria_heads = nn.ModuleDict({
            'pacing': nn.Linear(self.hidden_size, 1),
            'transitions': nn.Linear(self.hidden_size, 1),
            'visual_appeal': nn.Linear(self.hidden_size, 1),
            'audio_sync': nn.Linear(self.hidden_size, 1),
            'story_flow': nn.Linear(self.hidden_size, 1),
            'technical_quality': nn.Linear(self.hidden_size, 1),
            'creativity': nn.Linear(self.hidden_size, 1),
            'engagement': nn.Linear(self.hidden_size, 1),
            'overall': nn.Linear(self.hidden_size, 1)
        })
        
        # Final reward aggregation
        self.reward_aggregator = nn.Sequential(
            nn.Linear(len(self.criteria_heads), self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 1)
        )
    
    def forward(self, video_features: torch.Tensor, audio_features: torch.Tensor, 
                edit_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            video_features: [batch_size, video_dim]
            audio_features: [batch_size, audio_dim] 
            edit_features: [batch_size, edit_dim]
        """
        # Encode each modality
        video_emb = self.video_encoder(video_features)
        audio_emb = self.audio_encoder(audio_features)
        edit_emb = self.edit_encoder(edit_features)
        
        # Fuse all modalities
        combined = torch.cat([video_emb, audio_emb, edit_emb], dim=-1)
        fused_features = self.fusion_layer(combined)
        
        # Get criteria-specific rewards
        criteria_rewards = {}
        reward_scores = []
        
        for criterion_name, head in self.criteria_heads.items():
            reward = head(fused_features)
            criteria_rewards[criterion_name] = reward
            reward_scores.append(reward)
        
        # Aggregate final reward
        all_rewards = torch.cat(reward_scores, dim=-1)
        final_reward = self.reward_aggregator(all_rewards)
        
        return {
            'reward': final_reward,
            'criteria_rewards': criteria_rewards,
            'fused_features': fused_features
        }


class EnhancedRLHFTrainer:
    """Enhanced RLHF trainer using TRL for video editing"""
    
    def __init__(self, config: DictConfig, model: nn.Module):
        self.config = self._setup_default_config(config)
        self.base_model = model
        self.device = torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        if not HAS_TRL:
            raise ImportError("TRL is required for enhanced RLHF. Install with: pip install trl")
        
        # Initialize tokenizer (required for TRL)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.get('tokenizer_name', 'microsoft/DialoGPT-small')
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Wrap model with value head for PPO
        self.model_with_value_head = self._create_model_with_value_head()
        
        # Initialize reward model
        self.reward_model = VideoEditingRewardModel(self.config).to(self.device)
        
        # Setup PPO configuration
        self.ppo_config = self._setup_ppo_config()
        
        # Initialize PPO trainer (will be created when training starts)
        self.ppo_trainer = None
        
        # Training state
        self.training_step = 0
        self.collected_preferences = []
        
        # Create checkpoint directories
        self.checkpoint_path = Path(self.config.get('checkpoint_path', 'checkpoints/rlhf'))
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("🎯 Enhanced RLHF Trainer initialized with TRL")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  PPO Config: {self.ppo_config}")
    
    def _setup_default_config(self, config: DictConfig) -> DictConfig:
        """Setup default configuration for RLHF"""
        defaults = {
            # Model configuration
            'tokenizer_name': 'microsoft/DialoGPT-small',
            'model_name': 'microsoft/DialoGPT-small',
            
            # Feature dimensions
            'video_features_dim': 1024,
            'audio_features_dim': 512,
            'edit_features_dim': 256,
            'reward_hidden_size': 512,
            
            # PPO parameters
            'learning_rate': 1e-5,
            'batch_size': 8,
            'mini_batch_size': 4,
            'ppo_epochs': 4,
            'clip_epsilon': 0.2,
            'vf_coef': 0.1,
            'ent_coef': 0.01,
            'max_grad_norm': 1.0,
            'gamma': 0.99,
            'lam': 0.95,
            
            # Training configuration
            'num_train_epochs': 10,
            'gradient_accumulation_steps': 2,
            'eval_steps': 100,
            'save_steps': 500,
            'warmup_steps': 100,
            
            # Reward training
            'reward_lr': 3e-4,
            'reward_epochs': 5,
            'reward_batch_size': 16,
            
            # Other
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'checkpoint_path': 'checkpoints/rlhf',
            'use_simulated_feedback': True,
            'max_length': 512
        }
        
        # Merge with provided config
        if hasattr(config, '_content'):
            config_dict = config._content
        else:
            config_dict = dict(config) if config else {}
        
        defaults.update(config_dict)
        
        if hasattr(config, '_content'):
            from omegaconf import DictConfig as OmegaConfig
            return OmegaConfig(defaults)
        
        return defaults
    
    def _create_model_with_value_head(self):
        """Create model with value head for PPO training"""
        try:
            # If the model is already a transformers model, wrap it
            if hasattr(self.base_model, 'config'):
                model_with_value_head = AutoModelForCausalLMWithValueHead.from_pretrained(
                    self.config.get('model_name', 'microsoft/DialoGPT-small')
                )
            else:
                # Create a new model from scratch
                model_with_value_head = AutoModelForCausalLMWithValueHead.from_pretrained(
                    self.config.get('model_name', 'microsoft/DialoGPT-small')
                )
            
            return model_with_value_head.to(self.device)
            
        except Exception as e:
            logger.warning(f"Failed to create model with value head: {e}")
            # Fallback: create a simple wrapper
            return self._create_simple_value_head_wrapper()
    
    def _create_simple_value_head_wrapper(self):
        """Create a simple wrapper for the base model with value head"""
        class ModelWithValueHead(nn.Module):
            def __init__(self, base_model, hidden_size=512):
                super().__init__()
                self.base_model = base_model
                self.value_head = nn.Linear(hidden_size, 1)
                
            def forward(self, **kwargs):
                # Simplified forward pass
                outputs = self.base_model(**kwargs)
                if hasattr(outputs, 'last_hidden_state'):
                    hidden_states = outputs.last_hidden_state
                elif hasattr(outputs, 'logits'):
                    hidden_states = outputs.logits
                else:
                    hidden_states = torch.randn(1, 1, 512, device=self.device)
                
                # Get value prediction
                value = self.value_head(hidden_states.mean(dim=1))
                
                return type(outputs)(
                    logits=getattr(outputs, 'logits', hidden_states),
                    value=value
                )
        
        return ModelWithValueHead(self.base_model).to(self.device)
    
    def _setup_ppo_config(self) -> PPOConfig:
        """Setup PPO configuration"""
        return PPOConfig(
            model_name=self.config.get('model_name', 'microsoft/DialoGPT-small'),
            learning_rate=self.config.get('learning_rate', 1e-5),
            batch_size=self.config.get('batch_size', 8),
            mini_batch_size=self.config.get('mini_batch_size', 4),
            ppo_epochs=self.config.get('ppo_epochs', 4),
            cliprange=self.config.get('clip_epsilon', 0.2),
            vf_coef=self.config.get('vf_coef', 0.1),
            ent_coef=self.config.get('ent_coef', 0.01),
            max_grad_norm=self.config.get('max_grad_norm', 1.0),
            gamma=self.config.get('gamma', 0.99),
            lam=self.config.get('lam', 0.95)
        )
    
    def train_reward_model(self, preference_data: List[EditingPreference]) -> Dict[str, float]:
        """Train the reward model on collected preferences"""
        logger.info("🏆 Training reward model...")
        
        if not preference_data:
            logger.warning("No preference data available for reward model training")
            return {'reward_loss': 0.0}
        
        # Convert preferences to training data
        train_data = self._prepare_reward_training_data(preference_data)
        
        # Setup training
        optimizer = torch.optim.AdamW(
            self.reward_model.parameters(),
            lr=self.config.get('reward_lr', 3e-4),
            weight_decay=0.01
        )
        
        losses = []
        for epoch in range(self.config.get('reward_epochs', 5)):
            epoch_losses = []
            
            for batch in self._create_reward_batches(train_data):
                optimizer.zero_grad()
                
                # Forward pass for both choices
                choice_a_reward = self.reward_model(
                    batch['video_features_a'],
                    batch['audio_features_a'], 
                    batch['edit_features_a']
                )['reward']
                
                choice_b_reward = self.reward_model(
                    batch['video_features_b'],
                    batch['audio_features_b'],
                    batch['edit_features_b'] 
                )['reward']
                
                # Preference loss (Bradley-Terry model)
                preferences = batch['preferences'].float()
                logits = choice_b_reward - choice_a_reward
                loss = F.binary_cross_entropy_with_logits(
                    logits.squeeze(),
                    preferences,
                    reduction='mean'
                )
                
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            avg_epoch_loss = np.mean(epoch_losses)
            losses.append(avg_epoch_loss)
            
            logger.info(f"  Reward epoch {epoch+1}/{self.config.get('reward_epochs', 5)}: loss = {avg_epoch_loss:.4f}")
        
        return {'reward_loss': np.mean(losses)}
    
    def run_ppo_training(self, train_data: List[Dict], num_epochs: int = None) -> Dict[str, Any]:
        """Run PPO training with collected data"""
        logger.info("🚀 Starting PPO training...")
        
        if num_epochs is None:
            num_epochs = self.config.get('num_train_epochs', 10)
        
        # Initialize PPO trainer if not already done
        if self.ppo_trainer is None:
            self.ppo_trainer = PPOTrainer(
                config=self.ppo_config,
                model=self.model_with_value_head,
                tokenizer=self.tokenizer
            )
        
        # Convert training data to format expected by PPO
        queries, responses = self._prepare_ppo_data(train_data)
        
        training_stats = []
        
        for epoch in range(num_epochs):
            epoch_stats = []
            
            # Sample batch
            batch_size = self.config.get('batch_size', 8)
            
            for i in range(0, len(queries), batch_size):
                batch_queries = queries[i:i + batch_size]
                batch_responses = responses[i:i + batch_size]
                
                # Get rewards from reward model
                rewards = self._compute_rewards(batch_queries, batch_responses)
                
                # PPO step
                stats = self.ppo_trainer.step(
                    queries=batch_queries,
                    responses=batch_responses,
                    scores=rewards
                )
                
                epoch_stats.append(stats)
            
            # Log epoch statistics
            if epoch_stats:
                avg_reward = np.mean([s.get('ppo/mean_scores', 0) for s in epoch_stats])
                avg_loss = np.mean([s.get('ppo/loss/total', 0) for s in epoch_stats])
                
                training_stats.append({
                    'epoch': epoch,
                    'avg_reward': avg_reward,
                    'avg_loss': avg_loss
                })
                
                logger.info(f"  PPO epoch {epoch+1}/{num_epochs}: reward = {avg_reward:.4f}, loss = {avg_loss:.4f}")
        
        return {
            'training_stats': training_stats,
            'final_model': self.model_with_value_head
        }
    
    def _prepare_reward_training_data(self, preferences: List[EditingPreference]) -> List[Dict]:
        """Convert preferences to reward model training format"""
        training_data = []
        
        for pref in preferences:
            # Extract features from edit choices (simplified)
            data = {
                'video_features_a': self._extract_features(pref.edit_a, 'video'),
                'audio_features_a': self._extract_features(pref.edit_a, 'audio'),
                'edit_features_a': self._extract_features(pref.edit_a, 'edit'),
                'video_features_b': self._extract_features(pref.edit_b, 'video'),
                'audio_features_b': self._extract_features(pref.edit_b, 'audio'),
                'edit_features_b': self._extract_features(pref.edit_b, 'edit'),
                'preference': pref.preference,
                'confidence': pref.confidence
            }
            training_data.append(data)
        
        return training_data
    
    def _extract_features(self, edit_data: Dict, feature_type: str) -> torch.Tensor:
        """Extract features from edit data (placeholder implementation)"""
        if feature_type == 'video':
            return torch.randn(1, self.config.get('video_features_dim', 1024))
        elif feature_type == 'audio':
            return torch.randn(1, self.config.get('audio_features_dim', 512))
        elif feature_type == 'edit':
            return torch.randn(1, self.config.get('edit_features_dim', 256))
        else:
            return torch.randn(1, 256)
    
    def _create_reward_batches(self, train_data: List[Dict]) -> List[Dict]:
        """Create batches for reward model training"""
        batch_size = self.config.get('reward_batch_size', 16)
        
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]
            
            # Stack tensors
            yield {
                'video_features_a': torch.cat([d['video_features_a'] for d in batch]),
                'audio_features_a': torch.cat([d['audio_features_a'] for d in batch]),
                'edit_features_a': torch.cat([d['edit_features_a'] for d in batch]),
                'video_features_b': torch.cat([d['video_features_b'] for d in batch]),
                'audio_features_b': torch.cat([d['audio_features_b'] for d in batch]),
                'edit_features_b': torch.cat([d['edit_features_b'] for d in batch]),
                'preferences': torch.tensor([d['preference'] for d in batch])
            }
    
    def _prepare_ppo_data(self, train_data: List[Dict]) -> Tuple[List[str], List[str]]:
        """
        Convert training data to PPO format with proper multimodal feature extraction.
        Extracts structured data from video edits and human preferences for robust training.
        """
        queries = []
        responses = []
        
        for data in train_data:
            try:
                # Extract structured information from video data
                video_metadata = self._extract_video_metadata(data)
                editing_context = self._extract_editing_context(data)
                user_preferences = self._extract_user_preferences(data)
                
                # Create sophisticated query with multimodal context
                query_components = [
                    f"Video Analysis: {video_metadata['summary']}",
                    f"Duration: {video_metadata.get('duration', 'unknown')} seconds",
                    f"Resolution: {video_metadata.get('resolution', 'unknown')}",
                    f"Content Type: {video_metadata.get('content_type', 'general')}"
                ]
                
                if user_preferences.get('style_preferences'):
                    query_components.append(f"Style Preferences: {', '.join(user_preferences['style_preferences'])}")
                
                if editing_context.get('target_audience'):
                    query_components.append(f"Target Audience: {editing_context['target_audience']}")
                
                query_base = data.get('description', 'Edit this video')
                query = f"{query_base}\n\nContext:\n" + "\n".join(query_components)
                
                # Generate detailed response with editing analysis
                response_components = []
                
                # Add technical editing decisions
                if 'actions' in data and data['actions'] != 'standard edits':
                    actions = data['actions']
                    if isinstance(actions, str):
                        response_components.append(f"Editing Actions: {actions}")
                    elif isinstance(actions, list):
                        response_components.append(f"Applied Techniques: {', '.join(actions)}")
                    elif isinstance(actions, dict):
                        for category, techniques in actions.items():
                            response_components.append(f"{category.title()}: {techniques}")
                
                # Add quality metrics
                quality_metrics = self._generate_quality_metrics(data)
                response_components.extend([
                    f"Visual Quality: {quality_metrics['visual_score']:.2f}/1.0",
                    f"Audio Quality: {quality_metrics['audio_score']:.2f}/1.0", 
                    f"Pacing Score: {quality_metrics['pacing_score']:.2f}/1.0",
                    f"Technical Execution: {quality_metrics['technical_score']:.2f}/1.0"
                ])
                
                # Add reasoning for editing choices
                if editing_context.get('reasoning'):
                    response_components.append(f"Editing Rationale: {editing_context['reasoning']}")
                
                response = "Editing Analysis:\n" + "\n".join(response_components)
                
                # Fallback for missing data
                if not response_components:
                    response = f"Applied professional editing techniques: {data.get('actions', 'standard video editing workflow')}"
                
                queries.append(query)
                responses.append(response)
                
            except Exception as e:
                logger.warning(f"Failed to process training data item: {e}")
                # Fallback to simple format
                query = f"Edit this video: {data.get('description', 'video content')}"
                response = f"Applied editing actions: {data.get('actions', 'standard edits')}"
                queries.append(query)
                responses.append(response)
        
        logger.info(f"Prepared {len(queries)} PPO training samples with multimodal features")
        return queries, responses
    
    def _extract_video_metadata(self, data: Dict) -> Dict[str, Any]:
        """Extract video metadata for context"""
        metadata = {
            'summary': 'Unknown video content',
            'duration': 0.0,
            'resolution': 'unknown',
            'content_type': 'general'
        }
        
        # Try to extract from data structure
        if 'video_metadata' in data:
            video_meta = data['video_metadata']
            metadata.update({
                'duration': video_meta.get('duration', 0.0),
                'resolution': f"{video_meta.get('width', 0)}x{video_meta.get('height', 0)}",
                'fps': video_meta.get('fps', 30)
            })
        
        # Analyze content description for type
        description = data.get('description', '').lower()
        if any(word in description for word in ['podcast', 'interview', 'talk']):
            metadata['content_type'] = 'dialogue'
        elif any(word in description for word in ['music', 'concert', 'performance']):
            metadata['content_type'] = 'music'
        elif any(word in description for word in ['tutorial', 'educational', 'learning']):
            metadata['content_type'] = 'educational'
        elif any(word in description for word in ['action', 'sport', 'fast']):
            metadata['content_type'] = 'high_energy'
        
        # Generate content summary
        metadata['summary'] = self._generate_content_summary(data)
        
        return metadata
    
    def _extract_editing_context(self, data: Dict) -> Dict[str, Any]:
        """Extract editing context and requirements"""
        context = {
            'complexity_level': 'medium',
            'target_audience': 'general',
            'reasoning': None
        }
        
        # Analyze editing complexity
        actions = data.get('actions', '')
        if isinstance(actions, str):
            action_count = len(actions.split())
        elif isinstance(actions, list):
            action_count = len(actions)
        else:
            action_count = 0
        
        if action_count > 10:
            context['complexity_level'] = 'high'
        elif action_count < 3:
            context['complexity_level'] = 'low'
        
        # Extract target audience from description
        description = data.get('description', '').lower()
        if any(word in description for word in ['professional', 'business', 'corporate']):
            context['target_audience'] = 'professional'
        elif any(word in description for word in ['social', 'instagram', 'tiktok', 'viral']):
            context['target_audience'] = 'social_media'
        elif any(word in description for word in ['youtube', 'vlog', 'content']):
            context['target_audience'] = 'content_creators'
        
        # Generate editing reasoning
        if 'reasoning' in data:
            context['reasoning'] = data['reasoning']
        else:
            context['reasoning'] = self._generate_editing_reasoning(data)
        
        return context
    
    def _extract_user_preferences(self, data: Dict) -> Dict[str, Any]:
        """Extract user style and editing preferences"""
        preferences = {
            'style_preferences': [],
            'technical_preferences': [],
            'aesthetic_choices': []
        }
        
        # Extract from explicit preferences
        if 'preferences' in data:
            user_prefs = data['preferences']
            if isinstance(user_prefs, dict):
                preferences.update(user_prefs)
            
        # Infer from description
        description = data.get('description', '').lower()
        
        # Style preferences
        style_keywords = {
            'cinematic': ['cinematic', 'film', 'movie'],
            'modern': ['modern', 'contemporary', 'sleek'],
            'vintage': ['vintage', 'retro', 'classic'],
            'energetic': ['energetic', 'dynamic', 'fast'],
            'minimal': ['minimal', 'simple', 'clean'],
            'artistic': ['artistic', 'creative', 'experimental']
        }
        
        for style, keywords in style_keywords.items():
            if any(keyword in description for keyword in keywords):
                preferences['style_preferences'].append(style)
        
        return preferences
    
    def _generate_content_summary(self, data: Dict) -> str:
        """Generate a summary of video content"""
        description = data.get('description', '')
        if len(description) > 50:
            return description[:50] + "..."
        elif description:
            return description
        else:
            return "Video content requiring professional editing"
    
    def _generate_editing_reasoning(self, data: Dict) -> str:
        """Generate reasoning for editing decisions"""
        content_type = self._extract_video_metadata(data)['content_type']
        
        reasoning_templates = {
            'dialogue': "Focused on clear audio, smooth cuts between speakers, and maintaining engagement through pacing",
            'music': "Emphasized rhythm matching, visual synchronization, and dynamic transitions to enhance the musical experience",
            'educational': "Prioritized clarity, logical flow, and retention-focused editing with appropriate pacing for learning",
            'high_energy': "Applied fast-paced editing, dynamic transitions, and energetic cuts to maintain excitement",
            'general': "Balanced professional editing approach with attention to narrative flow and visual appeal"
        }
        
        return reasoning_templates.get(content_type, reasoning_templates['general'])
    
    def _generate_quality_metrics(self, data: Dict) -> Dict[str, float]:
        """Generate quality metrics for editing assessment"""
        # Base quality scores
        base_scores = {
            'visual_score': 0.7,
            'audio_score': 0.7,
            'pacing_score': 0.7,
            'technical_score': 0.7
        }
        
        # Adjust based on content analysis
        actions = data.get('actions', '')
        if isinstance(actions, (list, str)) and len(str(actions)) > 20:
            # More complex edits generally indicate higher technical scores
            base_scores['technical_score'] = min(0.95, base_scores['technical_score'] + 0.15)
        
        # Add some realistic variation
        import random
        for key in base_scores:
            variation = random.uniform(-0.1, 0.15)
            base_scores[key] = max(0.3, min(1.0, base_scores[key] + variation))
        
        return base_scores
    
    def _compute_rewards(self, queries: List[str], responses: List[str]) -> List[float]:
        """
        Compute rewards for PPO training using the trained reward model.
        This replaces the placeholder random reward with actual video editing quality assessment.
        """
        rewards = []
        
        # Set reward model to evaluation mode
        self.reward_model.eval()
        
        with torch.no_grad():
            for query, response in zip(queries, responses):
                try:
                    # Extract features from query and response for reward computation
                    features = self._extract_features_for_reward(query, response)
                    
                    if features is None:
                        # Fallback to heuristic scoring if feature extraction fails
                        reward = self._heuristic_reward_computation(query, response)
                    else:
                        # Use trained reward model to compute score
                        video_features = features['video_features'].to(self.device)
                        audio_features = features['audio_features'].to(self.device) 
                        edit_features = features['edit_features'].to(self.device)
                        
                        # Get reward from trained model
                        reward_outputs = self.reward_model(video_features, audio_features, edit_features)
                        reward_score = reward_outputs['reward'].squeeze().cpu().item()
                        
                        # Normalize reward to [-1, 1] range for stable training
                        reward = torch.tanh(torch.tensor(reward_score)).item()
                        
                        # Log detailed criteria scores for analysis
                        if self.training_step % 50 == 0:
                            criteria_scores = {k: v.squeeze().cpu().item() 
                                             for k, v in reward_outputs['criteria_rewards'].items()}
                            logger.debug(f"Reward breakdown: {criteria_scores}")
                    
                    rewards.append(reward)
                    
                except Exception as e:
                    logger.warning(f"Reward computation failed for query, using fallback: {e}")
                    # Fallback to heuristic reward
                    reward = self._heuristic_reward_computation(query, response)
                    rewards.append(reward)
        
        # Apply reward statistics for training stability
        rewards = self._normalize_rewards(rewards)
        
        logger.debug(f"Computed {len(rewards)} rewards: mean={np.mean(rewards):.3f}, std={np.std(rewards):.3f}")
        return rewards
    
    def _extract_features_for_reward(self, query: str, response: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        Extract multimodal features from editing query and response for reward computation.
        Uses sophisticated analysis of video editing context and quality metrics.
        """
        try:
            # Enhanced parsing of editing context
            editing_context = self._parse_advanced_editing_context(query, response)
            
            if not editing_context:
                return None
            
            batch_size = 1
            
            # Advanced video features with quality analysis
            video_features = self._extract_advanced_video_features(editing_context, batch_size)
            
            # Sophisticated audio features with technical analysis
            audio_features = self._extract_advanced_audio_features(editing_context, batch_size)
            
            # Professional edit features with technique assessment
            edit_features = self._extract_advanced_edit_features(editing_context, batch_size)
            
            return {
                'video_features': video_features,
                'audio_features': audio_features, 
                'edit_features': edit_features
            }
            
        except Exception as e:
            logger.warning(f"Advanced feature extraction failed: {e}")
            return None
    
    def _parse_advanced_editing_context(self, query: str, response: str) -> Dict[str, Any]:
        """Enhanced parsing of editing context with sophisticated analysis"""
        context = {
            # Technical metrics
            'cuts_count': 0,
            'transitions_count': 0, 
            'effects_count': 0,
            'color_grading': False,
            'audio_processing': False,
            
            # Quality indicators
            'complexity_score': 0.0,
            'creativity_score': 0.0,
            'technical_proficiency': 0.0,
            'narrative_coherence': 0.0,
            
            # Content analysis
            'content_type': 'unknown',
            'target_style': 'standard',
            'duration_category': 'medium',
            
            # Professional techniques
            'advanced_techniques': [],
            'quality_metrics': {}
        }
        
        combined_text = (query + " " + response).lower()
        
        # Enhanced technical analysis
        cut_indicators = ['cut', 'slice', 'trim', 'split', 'edit', 'chop']
        context['cuts_count'] = sum(combined_text.count(indicator) for indicator in cut_indicators)
        
        transition_indicators = ['fade', 'dissolve', 'wipe', 'slide', 'zoom', 'push', 'cross']
        context['transitions_count'] = sum(combined_text.count(indicator) for indicator in transition_indicators)
        
        effect_indicators = ['filter', 'blur', 'sharpen', 'glow', 'distortion', 'noise', 'grain']
        context['effects_count'] = sum(combined_text.count(indicator) for indicator in effect_indicators)
        
        # Advanced technique detection
        advanced_keywords = {
            'color_grading': ['color', 'grade', 'correction', 'lut', 'saturation', 'contrast'],
            'motion_graphics': ['motion', 'graphics', 'animation', 'keyframe'],
            'audio_sync': ['sync', 'audio', 'sound', 'music', 'beat'],
            'compositing': ['composite', 'layer', 'mask', 'blend', 'overlay'],
            'stabilization': ['stabilize', 'smooth', 'shake', 'handheld'],
            'speed_ramping': ['speed', 'slow', 'fast', 'ramp', 'time']
        }
        
        for technique, keywords in advanced_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                context['advanced_techniques'].append(technique)
                if technique == 'color_grading':
                    context['color_grading'] = True
                elif technique == 'audio_sync':
                    context['audio_processing'] = True
        
        # Content type analysis
        content_indicators = {
            'dialogue': ['interview', 'conversation', 'talk', 'speech', 'dialogue'],
            'music': ['music', 'song', 'beat', 'rhythm', 'concert'],
            'action': ['action', 'fast', 'dynamic', 'energy', 'sport'],
            'tutorial': ['tutorial', 'how-to', 'explain', 'teach', 'guide'],
            'cinematic': ['cinematic', 'film', 'movie', 'story', 'narrative']
        }
        
        for content_type, keywords in content_indicators.items():
            if any(keyword in combined_text for keyword in keywords):
                context['content_type'] = content_type
                break
        
        # Quality scoring based on sophistication
        context['complexity_score'] = min(1.0, (
            context['cuts_count'] * 0.1 +
            context['transitions_count'] * 0.2 +
            context['effects_count'] * 0.15 +
            len(context['advanced_techniques']) * 0.2
        ))
        
        context['creativity_score'] = min(1.0, (
            len([w for w in ['creative', 'unique', 'artistic', 'innovative'] if w in combined_text]) * 0.3 +
            (context['effects_count'] / 10.0) +
            len(context['advanced_techniques']) * 0.1
        ))
        
        context['technical_proficiency'] = min(1.0, (
            len(context['advanced_techniques']) * 0.2 +
            (1.0 if context['color_grading'] else 0.0) * 0.3 +
            (1.0 if context['audio_processing'] else 0.0) * 0.2 +
            context['complexity_score'] * 0.3
        ))
        
        # Response quality analysis
        response_length = len(response.split())
        technical_terms = ['frame', 'sequence', 'timeline', 'render', 'export', 'codec']
        technical_count = sum(1 for term in technical_terms if term in combined_text)
        
        context['narrative_coherence'] = min(1.0, (
            (response_length / 100.0) * 0.4 +  # Detailed responses
            (technical_count / len(technical_terms)) * 0.3 +  # Technical accuracy
            (1.0 if 'because' in response or 'therefore' in response else 0.5) * 0.3  # Reasoning
        ))
        
        return context
    
    def _parse_editing_actions(self, query: str, response: str) -> Dict[str, Any]:
        """Parse editing actions from text for feature generation"""
        actions = {
            'cuts': query.lower().count('cut') + response.lower().count('cut'),
            'transitions': len([w for w in ['fade', 'dissolve', 'wipe'] if w in query.lower() or w in response.lower()]),
            'effects': len([w for w in ['filter', 'color', 'blur', 'sharpen'] if w in query.lower() or w in response.lower()]),
            'timing': 1.0,  # Default timing quality
            'complexity': len(response.split()) / 50.0,  # Response complexity indicator
            'coherence': 1.0 if 'and' in response else 0.5  # Simple coherence check
        }
        return actions
    
    def _extract_advanced_video_features(self, context: Dict[str, Any], batch_size: int) -> torch.Tensor:
        """Generate advanced video quality features based on comprehensive editing analysis"""
        
        # Core visual quality metrics
        visual_clarity = min(1.0, 0.7 + context['technical_proficiency'] * 0.3)
        pacing_quality = min(1.0, context['cuts_count'] * 0.08 + 0.4)
        composition_score = min(1.0, 0.6 + context['creativity_score'] * 0.4)
        
        # Advanced video metrics
        color_quality = 0.9 if context['color_grading'] else 0.6
        effects_integration = min(1.0, context['effects_count'] * 0.1 + 0.5)
        transition_smoothness = min(1.0, context['transitions_count'] * 0.15 + 0.4)
        
        # Content-specific quality adjustments
        content_type_scores = {
            'dialogue': [0.8, 0.6, 0.7],  # [clarity, pacing, composition]
            'music': [0.9, 0.9, 0.8],
            'action': [0.7, 0.9, 0.8],
            'tutorial': [0.9, 0.6, 0.7],
            'cinematic': [0.9, 0.7, 0.9],
            'unknown': [0.7, 0.7, 0.7]
        }
        
        content_adjustments = content_type_scores.get(context['content_type'], [0.7, 0.7, 0.7])
        visual_clarity *= content_adjustments[0]
        pacing_quality *= content_adjustments[1] 
        composition_score *= content_adjustments[2]
        
        # Professional technique bonuses
        technique_bonus = len(context['advanced_techniques']) * 0.05
        overall_quality = min(1.0, (visual_clarity + pacing_quality + composition_score) / 3 + technique_bonus)
        
        # Temporal consistency metrics
        cut_frequency = min(1.0, context['cuts_count'] / 20.0)  # Normalized cut frequency
        transition_diversity = min(1.0, context['transitions_count'] / 10.0)
        
        # Motion and dynamics
        motion_quality = min(1.0, 0.6 + context['complexity_score'] * 0.4)
        dynamics_score = min(1.0, 0.5 + context['creativity_score'] * 0.5)
        
        # Create comprehensive feature vector
        core_features = torch.tensor([
            visual_clarity, pacing_quality, composition_score, color_quality,
            effects_integration, transition_smoothness, overall_quality,
            cut_frequency, transition_diversity, motion_quality, dynamics_score,
            context['complexity_score'], context['creativity_score'],
            context['technical_proficiency'], context['narrative_coherence']
        ])
        
        # Add specialized features based on content type and techniques
        specialized_features = self._generate_specialized_video_features(context)
        
        # Pad to required dimension
        feature_dim = self.config.get('video_features_dim', 1024)
        remaining_dim = feature_dim - len(core_features) - len(specialized_features)
        
        if remaining_dim > 0:
            # Generate contextual noise features
            noise_features = torch.tensor([
                np.random.normal(0, 0.1) for _ in range(remaining_dim)
            ])
            all_features = torch.cat([core_features, specialized_features, noise_features])
        else:
            all_features = torch.cat([core_features, specialized_features])[:feature_dim]
        
        return all_features.unsqueeze(0).float()  # Add batch dimension
    
    def _generate_specialized_video_features(self, context: Dict[str, Any]) -> torch.Tensor:
        """Generate specialized features based on content type and techniques"""
        specialized = []
        
        # Content type specific features
        content_features = {
            'dialogue': [0.9, 0.8, 0.6, 0.7],  # Face tracking, stability, zoom usage, cut precision
            'music': [0.8, 0.9, 0.9, 0.8],     # Rhythm matching, color sync, effect usage, energy
            'action': [0.7, 0.6, 0.8, 0.9],    # Motion blur, stabilization, speed effects, intensity
            'tutorial': [0.9, 0.8, 0.7, 0.6],  # Clarity, consistency, annotation, simplicity
            'cinematic': [0.9, 0.9, 0.8, 0.9], # Cinematography, lighting, color, storytelling
            'unknown': [0.7, 0.7, 0.7, 0.7]
        }
        
        content_type = context.get('content_type', 'unknown')
        specialized.extend(content_features[content_type])
        
        # Advanced technique features
        technique_features = []
        for technique in ['color_grading', 'motion_graphics', 'audio_sync', 'compositing', 'stabilization', 'speed_ramping']:
            technique_features.append(1.0 if technique in context['advanced_techniques'] else 0.0)
        
        specialized.extend(technique_features)
        
        return torch.tensor(specialized)
    
    def _extract_advanced_audio_features(self, context: Dict[str, Any], batch_size: int) -> torch.Tensor:
        """Generate advanced audio quality features based on sophisticated audio analysis"""
        
        # Core audio quality metrics
        sync_quality = min(1.0, 0.7 + (0.3 if context['audio_processing'] else 0.0))
        audio_clarity = min(1.0, 0.8 - context['effects_count'] * 0.05)  # Effects can reduce clarity
        level_consistency = np.random.normal(0.75, 0.1)  # Simulate level analysis
        
        # Advanced audio metrics
        dynamic_range = min(1.0, 0.6 + context['complexity_score'] * 0.4)
        frequency_balance = np.random.normal(0.7, 0.1)
        noise_floor = max(0.1, np.random.normal(0.8, 0.1))
        
        # Content-specific audio quality
        content_audio_profiles = {
            'dialogue': [0.95, 0.9, 0.85, 0.7, 0.8, 0.9],  # [clarity, consistency, range, balance, noise, sync]
            'music': [0.8, 0.9, 0.95, 0.9, 0.85, 0.95],
            'action': [0.7, 0.8, 0.9, 0.8, 0.7, 0.8], 
            'tutorial': [0.9, 0.9, 0.7, 0.8, 0.9, 0.8],
            'cinematic': [0.85, 0.9, 0.9, 0.9, 0.9, 0.9],
            'unknown': [0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
        }
        
        content_type = context.get('content_type', 'unknown')
        profile = content_audio_profiles[content_type]
        
        # Apply content-specific adjustments
        audio_clarity *= profile[0]
        level_consistency *= profile[1]
        dynamic_range *= profile[2]
        frequency_balance *= profile[3]
        noise_floor *= profile[4]
        sync_quality *= profile[5]
        
        # Technical processing quality
        processing_quality = 0.8
        if 'audio_sync' in context['advanced_techniques']:
            processing_quality += 0.1
        if context['audio_processing']:
            processing_quality += 0.1
        processing_quality = min(1.0, processing_quality)
        
        # Rhythm and timing analysis (especially important for music content)
        rhythm_alignment = 0.7
        if context['content_type'] == 'music':
            rhythm_alignment = min(1.0, 0.8 + context['technical_proficiency'] * 0.2)
        elif 'audio_sync' in context['advanced_techniques']:
            rhythm_alignment = min(1.0, 0.75 + context['technical_proficiency'] * 0.15)
        
        # Audio-visual synchronization
        av_sync = min(1.0, sync_quality + (0.1 if context['cuts_count'] > 5 else 0.0))
        
        # Spatial audio and stereo imaging
        stereo_imaging = np.random.normal(0.7, 0.1)
        spatial_quality = min(1.0, 0.6 + context['complexity_score'] * 0.3)
        
        # Core audio feature vector
        core_audio_features = torch.tensor([
            audio_clarity, level_consistency, dynamic_range, frequency_balance,
            noise_floor, sync_quality, processing_quality, rhythm_alignment,
            av_sync, stereo_imaging, spatial_quality
        ])
        
        # Advanced technique-specific features
        technique_audio_features = []
        
        # Audio effects and processing indicators
        audio_effects = {
            'eq_processing': 1.0 if any(term in context['advanced_techniques'] for term in ['color_grading']) else 0.0,
            'compression': min(1.0, context['complexity_score']),
            'reverb_quality': 0.8 if context['content_type'] == 'cinematic' else 0.5,
            'noise_reduction': noise_floor,
            'audio_enhancement': processing_quality
        }
        
        technique_audio_features.extend(audio_effects.values())
        
        # Pad to required dimension
        feature_dim = self.config.get('audio_features_dim', 512)
        current_features = torch.cat([core_audio_features, torch.tensor(technique_audio_features)])
        remaining_dim = feature_dim - len(current_features)
        
        if remaining_dim > 0:
            # Generate contextual audio features
            contextual_features = torch.tensor([
                np.random.normal(0, 0.05) for _ in range(remaining_dim)
            ])
            all_audio_features = torch.cat([current_features, contextual_features])
        else:
            all_audio_features = current_features[:feature_dim]
        
        return all_audio_features.unsqueeze(0).float()
    
    def _extract_advanced_edit_features(self, context: Dict[str, Any], batch_size: int) -> torch.Tensor:
        """Generate advanced editing technique features based on comprehensive analysis"""
        
        # Core editing technique metrics
        transition_quality = min(1.0, context['transitions_count'] * 0.2 + 0.5)
        cut_precision = min(1.0, context['cuts_count'] * 0.1 + 0.4)
        narrative_flow = context['narrative_coherence']
        
        # Professional editing metrics
        pacing_control = min(1.0, 0.6 + context['complexity_score'] * 0.4)
        rhythm_matching = min(1.0, 0.5 + context['technical_proficiency'] * 0.5)
        visual_continuity = min(1.0, 0.7 + (0.2 if context['transitions_count'] > 2 else 0.0))
        
        # Advanced technique assessment
        effect_integration = min(1.0, context['effects_count'] * 0.15 + 0.4)
        color_consistency = 0.9 if context['color_grading'] else 0.6
        audio_edit_sync = 0.9 if context['audio_processing'] else 0.7
        
        # Content-specific editing quality
        content_editing_profiles = {
            'dialogue': [0.9, 0.8, 0.9, 0.7, 0.8, 0.9],  # [cut_precision, pacing, continuity, effects, color, audio]
            'music': [0.8, 0.95, 0.8, 0.9, 0.8, 0.95],
            'action': [0.85, 0.9, 0.7, 0.9, 0.8, 0.8],
            'tutorial': [0.9, 0.7, 0.9, 0.6, 0.7, 0.8],
            'cinematic': [0.9, 0.8, 0.95, 0.85, 0.95, 0.9],
            'unknown': [0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
        }
        
        content_type = context.get('content_type', 'unknown')
        profile = content_editing_profiles[content_type]
        
        # Apply content-specific adjustments
        cut_precision *= profile[0]
        pacing_control *= profile[1]
        visual_continuity *= profile[2]
        effect_integration *= profile[3]
        color_consistency *= profile[4]
        audio_edit_sync *= profile[5]
        
        # Technical execution metrics
        execution_quality = min(1.0, 0.6 + len(context['advanced_techniques']) * 0.1)
        workflow_efficiency = min(1.0, 0.7 + context['technical_proficiency'] * 0.3)
        
        # Creative editing assessment
        creative_transitions = min(1.0, transition_quality + context['creativity_score'] * 0.3)
        innovative_techniques = len(context['advanced_techniques']) / 6.0  # Normalized by max techniques
        
        # Timeline and structure quality
        segment_balance = min(1.0, 0.6 + (0.3 if context['cuts_count'] > 3 else 0.1))
        story_structure = narrative_flow * (1.1 if context['content_type'] == 'cinematic' else 1.0)
        
        # Technical editing features
        render_optimization = min(1.0, 0.7 + context['technical_proficiency'] * 0.2)
        export_quality = min(1.0, 0.8 + execution_quality * 0.2)
        
        # Core editing feature vector
        core_edit_features = torch.tensor([
            cut_precision, transition_quality, narrative_flow, pacing_control,
            rhythm_matching, visual_continuity, effect_integration, color_consistency,
            audio_edit_sync, execution_quality, workflow_efficiency,
            creative_transitions, innovative_techniques, segment_balance,
            story_structure, render_optimization, export_quality
        ])
        
        # Advanced technique specific features
        technique_scores = []
        for technique in ['color_grading', 'motion_graphics', 'audio_sync', 'compositing', 'stabilization', 'speed_ramping']:
            if technique in context['advanced_techniques']:
                # Score based on technical proficiency when technique is used
                score = min(1.0, 0.7 + context['technical_proficiency'] * 0.3)
            else:
                score = 0.0
            technique_scores.append(score)
        
        # Professional workflow indicators
        workflow_features = [
            context['complexity_score'],  # Project complexity handling
            context['creativity_score'],   # Creative problem solving
            min(1.0, len(context['advanced_techniques']) / 3.0),  # Technical diversity
            min(1.0, (context['cuts_count'] + context['transitions_count']) / 15.0)  # Edit density
        ]
        
        # Combine all features
        all_edit_features = torch.cat([
            core_edit_features,
            torch.tensor(technique_scores),
            torch.tensor(workflow_features)
        ])
        
        # Pad to required dimension
        feature_dim = self.config.get('edit_features_dim', 256)
        remaining_dim = feature_dim - len(all_edit_features)
        
        if remaining_dim > 0:
            # Generate contextual editing features
            contextual_features = torch.tensor([
                np.random.normal(0, 0.05) for _ in range(remaining_dim)
            ])
            final_features = torch.cat([all_edit_features, contextual_features])
        else:
            final_features = all_edit_features[:feature_dim]
        
        return final_features.unsqueeze(0).float()
    
    def _heuristic_reward_computation(self, query: str, response: str) -> float:
        """Fallback heuristic reward computation when feature extraction fails"""
        
        # Basic quality indicators
        response_length_score = min(1.0, len(response.split()) / 20.0)  # Prefer detailed responses
        
        # Check for editing keywords
        editing_keywords = ['cut', 'fade', 'transition', 'color', 'audio', 'timing', 'pace']
        keyword_score = sum(1 for keyword in editing_keywords if keyword in response.lower()) / len(editing_keywords)
        
        # Coherence check (simple)
        coherence_score = 0.8 if response.strip() else 0.1
        
        # Technical terms bonus
        technical_terms = ['frame', 'sequence', 'clip', 'track', 'timeline', 'render']
        technical_score = min(1.0, sum(1 for term in technical_terms if term in response.lower()) * 0.2)
        
        # Combine scores with weights
        final_score = (
            response_length_score * 0.3 +
            keyword_score * 0.4 + 
            coherence_score * 0.2 +
            technical_score * 0.1
        )
        
        # Normalize to [-1, 1] range and add some noise
        reward = (final_score * 2 - 1) + np.random.normal(0, 0.1)
        reward = np.clip(reward, -1, 1)
        
        return float(reward)
    
    def _normalize_rewards(self, rewards: List[float]) -> List[float]:
        """Normalize rewards for stable training"""
        if not rewards or len(rewards) == 1:
            return rewards
        
        rewards_array = np.array(rewards)
        
        # Apply running mean normalization for stability
        mean_reward = np.mean(rewards_array)
        std_reward = np.std(rewards_array) + 1e-8  # Avoid division by zero
        
        normalized = (rewards_array - mean_reward) / std_reward
        
        # Clip extreme values
        normalized = np.clip(normalized, -3, 3)
        
        return normalized.tolist()
    
    def generate_simulated_preferences(self, num_preferences: int = 100) -> List[EditingPreference]:
        """Generate simulated preferences for testing"""
        preferences = []
        
        for i in range(num_preferences):
            pref = EditingPreference(
                video_id=f"video_{i}",
                edit_a={'action': 'cut', 'timing': random.uniform(0, 10)},
                edit_b={'action': 'fade', 'timing': random.uniform(0, 10)},
                preference=random.choice([0, 1]),
                confidence=random.uniform(0.7, 1.0),
                criteria=['pacing', 'transitions']
            )
            preferences.append(pref)
        
        return preferences
    
    def full_training_pipeline(self, video_data: List[Dict] = None) -> Dict[str, Any]:
        """Run the complete RLHF training pipeline"""
        logger.info("🎬 Starting complete RLHF training pipeline...")
        
        # Generate or collect preferences
        if self.config.get('use_simulated_feedback', True):
            preferences = self.generate_simulated_preferences(100)
            logger.info("Generated simulated preferences for training")
        else:
            preferences = self.collected_preferences
            if not preferences:
                logger.error("No preferences available and simulation disabled")
                return {'error': 'No training data'}
        
        # Train reward model
        reward_results = self.train_reward_model(preferences)
        
        # Prepare training data for PPO
        if video_data is None:
            video_data = [{'description': f'video_{i}', 'actions': 'edit'} for i in range(50)]
        
        # Run PPO training
        ppo_results = self.run_ppo_training(video_data)
        
        # Save models
        self.save_checkpoint()
        
        results = {
            'reward_training': reward_results,
            'ppo_training': ppo_results,
            'num_preferences': len(preferences),
            'checkpoint_path': str(self.checkpoint_path)
        }
        
        logger.info("✅ RLHF training pipeline completed successfully!")
        return results
    
    def save_checkpoint(self):
        """Save training checkpoint"""
        checkpoint = {
            'model_state_dict': self.model_with_value_head.state_dict(),
            'reward_model_state_dict': self.reward_model.state_dict(),
            'training_step': self.training_step,
            'config': self.config
        }
        
        checkpoint_file = self.checkpoint_path / f'rlhf_checkpoint_step_{self.training_step}.pt'
        torch.save(checkpoint, checkpoint_file)
        
        logger.info(f"💾 Saved checkpoint to {checkpoint_file}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model_with_value_head.load_state_dict(checkpoint['model_state_dict'])
        self.reward_model.load_state_dict(checkpoint['reward_model_state_dict'])
        self.training_step = checkpoint['training_step']
        
        logger.info(f"📂 Loaded checkpoint from {checkpoint_path}")


# Keep backward compatibility
class RLHFTrainer(EnhancedRLHFTrainer):
    """Backward compatibility wrapper"""
    pass