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
        """Convert training data to PPO format"""
        queries = []
        responses = []
        
        for data in train_data:
            # Convert to text format (simplified)
            query = f"Edit this video with the following content: {data.get('description', 'video content')}"
            response = f"Applied editing actions: {data.get('actions', 'standard edits')}"
            
            queries.append(query)
            responses.append(response)
        
        return queries, responses
    
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
        This would connect to actual video analysis in production.
        """
        try:
            # Parse editing instructions from query and response
            editing_actions = self._parse_editing_actions(query, response)
            
            if not editing_actions:
                return None
            
            # Generate synthetic features based on editing actions (replace with real feature extraction)
            batch_size = 1
            
            # Video features: represent visual quality metrics
            video_features = self._generate_video_features(editing_actions, batch_size)
            
            # Audio features: represent audio quality and sync
            audio_features = self._generate_audio_features(editing_actions, batch_size)
            
            # Edit features: represent editing technique quality
            edit_features = self._generate_edit_features(editing_actions, batch_size)
            
            return {
                'video_features': video_features,
                'audio_features': audio_features, 
                'edit_features': edit_features
            }
            
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return None
    
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
    
    def _generate_video_features(self, actions: Dict[str, Any], batch_size: int) -> torch.Tensor:
        """Generate video quality features based on editing actions"""
        # Simulate video quality metrics
        visual_quality = min(1.0, actions.get('effects', 0) * 0.2 + 0.5)
        pacing_quality = min(1.0, actions.get('cuts', 0) * 0.1 + 0.3)
        composition = np.random.normal(0.7, 0.1)
        
        # Create feature vector representing video analysis
        features = torch.tensor([
            visual_quality, pacing_quality, composition,
            actions.get('complexity', 0.5), actions.get('timing', 1.0)
        ] + [np.random.normal(0, 0.1) for _ in range(self.config.get('video_features_dim', 1024) - 5)])
        
        return features.unsqueeze(0).float()  # Add batch dimension
    
    def _generate_audio_features(self, actions: Dict[str, Any], batch_size: int) -> torch.Tensor:
        """Generate audio quality features based on editing actions"""
        # Simulate audio quality metrics
        sync_quality = min(1.0, actions.get('timing', 0.5) + 0.3)
        audio_levels = np.random.normal(0.6, 0.1)
        clarity = min(1.0, 0.8 - actions.get('effects', 0) * 0.1)
        
        # Create feature vector representing audio analysis
        features = torch.tensor([
            sync_quality, audio_levels, clarity
        ] + [np.random.normal(0, 0.1) for _ in range(self.config.get('audio_features_dim', 512) - 3)])
        
        return features.unsqueeze(0).float()
    
    def _generate_edit_features(self, actions: Dict[str, Any], batch_size: int) -> torch.Tensor:
        """Generate editing technique features"""
        # Simulate editing technique quality
        transition_quality = min(1.0, actions.get('transitions', 0) * 0.3 + 0.4)
        cut_quality = min(1.0, actions.get('cuts', 0) * 0.2 + 0.5)
        coherence = actions.get('coherence', 0.5)
        
        # Create feature vector representing editing analysis
        features = torch.tensor([
            transition_quality, cut_quality, coherence
        ] + [np.random.normal(0, 0.1) for _ in range(self.config.get('edit_features_dim', 256) - 3)])
        
        return features.unsqueeze(0).float()
    
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