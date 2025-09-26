"""
RLHF Trainer - Reinforcement Learning from Human Feedback implementation using TRL
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
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModel
from pathlib import Path

# TRL imports for robust RLHF
try:
    from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
    from trl.core import LengthSampler
    HAS_TRL = True
except ImportError:
    HAS_TRL = False
    logging.warning("TRL not available - falling back to basic implementation")

# Try to import video/audio processing libraries
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    logging.warning("OpenCV not available - video feature extraction will use fallback")

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    logging.warning("librosa not available - audio feature extraction will use fallback")

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


class PreferenceDataset(Dataset):
    """Dataset for preference learning"""
    
    def __init__(self, preferences: List[EditingPreference]):
        self.preferences = preferences
    
    def __len__(self):
        return len(self.preferences)
    
    def __getitem__(self, idx):
        pref = self.preferences[idx]
        return {
            'video_id': pref.video_id,
            'edit_a': pref.edit_a,
            'edit_b': pref.edit_b,
            'preference': pref.preference,
            'confidence': pref.confidence
        }


class RewardModel(nn.Module):
    """Reward model to predict human preferences"""
    
    def __init__(self, config: DictConfig):
        super().__init__()
        
        self.hidden_size = config.get('reward_hidden_size', 512)
        self.num_criteria = config.get('num_criteria', 10)  # Pacing, transitions, etc.
        
        # Video understanding encoder (simplified)
        self.video_encoder = nn.Sequential(
            nn.Linear(config.get('video_features_dim', 1024), self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size // 2)
        )
        
        # Edit understanding encoder
        self.edit_encoder = nn.Sequential(
            nn.Linear(config.get('edit_features_dim', 256), self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size // 2)
        )
        
        # Criteria-specific reward heads
        self.criteria_heads = nn.ModuleDict({
            'pacing': nn.Linear(self.hidden_size, 1),
            'transitions': nn.Linear(self.hidden_size, 1),
            'visual_appeal': nn.Linear(self.hidden_size, 1),
            'audio_sync': nn.Linear(self.hidden_size, 1),
            'story_flow': nn.Linear(self.hidden_size, 1),
            'technical_quality': nn.Linear(self.hidden_size, 1),
            'creativity': nn.Linear(self.hidden_size, 1),
            'style_consistency': nn.Linear(self.hidden_size, 1),
            'engagement': nn.Linear(self.hidden_size, 1),
            'overall': nn.Linear(self.hidden_size, 1)
        })
        
        # Final reward aggregation
        self.reward_aggregator = nn.Linear(len(self.criteria_heads), 1)
    
    def forward(self, video_features: torch.Tensor, edit_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            video_features: Features from the input video [batch_size, video_dim]
            edit_features: Features describing the edit decisions [batch_size, edit_dim]
        """
        # Encode inputs
        video_emb = self.video_encoder(video_features)  # [batch, hidden//2]
        edit_emb = self.edit_encoder(edit_features)     # [batch, hidden//2]
        
        # Combine representations
        combined = torch.cat([video_emb, edit_emb], dim=-1)  # [batch, hidden]
        
        # Get criteria-specific rewards
        criteria_rewards = {}
        reward_scores = []
        
        for criterion_name, head in self.criteria_heads.items():
            reward = head(combined)  # [batch, 1]
            criteria_rewards[criterion_name] = reward
            reward_scores.append(reward)
        
        # Aggregate final reward
        all_rewards = torch.cat(reward_scores, dim=-1)  # [batch, num_criteria]
        final_reward = self.reward_aggregator(all_rewards)  # [batch, 1]
        
        return {
            'final_reward': final_reward,
            'criteria_rewards': criteria_rewards,
            'combined_features': combined
        }


# Import enhanced implementation
try:
    from .enhanced_rlhf_trainer import EnhancedRLHFTrainer, VideoEditingRewardModel
    logger.info("Using enhanced RLHF trainer with TRL support")
except ImportError:
    logger.warning("Enhanced RLHF trainer not available, using fallback")
    EnhancedRLHFTrainer = None


class RLHFTrainer:
    """Reinforcement Learning from Human Feedback trainer for video editing"""
    
    def __init__(self, config: DictConfig, model: nn.Module):
        # Try to use enhanced trainer first
        if EnhancedRLHFTrainer is not None and HAS_TRL:
            try:
                self._enhanced_trainer = EnhancedRLHFTrainer(config, model)
                self._use_enhanced = True
                logger.info("✅ Using enhanced RLHF trainer with TRL")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize enhanced trainer: {e}")
        
        # Fallback to basic implementation
        self._use_enhanced = False
        self.config = self._setup_default_config(config)
        self.model = model  # The main video editing model
        self.device = torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize reward model
        self.reward_model = RewardModel(self.config).to(self.device)
        
        # Training hyperparameters
        self.learning_rate = self.config.get('rlhf_lr', 1e-5)
        self.reward_lr = self.config.get('reward_lr', 1e-4)
        self.batch_size = self.config.get('rlhf_batch_size', 4)
        self.ppo_epochs = self.config.get('ppo_epochs', 4)
        self.clip_ratio = self.config.get('ppo_clip_epsilon', 0.2)
        self.value_coef = self.config.get('value_loss_coef', 0.5)
        self.entropy_coef = self.config.get('entropy_coef', 0.01)
        
        # Optimizers
        self.policy_optimizer = AdamW(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.config.get('weight_decay', 0.01)
        )
        
        self.reward_optimizer = AdamW(
            self.reward_model.parameters(),
            lr=self.reward_lr,
            weight_decay=self.config.get('weight_decay', 0.01)
        )
        
        # Experience buffer and training tracking
        self.experience_buffer = []
        self.max_buffer_size = self.config.get('max_buffer_size', 1000)
        self.collected_preferences = []
        self.training_history = []
        
        # Human feedback interface (can be set externally)
        self.human_feedback_interface = None
        
        # Create checkpoint directory
        checkpoint_path = Path(self.config.get('rlhf_checkpoint_path', 'checkpoints/rlhf'))
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = checkpoint_path
        
        logger.info("🎯 RLHF Trainer initialized with comprehensive configuration")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Policy LR: {self.learning_rate}")
        logger.info(f"  Reward LR: {self.reward_lr}")
        logger.info(f"  PPO Epochs: {self.ppo_epochs}")
    
    def __getattr__(self, name):
        """Delegate method calls to enhanced trainer if available"""
        if hasattr(self, '_use_enhanced') and self._use_enhanced and hasattr(self, '_enhanced_trainer'):
            return getattr(self._enhanced_trainer, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def _setup_default_config(self, config: DictConfig) -> DictConfig:
        """Setup default configuration values for RLHF training"""
        
        # Create a copy of the config to avoid modifying the original
        default_config = {
            # Model dimensions
            'video_features_dim': 1024,
            'edit_features_dim': 256,
            'reward_hidden_size': 512,
            
            # Training parameters
            'rlhf_lr': 1e-5,
            'reward_lr': 1e-4,
            'rlhf_batch_size': 4,
            'weight_decay': 0.01,
            'max_grad_norm': 0.5,
            
            # PPO parameters
            'ppo_epochs': 4,
            'ppo_clip_epsilon': 0.2,
            'ppo_batch_size': 8,
            'value_loss_coef': 0.5,
            'entropy_coef': 0.01,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            
            # Reward model parameters
            'num_criteria': 10,
            'reward_mixing_alpha': 0.7,
            
            # Data and training
            'max_buffer_size': 1000,
            'use_simulated_feedback': True,
            'feedback_noise': 0.1,
            
            # Checkpoints and logging
            'rlhf_checkpoint_path': 'checkpoints/rlhf',
            'save_every': 100,
            'eval_every': 50,
            
            # Device
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        # Merge with provided config, giving priority to provided values
        merged_config = default_config.copy()
        
        # Handle both DictConfig and regular dict
        if hasattr(config, '_content'):
            config_dict = config._content
        else:
            config_dict = dict(config) if config else {}
        
        merged_config.update(config_dict)
        
        # Convert back to DictConfig if original was DictConfig
        if hasattr(config, '_content'):
            from omegaconf import DictConfig as OmegaConfig
            return OmegaConfig(merged_config)
        
        return merged_config
    
    def collect_human_feedback(self, video_data: Dict, edit_options: List[Dict]) -> EditingPreference:
        """Collect human feedback on editing choices - supports both real and simulated feedback"""
        
        # Try real human feedback first
        if hasattr(self, 'human_feedback_interface') and self.human_feedback_interface is not None:
            try:
                return self._collect_real_human_feedback(video_data, edit_options)
            except Exception as e:
                logger.warning(f"Real human feedback failed, falling back to simulation: {e}")
        
        # Fallback to simulation
        if self.config.get('use_simulated_feedback', True):
            preference = self._simulate_human_preference(edit_options, video_data)
            return preference
        else:
            raise ValueError("No human feedback interface available and simulation disabled")
    
    def _collect_real_human_feedback(self, video_data: Dict, edit_options: List[Dict]) -> EditingPreference:
        """Collect real human feedback through an interface"""
        
        if not hasattr(self.model, 'autonomous_edit'):
            logger.warning("Model does not have autonomous_edit method, using basic edit simulation")
            return self._simulate_advanced_human_preference(edit_options, video_data)
        
        # Generate actual video edits using the model
        try:
            video_path = video_data.get('video_path') or video_data.get('path')
            if not video_path:
                raise ValueError("No video path provided for real editing")
            
            # Generate two different edits
            edit_a_result = self.model.autonomous_edit(video_path, edit_options[0])
            edit_b_result = self.model.autonomous_edit(video_path, edit_options[1])
            
            # Present to human evaluator through interface
            preference = self.human_feedback_interface.collect_preference(
                video_data=video_data,
                edit_a_path=edit_a_result.get('output_path'),
                edit_b_path=edit_b_result.get('output_path'),
                edit_a_options=edit_options[0],
                edit_b_options=edit_options[1]
            )
            
            return preference
            
        except Exception as e:
            logger.error(f"Real human feedback collection failed: {e}")
            return self._simulate_advanced_human_preference(edit_options, video_data)
    
    def _simulate_advanced_human_preference(self, edit_options: List[Dict], video_data: Dict) -> EditingPreference:
        """Advanced simulation of human preferences with multiple criteria"""
        
        edit_a, edit_b = edit_options[0], edit_options[1]
        
        # Extract comprehensive features for comparison
        features_a = self._extract_single_edit_features(edit_a, 256).numpy()
        features_b = self._extract_single_edit_features(edit_b, 256).numpy()
        
        # Simulate evaluation on multiple criteria
        criteria_scores = {}
        
        # 1. Pacing evaluation
        pacing_score_a = self._evaluate_pacing(edit_a, video_data)
        pacing_score_b = self._evaluate_pacing(edit_b, video_data) 
        criteria_scores['pacing'] = (pacing_score_a, pacing_score_b)
        
        # 2. Visual appeal
        visual_score_a = self._evaluate_visual_appeal(edit_a, features_a)
        visual_score_b = self._evaluate_visual_appeal(edit_b, features_b)
        criteria_scores['visual_appeal'] = (visual_score_a, visual_score_b)
        
        # 3. Story flow
        story_score_a = self._evaluate_story_flow(edit_a, video_data)
        story_score_b = self._evaluate_story_flow(edit_b, video_data)
        criteria_scores['story_flow'] = (story_score_a, story_score_b)
        
        # 4. Technical quality
        tech_score_a = self._evaluate_technical_quality(edit_a)
        tech_score_b = self._evaluate_technical_quality(edit_b)
        criteria_scores['technical_quality'] = (tech_score_a, tech_score_b)
        
        # 5. Creativity
        creative_score_a = self._evaluate_creativity(edit_a, features_a)
        creative_score_b = self._evaluate_creativity(edit_b, features_b)
        criteria_scores['creativity'] = (creative_score_a, creative_score_b)
        
        # Weighted combination of criteria
        weights = {
            'pacing': 0.25,
            'visual_appeal': 0.25,
            'story_flow': 0.20,
            'technical_quality': 0.20,
            'creativity': 0.10
        }
        
        total_score_a = sum(weights[k] * v[0] for k, v in criteria_scores.items())
        total_score_b = sum(weights[k] * v[1] for k, v in criteria_scores.items())
        
        # Add noise to simulate human variability
        noise_factor = self.config.get('feedback_noise', 0.1)
        total_score_a += np.random.normal(0, noise_factor)
        total_score_b += np.random.normal(0, noise_factor)
        
        # Determine preference
        preference = 0 if total_score_a > total_score_b else 1
        confidence = min(0.95, abs(total_score_a - total_score_b) / max(abs(total_score_a), abs(total_score_b)))
        
        # Determine which criteria influenced the decision
        influential_criteria = []
        for criterion, (score_a, score_b) in criteria_scores.items():
            if abs(score_a - score_b) > 0.1:  # Significant difference
                influential_criteria.append(criterion)
        
        return EditingPreference(
            video_id=video_data.get('video_id', 'unknown'),
            edit_a=edit_a,
            edit_b=edit_b,
            preference=preference,
            confidence=confidence,
            criteria=influential_criteria if influential_criteria else ['overall']
        )
    
    def _evaluate_pacing(self, edit: Dict, video_data: Dict) -> float:
        """Evaluate pacing quality of an edit"""
        
        cuts = edit.get('cuts', [])
        total_duration = video_data.get('duration', edit.get('total_duration', 10.0))
        
        if not cuts:
            return 0.5  # Single shot - neutral pacing
        
        # Calculate shot lengths
        cut_times = [0] + [cut.get('time', 0) for cut in cuts] + [total_duration]
        shot_lengths = [cut_times[i+1] - cut_times[i] for i in range(len(cut_times)-1)]
        
        # Ideal pacing varies by content type
        content_type = video_data.get('content_type', 'general')
        if content_type in ['action', 'sports']:
            ideal_length = 1.5  # Fast paced
        elif content_type in ['documentary', 'interview']:
            ideal_length = 4.0  # Slower paced
        else:
            ideal_length = 2.5  # Balanced
        
        # Score based on deviation from ideal
        avg_length = np.mean(shot_lengths)
        length_score = 1.0 / (1.0 + abs(avg_length - ideal_length))
        
        # Penalty for too much variance (inconsistent pacing)
        variance_penalty = min(1.0, np.std(shot_lengths) / avg_length) if avg_length > 0 else 0.0
        consistency_score = 1.0 - variance_penalty
        
        return (length_score + consistency_score) / 2.0
    
    def _evaluate_visual_appeal(self, edit: Dict, features: np.ndarray) -> float:
        """Evaluate visual appeal of an edit"""
        
        # Extract visual-related features
        num_effects = len(edit.get('effects', []))
        color_correction = any(e.get('type') == 'color_correction' for e in edit.get('effects', []))
        
        # Score based on visual enhancements
        effects_score = min(1.0, num_effects / 3.0)  # Moderate effects are good
        color_score = 0.8 if color_correction else 0.5
        
        # Use features to evaluate overall appeal
        feature_score = np.mean(features[:10])  # Use first 10 features as proxy
        
        return (effects_score + color_score + feature_score) / 3.0
    
    def _evaluate_story_flow(self, edit: Dict, video_data: Dict) -> float:
        """Evaluate story flow and narrative coherence"""
        
        structure = edit.get('structure', {})
        has_intro = structure.get('intro_duration', 0) > 0
        has_outro = structure.get('outro_duration', 0) > 0
        climax_position = structure.get('climax_position', 0.5)
        
        # Good story flow has clear structure
        structure_score = 0.0
        if has_intro:
            structure_score += 0.3
        if has_outro:
            structure_score += 0.3
        
        # Climax should be in the middle-to-late part (0.4-0.8)
        if 0.4 <= climax_position <= 0.8:
            structure_score += 0.4
        else:
            structure_score += max(0.0, 0.4 - abs(climax_position - 0.6))
        
        return min(1.0, structure_score)
    
    def _evaluate_technical_quality(self, edit: Dict) -> float:
        """Evaluate technical aspects of the edit"""
        
        quality_metrics = edit.get('quality_metrics', {})
        
        # Average of available quality metrics
        metrics = [
            quality_metrics.get('sharpness_score', 0.5),
            quality_metrics.get('exposure_score', 0.5),
            quality_metrics.get('color_balance', 0.5),
            quality_metrics.get('audio_quality', 0.5),
            quality_metrics.get('stability_score', 0.5),
        ]
        
        # Bonus for having stabilization or noise reduction
        effects = edit.get('effects', [])
        has_stabilization = any(e.get('type') == 'stabilization' for e in effects)
        has_noise_reduction = any(e.get('type') == 'noise_reduction' for e in effects)
        
        bonus = 0.0
        if has_stabilization:
            bonus += 0.1
        if has_noise_reduction:
            bonus += 0.1
        
        return min(1.0, np.mean(metrics) + bonus)
    
    def _evaluate_creativity(self, edit: Dict, features: np.ndarray) -> float:
        """Evaluate creative aspects of the edit"""
        
        # Look for creative elements
        effects = edit.get('effects', [])
        transitions = edit.get('transitions', [])
        style_tags = edit.get('style_tags', [])
        
        # Creative scores
        effects_creativity = min(1.0, len([e for e in effects if e.get('type') not in ['color_correction', 'noise_reduction']]) / 2.0)
        transition_creativity = min(1.0, len([t for t in transitions if t.get('type') != 'cut']) / 3.0)
        style_creativity = min(1.0, len(style_tags) / 3.0)
        
        # Unusual feature combinations (proxy for creativity)
        feature_variance = np.std(features) if len(features) > 0 else 0.0
        feature_creativity = min(1.0, feature_variance * 2.0)
        
        return (effects_creativity + transition_creativity + style_creativity + feature_creativity) / 4.0
    
    def _simulate_human_preference(self, edit_options: List[Dict], video_data: Dict) -> EditingPreference:
        """Simulate human preference for development/testing"""
        
        # Simple heuristics to simulate preferences
        # In practice, this would be actual human feedback
        
        edit_a, edit_b = edit_options[0], edit_options[1]
        
        # Simulate preference based on number of cuts (prefer moderate cutting)
        cuts_a = len(edit_a.get('cuts', []))
        cuts_b = len(edit_b.get('cuts', []))
        
        video_duration = video_data.get('duration', 10.0)
        ideal_cuts = video_duration / 3  # ~3 seconds per segment
        
        score_a = 1.0 / (1.0 + abs(cuts_a - ideal_cuts))
        score_b = 1.0 / (1.0 + abs(cuts_b - ideal_cuts))
        
        preference = 0 if score_a > score_b else 1
        confidence = abs(score_a - score_b) / max(score_a, score_b)
        
        return EditingPreference(
            video_id=video_data.get('video_id', 'unknown'),
            edit_a=edit_a,
            edit_b=edit_b,
            preference=preference,
            confidence=confidence,
            criteria=['pacing', 'technical_quality']
        )
    
    def train_reward_model(self, preferences: List[EditingPreference], epochs: int = 10):
        """Train the reward model on human preferences"""
        
        logger.info(f"🏆 Training reward model on {len(preferences)} preferences...")
        
        dataset = PreferenceDataset(preferences)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.reward_model.train()
        total_loss = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            for batch in dataloader:
                self.reward_optimizer.zero_grad()
                
                # Extract features for both edits (simplified)
                video_features = self._extract_video_features(batch)
                edit_a_features = self._extract_edit_features(batch['edit_a'])
                edit_b_features = self._extract_edit_features(batch['edit_b'])
                
                # Get reward predictions
                rewards_a = self.reward_model(video_features, edit_a_features)['final_reward']
                rewards_b = self.reward_model(video_features, edit_b_features)['final_reward']
                
                # Preference loss (Bradley-Terry model)
                preferences = batch['preference'].float().to(self.device)
                confidence = batch['confidence'].float().to(self.device)
                
                # Logits for preference probability
                logits = rewards_b - rewards_a  # Higher reward for preferred option
                
                # Binary cross-entropy loss weighted by confidence
                loss = F.binary_cross_entropy_with_logits(
                    logits.squeeze(), 
                    preferences,
                    weight=confidence
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), 1.0)
                self.reward_optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / len(dataloader)
            total_loss += avg_epoch_loss
            
            if epoch % 2 == 0:
                logger.info(f"  Epoch {epoch}: Loss = {avg_epoch_loss:.4f}")
        
        logger.info(f"✅ Reward model training complete. Average loss: {total_loss/epochs:.4f}")
    
    def _extract_video_features(self, batch) -> torch.Tensor:
        """Extract real features from video data"""
        
        batch_size = len(batch['video_id'])
        video_dim = self.config.get('video_features_dim', 1024)
        
        if not HAS_CV2:
            logger.warning("OpenCV not available, using random features for video")
            return torch.randn(batch_size, video_dim, device=self.device)
        
        features = []
        
        for i in range(batch_size):
            try:
                video_id = batch['video_id'][i]
                
                # Try to get video data from batch or load from path
                if 'video_path' in batch:
                    video_path = batch['video_path'][i]
                    video_features = self._extract_single_video_features(video_path, video_dim)
                elif 'video_frames' in batch:
                    # Pre-extracted frames
                    frames = batch['video_frames'][i]
                    video_features = self._extract_features_from_frames(frames, video_dim)
                else:
                    # Fallback: try to find video file based on video_id
                    video_features = self._extract_features_by_id(video_id, video_dim)
                
                features.append(video_features)
                
            except Exception as e:
                logger.warning(f"Failed to extract features for video {video_id}: {e}")
                # Use random fallback for this video
                features.append(torch.randn(video_dim))
        
        return torch.stack(features).to(self.device)
    
    def _extract_single_video_features(self, video_path: str, target_dim: int) -> torch.Tensor:
        """Extract features from a single video file"""
        
        video_path = Path(video_path)
        if not video_path.exists():
            logger.warning(f"Video file not found: {video_path}")
            return torch.randn(target_dim)
        
        try:
            # Open video
            cap = cv2.VideoCapture(str(video_path))
            
            # Get basic video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 1.0
            
            # Sample frames uniformly
            frame_indices = np.linspace(0, frame_count - 1, min(16, frame_count), dtype=int)
            
            frame_features = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Extract visual features from frame
                    frame_feature = self._extract_frame_features(frame)
                    frame_features.append(frame_feature)
            
            cap.release()
            
            if frame_features:
                # Aggregate frame features
                video_feature = np.mean(frame_features, axis=0)
                
                # Add temporal/global features
                temporal_features = [
                    duration / 60.0,  # Duration in minutes
                    fps / 30.0,       # FPS normalized to 30
                    width / 1920.0,   # Width normalized to 1080p
                    height / 1080.0,  # Height normalized to 1080p
                    frame_count / 1000.0,  # Frame count normalized
                    len(frame_features) / 16.0,  # Sample ratio
                ]
                
                # Combine features
                combined_features = np.concatenate([video_feature, temporal_features])
                
                # Resize to target dimension
                if len(combined_features) > target_dim:
                    combined_features = combined_features[:target_dim]
                elif len(combined_features) < target_dim:
                    combined_features = np.pad(combined_features, (0, target_dim - len(combined_features)))
                
                return torch.tensor(combined_features, dtype=torch.float32)
            
        except Exception as e:
            logger.warning(f"Video feature extraction failed for {video_path}: {e}")
        
        return torch.randn(target_dim)
    
    def _extract_frame_features(self, frame: np.ndarray) -> np.ndarray:
        """Extract features from a single video frame"""
        
        # Resize frame for consistent processing
        frame = cv2.resize(frame, (224, 224))
        
        # Color statistics
        mean_color = np.mean(frame.reshape(-1, 3), axis=0)
        std_color = np.std(frame.reshape(-1, 3), axis=0)
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Histogram features
        hist_r = cv2.calcHist([frame], [0], None, [8], [0, 256]).flatten()
        hist_g = cv2.calcHist([frame], [1], None, [8], [0, 256]).flatten()
        hist_b = cv2.calcHist([frame], [2], None, [8], [0, 256]).flatten()
        
        # Edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Texture features (simplified)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = np.var(laplacian)
        
        # Combine all features
        features = np.concatenate([
            mean_color / 255.0,  # Normalized RGB means
            std_color / 255.0,   # Normalized RGB stds  
            np.mean(hsv.reshape(-1, 3), axis=0) / [180, 255, 255],  # HSV means
            np.mean(lab.reshape(-1, 3), axis=0) / [100, 255, 255],  # LAB means
            hist_r / np.sum(hist_r),  # Normalized histograms
            hist_g / np.sum(hist_g),
            hist_b / np.sum(hist_b),
            [edge_density, texture_variance / 1000.0]  # Texture features
        ])
        
        return features
    
    def _extract_features_from_frames(self, frames: List[np.ndarray], target_dim: int) -> torch.Tensor:
        """Extract features from pre-loaded frames"""
        
        frame_features = []
        for frame in frames[:16]:  # Use up to 16 frames
            frame_feature = self._extract_frame_features(frame)
            frame_features.append(frame_feature)
        
        if frame_features:
            video_feature = np.mean(frame_features, axis=0)
            
            # Resize to target dimension
            if len(video_feature) > target_dim:
                video_feature = video_feature[:target_dim]
            elif len(video_feature) < target_dim:
                video_feature = np.pad(video_feature, (0, target_dim - len(video_feature)))
                
            return torch.tensor(video_feature, dtype=torch.float32)
        
        return torch.randn(target_dim)
    
    def _extract_features_by_id(self, video_id: str, target_dim: int) -> torch.Tensor:
        """Try to find and extract features by video ID"""
        
        # Common video file extensions
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        
        # Common directory patterns
        search_dirs = [
            Path('data/videos'),
            Path('datasets/videos'), 
            Path(f'data/{video_id}'),
            Path('.')
        ]
        
        for search_dir in search_dirs:
            if search_dir.exists():
                for ext in extensions:
                    video_path = search_dir / f"{video_id}{ext}"
                    if video_path.exists():
                        return self._extract_single_video_features(str(video_path), target_dim)
        
        logger.warning(f"Could not find video file for ID: {video_id}")
        return torch.randn(target_dim)
    
    def _extract_edit_features(self, edits: List[Dict]) -> torch.Tensor:
        """Extract comprehensive features from edit decisions"""
        
        batch_size = len(edits)
        target_dim = self.config.get('edit_features_dim', 256)
        features = []
        
        for edit in edits:
            try:
                edit_features = self._extract_single_edit_features(edit, target_dim)
                features.append(edit_features)
            except Exception as e:
                logger.warning(f"Failed to extract edit features: {e}")
                features.append(torch.randn(target_dim))
        
        return torch.stack(features).to(self.device)
    
    def _extract_single_edit_features(self, edit: Dict, target_dim: int) -> torch.Tensor:
        """Extract detailed features from a single edit decision"""
        
        features = []
        
        # 1. Cut analysis
        cuts = edit.get('cuts', [])
        num_cuts = len(cuts)
        
        if cuts:
            cut_times = [cut.get('time', 0) for cut in cuts]
            cut_intervals = [cut_times[i+1] - cut_times[i] for i in range(len(cut_times)-1)]
            
            features.extend([
                num_cuts / 20.0,  # Normalized cut count
                np.mean(cut_intervals) if cut_intervals else 0.0,  # Average cut interval
                np.std(cut_intervals) if len(cut_intervals) > 1 else 0.0,  # Cut interval variance
                min(cut_intervals) if cut_intervals else 0.0,  # Shortest cut
                max(cut_intervals) if cut_intervals else 0.0,  # Longest cut
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # 2. Transition analysis
        transitions = edit.get('transitions', [])
        num_transitions = len(transitions)
        
        transition_types = {}
        for trans in transitions:
            trans_type = trans.get('type', 'cut')
            transition_types[trans_type] = transition_types.get(trans_type, 0) + 1
        
        # Common transition types
        common_transitions = ['cut', 'fade', 'dissolve', 'wipe', 'iris']
        for trans_type in common_transitions:
            features.append(transition_types.get(trans_type, 0) / max(1, num_transitions))
        
        # Transition timing
        if transitions:
            trans_durations = [t.get('duration', 0.5) for t in transitions]
            features.extend([
                num_transitions / 10.0,  # Normalized transition count
                np.mean(trans_durations),  # Average transition duration
                np.std(trans_durations) if len(trans_durations) > 1 else 0.0
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # 3. Effects analysis
        effects = edit.get('effects', [])
        num_effects = len(effects)
        
        effect_types = {}
        for effect in effects:
            effect_type = effect.get('type', 'none')
            effect_types[effect_type] = effect_types.get(effect_type, 0) + 1
        
        # Common effect categories
        common_effects = ['color_correction', 'stabilization', 'blur', 'sharpen', 'noise_reduction']
        for effect_type in common_effects:
            features.append(effect_types.get(effect_type, 0) / max(1, num_effects))
        
        # Effect intensity analysis
        if effects:
            intensities = [e.get('intensity', 0.5) for e in effects]
            features.extend([
                num_effects / 5.0,  # Normalized effect count
                np.mean(intensities),  # Average effect intensity
                np.std(intensities) if len(intensities) > 1 else 0.0
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # 4. Audio analysis
        audio_edits = edit.get('audio_edits', {})
        features.extend([
            audio_edits.get('volume_changes', 0) / 10.0,  # Volume adjustments
            audio_edits.get('fade_ins', 0) / 5.0,  # Audio fade ins
            audio_edits.get('fade_outs', 0) / 5.0,  # Audio fade outs
            audio_edits.get('noise_reduction', 0.0),  # Noise reduction level
            1.0 if audio_edits.get('music_added', False) else 0.0,  # Music overlay
            1.0 if audio_edits.get('voice_enhanced', False) else 0.0,  # Voice enhancement
        ])
        
        # 5. Pacing and rhythm analysis
        total_duration = edit.get('total_duration', 1.0)
        if cuts:
            # Shot length analysis
            shot_lengths = cut_intervals if cut_intervals else [total_duration]
            
            # Pacing metrics
            avg_shot_length = np.mean(shot_lengths)
            shot_length_variance = np.var(shot_lengths)
            pacing_score = 1.0 / (1.0 + shot_length_variance)  # Lower variance = better pacing
            
            features.extend([
                avg_shot_length / total_duration,  # Relative shot length
                pacing_score,  # Pacing consistency
                len([s for s in shot_lengths if s < 1.0]) / len(shot_lengths),  # Quick cuts ratio
                len([s for s in shot_lengths if s > 5.0]) / len(shot_lengths),  # Long shots ratio
            ])
        else:
            features.extend([1.0, 1.0, 0.0, 1.0])  # Single shot
        
        # 6. Style and genre indicators
        style_features = []
        style_tags = edit.get('style_tags', [])
        common_styles = ['cinematic', 'documentary', 'vlog', 'commercial', 'artistic', 'fast_paced', 'slow_paced']
        
        for style in common_styles:
            style_features.append(1.0 if style in style_tags else 0.0)
        
        features.extend(style_features)
        
        # 7. Technical quality metrics
        quality_metrics = edit.get('quality_metrics', {})
        features.extend([
            quality_metrics.get('sharpness_score', 0.5),  # Image sharpness
            quality_metrics.get('exposure_score', 0.5),   # Exposure quality
            quality_metrics.get('color_balance', 0.5),    # Color balance
            quality_metrics.get('audio_quality', 0.5),    # Audio quality
            quality_metrics.get('stability_score', 0.5),  # Camera stability
        ])
        
        # 8. Semantic features
        semantic_features = edit.get('semantic_features', {})
        features.extend([
            semantic_features.get('action_intensity', 0.5),  # How action-packed
            semantic_features.get('emotion_score', 0.5),     # Emotional content
            semantic_features.get('dialogue_ratio', 0.5),    # Speech vs other audio
            semantic_features.get('scene_changes', 0) / 10.0, # Location changes
        ])
        
        # 9. User preferences and constraints
        preferences = edit.get('user_preferences', {})
        features.extend([
            preferences.get('preferred_pace', 0.5),      # User's pacing preference
            preferences.get('creativity_level', 0.5),    # How creative to be
            preferences.get('music_preference', 0.5),    # Music inclusion level
            1.0 if preferences.get('family_friendly', True) else 0.0,  # Content rating
        ])
        
        # 10. Temporal and structural features
        structure = edit.get('structure', {})
        features.extend([
            structure.get('intro_duration', 0) / total_duration,    # Intro proportion
            structure.get('main_duration', total_duration) / total_duration,  # Main content
            structure.get('outro_duration', 0) / total_duration,    # Outro proportion
            structure.get('climax_position', 0.5),  # Where the peak occurs (0-1)
        ])
        
        # Convert to numpy array and handle sizing
        features = np.array(features, dtype=np.float32)
        
        # Handle NaN values
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Resize to target dimension
        if len(features) > target_dim:
            # Use PCA-like reduction or truncation
            features = features[:target_dim]
        elif len(features) < target_dim:
            # Pad with zeros or repeat pattern
            padding_needed = target_dim - len(features)
            if len(features) > 0:
                # Repeat features cyclically
                repeats = (padding_needed // len(features)) + 1
                extended = np.tile(features, repeats)[:padding_needed]
                features = np.concatenate([features, extended])
            else:
                features = np.zeros(target_dim, dtype=np.float32)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def train_with_human_feedback(self, video_dataset: List[Dict], num_iterations: int = 100):
        """Main RLHF training loop"""
        
        logger.info(f"🚀 Starting RLHF training for {num_iterations} iterations...")
        
        all_preferences = []
        
        for iteration in range(num_iterations):
            logger.info(f"📊 RLHF Iteration {iteration + 1}/{num_iterations}")
            
            # Collect preferences for this iteration
            iteration_preferences = []
            
            # Sample videos from dataset
            sampled_videos = random.sample(video_dataset, min(5, len(video_dataset)))
            
            for video_data in sampled_videos:
                try:
                    preference = self.collect_human_feedback(video_data)
                    iteration_preferences.append(preference)
                except Exception as e:
                    logger.warning(f"Failed to collect feedback for video {video_data.get('video_id', 'unknown')}: {e}")
            
            all_preferences.extend(iteration_preferences)
            
            # Train reward model every 10 iterations
            if iteration % 10 == 0 and all_preferences:
                self.train_reward_model(all_preferences[-50:])  # Use recent preferences
            
            # Policy optimization step
            if iteration % 5 == 0 and len(all_preferences) >= 10:
                self._policy_optimization_step(all_preferences[-20:])
            
            # Log progress
            if iteration % 10 == 0:
                avg_confidence = np.mean([p.confidence for p in iteration_preferences])
                logger.info(f"  Average preference confidence: {avg_confidence:.3f}")
        
        logger.info("✅ RLHF training completed!")
        
        # Save final models
        self._save_rlhf_checkpoint()
    
    def _policy_optimization_step(self, recent_preferences: List[EditingPreference]):
        """Proper PPO (Proximal Policy Optimization) implementation"""
        
        logger.info("🎯 Performing PPO policy optimization...")
        
        if len(recent_preferences) < 4:  # Need sufficient data for PPO
            logger.warning("Insufficient preferences for PPO, skipping optimization")
            return
        
        # PPO hyperparameters
        ppo_epochs = self.config.get('ppo_epochs', 4)
        clip_epsilon = self.config.get('ppo_clip_epsilon', 0.2)
        value_loss_coef = self.config.get('value_loss_coef', 0.5)
        entropy_coef = self.config.get('entropy_coef', 0.01)
        
        # Collect rollout data
        rollout_data = self._collect_rollout_data(recent_preferences)
        
        if not rollout_data['states']:
            logger.warning("No valid rollout data collected")
            return
        
        # Convert to tensors
        states = torch.stack(rollout_data['states']).to(self.device)
        actions = torch.stack(rollout_data['actions']).to(self.device)
        old_log_probs = torch.stack(rollout_data['log_probs']).to(self.device)
        rewards = torch.tensor(rollout_data['rewards'], dtype=torch.float32).to(self.device)
        values = torch.stack(rollout_data['values']).to(self.device)
        
        # Compute advantages using GAE (Generalized Advantage Estimation)
        advantages = self._compute_gae_advantages(rewards, values)
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        
        for ppo_epoch in range(ppo_epochs):
            # Generate mini-batches
            batch_size = min(len(states), self.config.get('ppo_batch_size', 8))
            indices = torch.randperm(len(states))
            
            for start_idx in range(0, len(states), batch_size):
                end_idx = min(start_idx + batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass through policy
                policy_output = self._policy_forward(batch_states)
                new_log_probs = self._get_action_log_probs(policy_output, batch_actions)
                entropy = self._compute_entropy(policy_output)
                
                # Value function forward pass
                predicted_values = self._value_forward(batch_states)
                
                # Compute PPO loss components
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(predicted_values.squeeze(), batch_returns)
                
                # Entropy loss (for exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = policy_loss + value_loss_coef * value_loss + entropy_coef * entropy_loss
                
                # Backward pass
                self.policy_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('max_grad_norm', 0.5))
                self.policy_optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
        
        # Log results
        num_updates = ppo_epochs * max(1, len(states) // batch_size)
        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates
        avg_entropy = total_entropy_loss / num_updates
        
        logger.info(f"  PPO Update - Policy Loss: {avg_policy_loss:.4f}, Value Loss: {avg_value_loss:.4f}, Entropy: {avg_entropy:.4f}")
    
    def _collect_rollout_data(self, preferences: List[EditingPreference]) -> Dict[str, List]:
        """Collect rollout data for PPO training"""
        
        rollout_data = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'values': []
        }
        
        self.model.eval()
        self.reward_model.eval()
        
        with torch.no_grad():
            for preference in preferences:
                try:
                    # Create state representation (video + context)
                    video_data = {'video_id': preference.video_id}
                    video_features = self._extract_video_features({'video_id': [preference.video_id]})
                    
                    # For both edit options, collect policy data
                    for edit_idx, edit in enumerate([preference.edit_a, preference.edit_b]):
                        # State: video features + edit context
                        state = self._create_state_representation(video_features[0], edit)
                        
                        # Action: edit decision features
                        action = self._extract_single_edit_features(edit, self.config.get('edit_features_dim', 256))
                        
                        # Get policy output and log probability
                        policy_output = self._policy_forward(state.unsqueeze(0))
                        log_prob = self._get_action_log_probs(policy_output, action.unsqueeze(0))
                        
                        # Get value estimate
                        value = self._value_forward(state.unsqueeze(0))
                        
                        # Reward based on human preference
                        if edit_idx == preference.preference:
                            # This is the preferred edit
                            base_reward = 1.0
                        else:
                            # This is the non-preferred edit
                            base_reward = -1.0
                        
                        # Scale reward by confidence
                        reward = base_reward * preference.confidence
                        
                        # Add reward model prediction as additional signal
                        edit_features = self._extract_single_edit_features(edit, self.config.get('edit_features_dim', 256))
                        reward_prediction = self.reward_model(
                            video_features[0:1], 
                            edit_features.unsqueeze(0)
                        )['final_reward']
                        
                        # Combine human feedback with reward model (weighted)
                        alpha = self.config.get('reward_mixing_alpha', 0.7)  # Weight for human feedback
                        final_reward = alpha * reward + (1 - alpha) * reward_prediction.item()
                        
                        # Store rollout data
                        rollout_data['states'].append(state)
                        rollout_data['actions'].append(action)
                        rollout_data['log_probs'].append(log_prob.squeeze())
                        rollout_data['rewards'].append(final_reward)
                        rollout_data['values'].append(value.squeeze())
                        
                except Exception as e:
                    logger.warning(f"Failed to collect rollout data for preference: {e}")
                    continue
        
        self.model.train()
        return rollout_data
    
    def _create_state_representation(self, video_features: torch.Tensor, edit_context: Dict) -> torch.Tensor:
        """Create state representation for RL policy"""
        
        # Extract basic context features
        context_features = []
        
        # Video metadata
        context_features.extend([
            edit_context.get('total_duration', 10.0) / 60.0,  # Duration in minutes
            len(edit_context.get('cuts', [])) / 20.0,  # Normalized cut count
            len(edit_context.get('effects', [])) / 5.0,  # Normalized effect count
        ])
        
        # User preferences (if available)
        preferences = edit_context.get('user_preferences', {})
        context_features.extend([
            preferences.get('preferred_pace', 0.5),
            preferences.get('creativity_level', 0.5),
            preferences.get('music_preference', 0.5),
        ])
        
        # Convert to tensor and combine with video features
        context_tensor = torch.tensor(context_features, dtype=torch.float32)
        
        # Concatenate video features with context
        state = torch.cat([video_features, context_tensor])
        
        return state
    
    def _policy_forward(self, states: torch.Tensor) -> torch.Tensor:
        """Forward pass through the policy network"""
        
        # Use the model's policy head if available, otherwise use a simple linear layer
        if hasattr(self.model, 'policy_head'):
            return self.model.policy_head(states)
        else:
            # Fallback: create a simple policy head
            if not hasattr(self, '_policy_head'):
                state_dim = states.shape[-1]
                action_dim = self.config.get('edit_features_dim', 256)
                self._policy_head = nn.Sequential(
                    nn.Linear(state_dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, action_dim * 2)  # Mean and log_std for each action dim
                ).to(self.device)
            
            return self._policy_head(states)
    
    def _value_forward(self, states: torch.Tensor) -> torch.Tensor:
        """Forward pass through the value network"""
        
        # Use the model's value head if available
        if hasattr(self.model, 'value_head'):
            return self.model.value_head(states)
        else:
            # Fallback: create a simple value head
            if not hasattr(self, '_value_head'):
                state_dim = states.shape[-1]
                self._value_head = nn.Sequential(
                    nn.Linear(state_dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1)
                ).to(self.device)
            
            return self._value_head(states)
    
    def _get_action_log_probs(self, policy_output: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Get log probabilities of actions under current policy"""
        
        action_dim = actions.shape[-1]
        
        # Interpret policy output as Gaussian distribution parameters
        mean = policy_output[..., :action_dim]
        log_std = policy_output[..., action_dim:]
        std = torch.exp(log_std.clamp(-20, 2))  # Clamp for numerical stability
        
        # Compute log probability under Gaussian
        log_prob = -0.5 * torch.sum(
            ((actions - mean) / std) ** 2 + 2 * log_std + math.log(2 * math.pi),
            dim=-1
        )
        
        return log_prob
    
    def _compute_entropy(self, policy_output: torch.Tensor) -> torch.Tensor:
        """Compute entropy of the policy distribution"""
        
        action_dim = policy_output.shape[-1] // 2
        log_std = policy_output[..., action_dim:]
        
        # Entropy of multivariate Gaussian
        entropy = 0.5 * torch.sum(log_std + 0.5 * math.log(2 * math.pi * math.e), dim=-1)
        
        return entropy
    
    def _compute_gae_advantages(self, rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Compute Generalized Advantage Estimation (GAE)"""
        
        gamma = self.config.get('gamma', 0.99)
        gae_lambda = self.config.get('gae_lambda', 0.95)
        
        advantages = []
        gae = 0
        
        # Work backwards through the trajectory
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0  # Terminal state
            else:
                next_value = values[t + 1]
            
            # TD error
            delta = rewards[t] + gamma * next_value - values[t]
            
            # GAE accumulation
            gae = delta + gamma * gae_lambda * gae
            advantages.insert(0, gae)
        
        return torch.tensor(advantages, dtype=torch.float32)
    
    def _save_rlhf_checkpoint(self):
        """Save RLHF training checkpoint"""
        
        checkpoint = {
            'model_state': self.model.state_dict(),
            'reward_model_state': self.reward_model.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'reward_optimizer': self.reward_optimizer.state_dict(),
            'config': self.config
        }
        
        checkpoint_path = self.config.get('rlhf_checkpoint_path', 'checkpoints/rlhf_latest.pt')
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"💾 RLHF checkpoint saved to {checkpoint_path}")
    
    def evaluate_model(self, test_preferences: List[EditingPreference]) -> Dict[str, float]:
        """Comprehensive evaluation of the current model against human preferences"""
        
        self.reward_model.eval()
        self.model.eval()
        
        evaluation_results = {
            'preference_accuracy': 0.0,
            'confidence_correlation': 0.0,
            'criteria_breakdown': {},
            'reward_consistency': 0.0,
            'policy_agreement': 0.0,
            'total_evaluations': 0,
            'correct_predictions': 0
        }
        
        correct_predictions = 0
        confidence_errors = []
        criteria_performance = {}
        reward_consistencies = []
        policy_agreements = []
        
        with torch.no_grad():
            for preference in test_preferences:
                try:
                    # Extract real video features instead of random
                    video_data = {'video_id': [preference.video_id]}
                    video_features = self._extract_video_features(video_data)
                    
                    edit_a_features = self._extract_edit_features([preference.edit_a])
                    edit_b_features = self._extract_edit_features([preference.edit_b])
                    
                    # Get reward predictions
                    rewards_a = self.reward_model(video_features, edit_a_features)
                    rewards_b = self.reward_model(video_features, edit_b_features)
                    
                    final_reward_a = rewards_a['final_reward'].item()
                    final_reward_b = rewards_b['final_reward'].item()
                    
                    # Predicted preference
                    predicted_preference = 1 if final_reward_b > final_reward_a else 0
                    
                    # Check accuracy
                    if predicted_preference == preference.preference:
                        correct_predictions += 1
                    
                    # Confidence correlation (how well reward difference correlates with human confidence)
                    reward_diff = abs(final_reward_a - final_reward_b)
                    confidence_errors.append(abs(reward_diff - preference.confidence))
                    
                    # Evaluate criteria-specific performance
                    for criterion in preference.criteria:
                        if criterion not in criteria_performance:
                            criteria_performance[criterion] = {'correct': 0, 'total': 0}
                        
                        criteria_performance[criterion]['total'] += 1
                        if predicted_preference == preference.preference:
                            criteria_performance[criterion]['correct'] += 1
                    
                    # Reward consistency (how consistent are rewards for similar edits)
                    reward_consistency = 1.0 - abs(final_reward_a - final_reward_b) if final_reward_a != 0 or final_reward_b != 0 else 1.0
                    reward_consistencies.append(reward_consistency)
                    
                    # Policy agreement (if we have policy predictions)
                    try:
                        state_a = self._create_state_representation(video_features[0], preference.edit_a)
                        state_b = self._create_state_representation(video_features[0], preference.edit_b)
                        
                        policy_a = self._policy_forward(state_a.unsqueeze(0))
                        policy_b = self._policy_forward(state_b.unsqueeze(0))
                        
                        # Simplified policy agreement metric
                        policy_diff = torch.norm(policy_a - policy_b).item()
                        policy_agreement = 1.0 / (1.0 + policy_diff)  # Higher is better
                        policy_agreements.append(policy_agreement)
                        
                    except Exception as e:
                        logger.debug(f"Policy evaluation failed: {e}")
                    
                    evaluation_results['total_evaluations'] += 1
                    
                except Exception as e:
                    logger.warning(f"Evaluation failed for preference {preference.video_id}: {e}")
                    continue
        
        # Calculate final metrics
        if evaluation_results['total_evaluations'] > 0:
            evaluation_results['preference_accuracy'] = correct_predictions / evaluation_results['total_evaluations']
            evaluation_results['correct_predictions'] = correct_predictions
            
            if confidence_errors:
                evaluation_results['confidence_correlation'] = 1.0 - (np.mean(confidence_errors))
            
            if reward_consistencies:
                evaluation_results['reward_consistency'] = np.mean(reward_consistencies)
            
            if policy_agreements:
                evaluation_results['policy_agreement'] = np.mean(policy_agreements)
            
            # Criteria breakdown
            for criterion, stats in criteria_performance.items():
                if stats['total'] > 0:
                    evaluation_results['criteria_breakdown'][criterion] = stats['correct'] / stats['total']
        
        self.latest_eval_results = evaluation_results
        
        logger.info(f"🔍 Evaluation Results:")
        logger.info(f"  Preference Accuracy: {evaluation_results['preference_accuracy']:.3f}")
        logger.info(f"  Reward Consistency: {evaluation_results['reward_consistency']:.3f}")
        logger.info(f"  Total Evaluations: {evaluation_results['total_evaluations']}")
        
        return evaluation_results
    
    def generate_feedback_report(self) -> Dict[str, Any]:
        """Generate a comprehensive feedback and improvement report with real analysis"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_performance': {},
            'reward_model_stats': {},
            'training_progress': {},
            'improvement_suggestions': [],
            'data_statistics': {},
            'configuration': {}
        }
        
        # Model performance
        if hasattr(self, 'latest_eval_results'):
            report['model_performance'] = self.latest_eval_results
        
        # Reward model statistics
        if hasattr(self, 'reward_model'):
            try:
                # Analyze reward model parameters
                total_params = sum(p.numel() for p in self.reward_model.parameters())
                trainable_params = sum(p.numel() for p in self.reward_model.parameters() if p.requires_grad)
                
                report['reward_model_stats'] = {
                    'total_parameters': total_params,
                    'trainable_parameters': trainable_params,
                    'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
                }
                
                # Get average weights to check for training progress
                avg_weights = []
                for name, param in self.reward_model.named_parameters():
                    if param.requires_grad and len(param.shape) > 0:
                        avg_weights.append(param.abs().mean().item())
                
                if avg_weights:
                    report['reward_model_stats']['avg_weight_magnitude'] = np.mean(avg_weights)
                    report['reward_model_stats']['weight_std'] = np.std(avg_weights)
                
            except Exception as e:
                logger.warning(f"Failed to analyze reward model stats: {e}")
        
        # Training progress
        if hasattr(self, 'training_history'):
            report['training_progress'] = {
                'total_iterations': len(getattr(self, 'training_history', [])),
                'recent_loss_trend': self._analyze_loss_trend(),
                'convergence_indicator': self._check_convergence()
            }
        
        # Configuration
        config_summary = {}
        important_configs = [
            'batch_size', 'learning_rate', 'ppo_epochs', 'ppo_clip_epsilon',
            'video_features_dim', 'edit_features_dim', 'reward_hidden_size'
        ]
        for key in important_configs:
            config_summary[key] = self.config.get(key, 'Not set')
        
        report['configuration'] = config_summary
        
        # Generate intelligent improvement suggestions based on performance
        suggestions = self._generate_improvement_suggestions(report)
        report['improvement_suggestions'] = suggestions
        
        # Data statistics
        if hasattr(self, 'collected_preferences'):
            preferences = self.collected_preferences
            report['data_statistics'] = {
                'total_preferences': len(preferences),
                'avg_confidence': np.mean([p.confidence for p in preferences]) if preferences else 0.0,
                'criteria_distribution': self._analyze_criteria_distribution(preferences),
                'preference_balance': self._analyze_preference_balance(preferences)
            }
        
        logger.info("📋 Generated comprehensive feedback report with real analysis")
        return report
    
    def _analyze_loss_trend(self) -> str:
        """Analyze recent loss trend for training progress"""
        
        if not hasattr(self, 'training_history') or len(self.training_history) < 5:
            return "Insufficient data"
        
        recent_losses = self.training_history[-10:]  # Last 10 losses
        early_losses = recent_losses[:5]
        late_losses = recent_losses[5:]
        
        early_avg = np.mean(early_losses)
        late_avg = np.mean(late_losses)
        
        if late_avg < early_avg * 0.95:
            return "Improving"
        elif late_avg > early_avg * 1.05:
            return "Worsening"
        else:
            return "Stable"
    
    def _check_convergence(self) -> bool:
        """Check if training appears to have converged"""
        
        if not hasattr(self, 'training_history') or len(self.training_history) < 20:
            return False
        
        recent_losses = self.training_history[-20:]
        loss_std = np.std(recent_losses)
        loss_mean = np.mean(recent_losses)
        
        # Converged if coefficient of variation is low
        cv = loss_std / max(loss_mean, 1e-8)
        return cv < 0.05  # 5% coefficient of variation threshold
    
    def _generate_improvement_suggestions(self, report: Dict) -> List[str]:
        """Generate intelligent improvement suggestions based on performance metrics"""
        
        suggestions = []
        
        # Performance-based suggestions
        performance = report.get('model_performance', {})
        accuracy = performance.get('preference_accuracy', 0.0)
        
        if accuracy < 0.6:
            suggestions.append("Low preference accuracy detected. Consider collecting more diverse training data or adjusting model architecture.")
        elif accuracy < 0.75:
            suggestions.append("Moderate preference accuracy. Try fine-tuning hyperparameters or increasing model capacity.")
        
        reward_consistency = performance.get('reward_consistency', 0.0)
        if reward_consistency < 0.5:
            suggestions.append("Low reward consistency. Consider adding regularization or improving feature extraction.")
        
        # Criteria-specific suggestions
        criteria_breakdown = performance.get('criteria_breakdown', {})
        for criterion, score in criteria_breakdown.items():
            if score < 0.6:
                if criterion == 'pacing':
                    suggestions.append("Poor pacing evaluation. Focus on temporal features and cut timing analysis.")
                elif criterion == 'visual_appeal':
                    suggestions.append("Low visual appeal scores. Improve color correction and effects analysis.")
                elif criterion == 'technical_quality':
                    suggestions.append("Technical quality issues. Enhance stabilization and noise reduction detection.")
                elif criterion == 'creativity':
                    suggestions.append("Low creativity scores. Diversify editing techniques and style variations.")
        
        # Training-specific suggestions
        training_progress = report.get('training_progress', {})
        loss_trend = training_progress.get('recent_loss_trend', '')
        
        if loss_trend == 'Worsening':
            suggestions.append("Loss is increasing. Consider reducing learning rate or adding early stopping.")
        elif loss_trend == 'Stable' and not training_progress.get('convergence_indicator', False):
            suggestions.append("Training appears stalled. Try curriculum learning or data augmentation.")
        
        # Data-specific suggestions
        data_stats = report.get('data_statistics', {})
        total_prefs = data_stats.get('total_preferences', 0)
        
        if total_prefs < 100:
            suggestions.append("Limited training data. Collect more human preferences for better generalization.")
        
        avg_confidence = data_stats.get('avg_confidence', 0.0)
        if avg_confidence < 0.5:
            suggestions.append("Low average human confidence. Consider clearer evaluation criteria or better edit options.")
        
        # Configuration suggestions
        config = report.get('configuration', {})
        batch_size = config.get('batch_size', 0)
        
        if isinstance(batch_size, int) and batch_size < 4:
            suggestions.append("Small batch size may hurt training stability. Consider increasing if memory allows.")
        
        # Default suggestions if none generated
        if not suggestions:
            suggestions = [
                "Performance looks good! Consider experimenting with more advanced editing techniques.",
                "Try collecting preferences for different video genres to improve generalization.",
                "Consider implementing active learning to focus on most informative examples."
            ]
        
        return suggestions
    
    def _analyze_criteria_distribution(self, preferences: List[EditingPreference]) -> Dict[str, int]:
        """Analyze distribution of criteria in preferences"""
        
        criteria_counts = {}
        for pref in preferences:
            for criterion in pref.criteria:
                criteria_counts[criterion] = criteria_counts.get(criterion, 0) + 1
        
        return criteria_counts
    
    def _analyze_preference_balance(self, preferences: List[EditingPreference]) -> Dict[str, float]:
        """Analyze balance of preferences (0 vs 1)"""
        
        if not preferences:
            return {'balance_ratio': 0.5, 'total_count': 0}
        
        pref_0_count = sum(1 for p in preferences if p.preference == 0)
        pref_1_count = len(preferences) - pref_0_count
        
        balance_ratio = pref_0_count / len(preferences)
        
        return {
            'balance_ratio': balance_ratio,
            'preference_0_count': pref_0_count,
            'preference_1_count': pref_1_count,
            'total_count': len(preferences)
        }
