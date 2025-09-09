"""
RLHF Trainer - Reinforcement Learning from Human Feedback implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import json
import random
from typing import Dict, List, Any, Optional, Tuple, Union
from omegaconf import DictConfig
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, get_linear_schedule_with_warmup

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


class RLHFTrainer:
    """Reinforcement Learning from Human Feedback trainer for video editing"""
    
    def __init__(self, config: DictConfig, model: nn.Module):
        self.config = config
        self.model = model  # The main video editing model
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Initialize reward model
        self.reward_model = RewardModel(config).to(self.device)
        
        # Training hyperparameters
        self.learning_rate = config.get('rlhf_lr', 1e-5)
        self.reward_lr = config.get('reward_lr', 1e-4)
        self.batch_size = config.get('rlhf_batch_size', 4)
        self.ppo_epochs = config.get('ppo_epochs', 4)
        self.clip_ratio = config.get('clip_ratio', 0.2)
        self.value_coef = config.get('value_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        
        # Optimizers
        self.policy_optimizer = AdamW(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        self.reward_optimizer = AdamW(
            self.reward_model.parameters(),
            lr=self.reward_lr,
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Experience buffer
        self.experience_buffer = []
        self.max_buffer_size = config.get('max_buffer_size', 1000)
        
        logger.info("🎯 RLHF Trainer initialized with reward model")
    
    def collect_human_feedback(self, video_data: Dict, num_samples: int = 2) -> EditingPreference:
        """
        Generate multiple edit options and collect human preference
        In practice, this would interface with a human annotation system
        """
        
        # Generate multiple editing options
        edit_options = []
        
        with torch.no_grad():
            for i in range(num_samples):
                # Add some randomness for diverse options
                random_seed = torch.randint(0, 10000, (1,)).item()
                torch.manual_seed(random_seed)
                
                # Generate edit with the current model
                edit_output = self.model.autonomous_edit(
                    video_data,
                    style_prompt=f"Create an engaging edit (seed: {random_seed})"
                )
                
                edit_options.append({
                    'timeline': edit_output.get('timeline', {}),
                    'cuts': edit_output.get('cuts', []),
                    'transitions': edit_output.get('transitions', []),
                    'effects': edit_output.get('effects', []),
                    'seed': random_seed
                })
        
        # For now, simulate human feedback (in production, this would be real human input)
        preference = self._simulate_human_preference(edit_options, video_data)
        
        return preference
    
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
        """Extract features from video data (simplified)"""
        # In practice, this would use the actual video features from the model
        batch_size = len(batch['video_id'])
        return torch.randn(batch_size, 1024, device=self.device)
    
    def _extract_edit_features(self, edits: List[Dict]) -> torch.Tensor:
        """Extract features from edit decisions"""
        batch_size = len(edits)
        features = []
        
        for edit in edits:
            # Simple feature extraction from edit parameters
            num_cuts = len(edit.get('cuts', []))
            num_transitions = len(edit.get('transitions', []))
            num_effects = len(edit.get('effects', []))
            
            # Create feature vector
            edit_features = [
                num_cuts / 10.0,  # Normalized
                num_transitions / 5.0,
                num_effects / 3.0,
                edit.get('seed', 0) / 10000.0
            ]
            
            # Pad to 256 dimensions
            while len(edit_features) < 256:
                edit_features.append(0.0)
            
            features.append(edit_features[:256])
        
        return torch.tensor(features, device=self.device, dtype=torch.float)
    
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
        """Simplified policy optimization step"""
        
        logger.info("🎯 Performing policy optimization step...")
        
        self.model.train()
        total_policy_loss = 0
        
        for preference in recent_preferences:
            try:
                self.policy_optimizer.zero_grad()
                
                # Get current policy predictions for both edits
                video_data = {'video_id': preference.video_id}  # Simplified
                
                # Simplified policy loss based on reward model feedback
                # In full PPO, this would include advantage estimation and clipping
                
                video_features = torch.randn(1, 1024, device=self.device)
                preferred_edit = preference.edit_a if preference.preference == 0 else preference.edit_b
                non_preferred_edit = preference.edit_b if preference.preference == 0 else preference.edit_a
                
                preferred_features = self._extract_edit_features([preferred_edit])
                non_preferred_features = self._extract_edit_features([non_preferred_edit])
                
                # Get rewards
                preferred_reward = self.reward_model(video_features, preferred_features)['final_reward']
                non_preferred_reward = self.reward_model(video_features, non_preferred_features)['final_reward']
                
                # Simple policy loss: maximize reward for preferred edits
                policy_loss = -preferred_reward.mean() + non_preferred_reward.mean()
                
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.policy_optimizer.step()
                
                total_policy_loss += policy_loss.item()
                
            except Exception as e:
                logger.warning(f"Policy optimization failed for preference: {e}")
        
        avg_policy_loss = total_policy_loss / max(1, len(recent_preferences))
        logger.info(f"  Policy optimization loss: {avg_policy_loss:.4f}")
    
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
        """Evaluate the current model against human preferences"""
        
        self.reward_model.eval()
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for preference in test_preferences:
                try:
                    video_features = torch.randn(1, 1024, device=self.device)
                    edit_a_features = self._extract_edit_features([preference.edit_a])
                    edit_b_features = self._extract_edit_features([preference.edit_b])
                    
                    rewards_a = self.reward_model(video_features, edit_a_features)['final_reward']
                    rewards_b = self.reward_model(video_features, edit_b_features)['final_reward']
                    
                    predicted_preference = 1 if rewards_b > rewards_a else 0
                    
                    if predicted_preference == preference.preference:
                        correct_predictions += 1
                    
                    total_predictions += 1
                    
                except Exception as e:
                    logger.warning(f"Evaluation failed for preference: {e}")
        
        accuracy = correct_predictions / max(1, total_predictions)
        
        return {
            'preference_accuracy': accuracy,
            'total_evaluations': total_predictions,
            'correct_predictions': correct_predictions
        }
    
    def generate_feedback_report(self) -> Dict[str, Any]:
        """Generate a comprehensive feedback and improvement report"""
        
        report = {
            'timestamp': torch.datetime.now().isoformat(),
            'model_performance': {},
            'reward_model_stats': {},
            'improvement_suggestions': []
        }
        
        # Add performance metrics
        if hasattr(self, 'latest_eval_results'):
            report['model_performance'] = self.latest_eval_results
        
        # Add improvement suggestions based on common feedback patterns
        report['improvement_suggestions'] = [
            "Consider adjusting cut frequency based on video content type",
            "Improve transition smoothness for better visual flow", 
            "Balance audio levels more consistently across cuts",
            "Enhance style consistency throughout the edit"
        ]
        
        logger.info("📋 Generated comprehensive feedback report")
        return report
