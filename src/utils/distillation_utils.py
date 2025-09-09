"""
Distillation utilities for knowledge transfer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class DistillationLoss(nn.Module):
    """Comprehensive distillation loss functions"""
    
    def __init__(self, temperature: float = 3.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()
        
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute distillation loss combining KL divergence and hard target loss
        """
        # Soft target loss (knowledge distillation)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        soft_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Hard target loss (if target labels provided)
        if target is not None:
            hard_loss = F.cross_entropy(student_logits, target)
            total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        else:
            total_loss = soft_loss
            
        return total_loss
    
    def feature_distillation_loss(self, student_features: torch.Tensor, 
                                 teacher_features: torch.Tensor) -> torch.Tensor:
        """Feature-level distillation loss using MSE"""
        # Ensure features have same shape
        if student_features.shape != teacher_features.shape:
            # Add projection layer if needed
            if len(student_features.shape) == 3:  # Sequence features
                student_features = self._project_features(student_features, teacher_features.shape[-1])
        
        return self.mse_loss(student_features, teacher_features)
    
    def attention_distillation_loss(self, student_attention: torch.Tensor,
                                   teacher_attention: torch.Tensor) -> torch.Tensor:
        """Attention map distillation loss"""
        # Normalize attention maps
        student_att_norm = F.softmax(student_attention.view(-1, student_attention.size(-1)), dim=-1)
        teacher_att_norm = F.softmax(teacher_attention.view(-1, teacher_attention.size(-1)), dim=-1)
        
        return self.kl_loss(
            F.log_softmax(student_att_norm, dim=-1),
            teacher_att_norm
        )
    
    def _project_features(self, features: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Project features to target dimension"""
        current_dim = features.shape[-1]
        if current_dim != target_dim:
            # Simple linear projection
            projection = nn.Linear(current_dim, target_dim).to(features.device)
            return projection(features)
        return features


class FeatureMatching(nn.Module):
    """Advanced feature matching for distillation"""
    
    def __init__(self, matching_type: str = 'mse'):
        super().__init__()
        self.matching_type = matching_type
        
        if matching_type == 'mse':
            self.criterion = nn.MSELoss()
        elif matching_type == 'cosine':
            self.criterion = nn.CosineEmbeddingLoss()
        elif matching_type == 'kl':
            self.criterion = nn.KLDivLoss(reduction='batchmean')
        else:
            self.criterion = nn.MSELoss()
    
    def match_features(self, student_features: torch.Tensor, 
                      teacher_features: torch.Tensor) -> torch.Tensor:
        """Match student and teacher features using specified criterion"""
        
        # Ensure features are compatible
        if student_features.shape != teacher_features.shape:
            student_features = self._align_features(student_features, teacher_features)
        
        if self.matching_type == 'mse':
            return self.criterion(student_features, teacher_features)
        
        elif self.matching_type == 'cosine':
            # Flatten features for cosine similarity
            student_flat = student_features.view(student_features.size(0), -1)
            teacher_flat = teacher_features.view(teacher_features.size(0), -1)
            target = torch.ones(student_flat.size(0)).to(student_flat.device)
            return self.criterion(student_flat, teacher_flat, target)
        
        elif self.matching_type == 'kl':
            # Apply softmax for probability distributions
            student_prob = F.softmax(student_features, dim=-1)
            teacher_prob = F.softmax(teacher_features, dim=-1)
            return self.criterion(F.log_softmax(student_features, dim=-1), teacher_prob)
        
        else:
            return self.criterion(student_features, teacher_features)
    
    def _align_features(self, student_features: torch.Tensor, 
                       teacher_features: torch.Tensor) -> torch.Tensor:
        """Align student features to match teacher feature dimensions"""
        
        student_shape = student_features.shape
        teacher_shape = teacher_features.shape
        
        # Handle different cases of shape mismatch
        if len(student_shape) != len(teacher_shape):
            # Different number of dimensions - reshape student
            if len(student_shape) < len(teacher_shape):
                # Add dimensions
                for _ in range(len(teacher_shape) - len(student_shape)):
                    student_features = student_features.unsqueeze(-1)
            else:
                # Remove dimensions by averaging
                for _ in range(len(student_shape) - len(teacher_shape)):
                    student_features = student_features.mean(dim=-1)
        
        # Handle dimension size mismatch
        if student_features.shape[-1] != teacher_shape[-1]:
            # Project to target dimension
            projection = nn.Linear(student_features.shape[-1], teacher_shape[-1])
            projection = projection.to(student_features.device)
            student_features = projection(student_features)
        
        return student_features


class ProgressiveKnowledgeTransfer:
    """Manages progressive knowledge transfer during distillation"""
    
    def __init__(self, num_stages: int = 4, warmup_epochs: int = 5):
        self.num_stages = num_stages
        self.warmup_epochs = warmup_epochs
        self.current_stage = 0
        
    def get_distillation_weights(self, epoch: int) -> Dict[str, float]:
        """Get distillation weights for current training stage"""
        
        # Determine current stage based on epoch
        stage_length = self.warmup_epochs
        self.current_stage = min(epoch // stage_length, self.num_stages - 1)
        
        # Progressive weights - start with low-level, gradually add high-level
        weights = {
            'feature_matching': 1.0,
            'attention_matching': 0.0,
            'output_matching': 0.0,
            'cross_modal_matching': 0.0
        }
        
        if self.current_stage >= 1:
            weights['attention_matching'] = 0.5
        
        if self.current_stage >= 2:
            weights['output_matching'] = 0.7
            
        if self.current_stage >= 3:
            weights['cross_modal_matching'] = 0.8
            
        logger.debug(f"Stage {self.current_stage} distillation weights: {weights}")
        return weights
    
    def should_update_stage(self, epoch: int, loss_history: List[float]) -> bool:
        """Determine if we should move to next distillation stage"""
        
        if len(loss_history) < 5:
            return False
            
        # Check if loss has plateaued
        recent_losses = loss_history[-5:]
        loss_improvement = (recent_losses[0] - recent_losses[-1]) / recent_losses[0]
        
        # Move to next stage if improvement is small
        return loss_improvement < 0.01


class MultiModalDistillationLoss(nn.Module):
    """Specialized loss for multi-modal distillation"""
    
    def __init__(self, modality_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        
        self.modality_weights = modality_weights or {
            'vision': 1.0,
            'audio': 1.0, 
            'text': 1.0,
            'fusion': 1.5  # Higher weight for fusion
        }
        
        self.distill_loss = DistillationLoss()
        self.feature_matcher = FeatureMatching('mse')
        
    def forward(self, student_outputs: Dict[str, torch.Tensor],
                teacher_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute multi-modal distillation loss"""
        
        total_loss = 0.0
        loss_components = {}
        
        # Vision distillation
        if 'vision' in student_outputs and 'vision' in teacher_outputs:
            vision_loss = self.feature_matcher.match_features(
                student_outputs['vision'], teacher_outputs['vision']
            )
            total_loss += self.modality_weights['vision'] * vision_loss
            loss_components['vision'] = vision_loss.item()
        
        # Audio distillation  
        if 'audio' in student_outputs and 'audio' in teacher_outputs:
            audio_loss = self.feature_matcher.match_features(
                student_outputs['audio'], teacher_outputs['audio']
            )
            total_loss += self.modality_weights['audio'] * audio_loss
            loss_components['audio'] = audio_loss.item()
        
        # Text distillation
        if 'text' in student_outputs and 'text' in teacher_outputs:
            text_loss = self.feature_matcher.match_features(
                student_outputs['text'], teacher_outputs['text']
            )
            total_loss += self.modality_weights['text'] * text_loss
            loss_components['text'] = text_loss.item()
        
        # Fusion distillation (most important)
        if 'fusion' in student_outputs and 'fusion' in teacher_outputs:
            fusion_loss = self.feature_matcher.match_features(
                student_outputs['fusion'], teacher_outputs['fusion']
            )
            total_loss += self.modality_weights['fusion'] * fusion_loss
            loss_components['fusion'] = fusion_loss.item()
        
        logger.debug(f"Multi-modal loss components: {loss_components}")
        return total_loss
