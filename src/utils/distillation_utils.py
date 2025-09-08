"""
Distillation utilities
"""

import torch.nn as nn


class DistillationLoss(nn.Module):
    """Distillation loss functions"""
    
    def __init__(self, temperature: float = 3.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha


class FeatureMatching(nn.Module):
    """Feature matching for distillation"""
    
    def match_features(self, student_features, teacher_features):
        """Match student and teacher features"""
        # Placeholder
        return torch.tensor(0.0)
