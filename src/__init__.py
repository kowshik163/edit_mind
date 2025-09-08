"""
Autonomous AI Video Editor
==========================

A self-thinking, self-learning, and self-improving AI system for video editing
that combines fine-tuning, distillation, and multimodal fusion.
"""

__version__ = "0.1.0"
__author__ = "Auto Editor Team"
__email__ = "team@auto-editor.ai"

from .core.hybrid_ai import HybridVideoAI
from .core.orchestrator import ModelOrchestrator
from .training.trainer import MultiModalTrainer
from .distillation.distiller import KnowledgeDistiller

__all__ = [
    "HybridVideoAI",
    "ModelOrchestrator", 
    "MultiModalTrainer",
    "KnowledgeDistiller",
]
