"""
Knowledge distillation modules
"""

from .distiller import KnowledgeDistiller, ProgressiveDistillation

__all__ = [
    "KnowledgeDistiller",
    "ProgressiveDistillation"
]
