"""
Training modules and utilities
"""

try:
    from .trainer import MultiModalTrainer
    __all__ = ["MultiModalTrainer"]
except ImportError:
    # Handle gracefully when running modules directly
    __all__ = []
