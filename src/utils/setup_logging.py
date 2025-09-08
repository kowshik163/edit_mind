"""
Logging setup utility
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(level: int = logging.INFO, 
                  log_file: str = None,
                  format_str: str = None) -> None:
    """
    Setup logging configuration
    
    Args:
        level: Logging level
        log_file: Optional log file path
        format_str: Custom format string
    """
    
    # Default format
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create formatter
    formatter = logging.Formatter(format_str)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def get_experiment_logger(experiment_name: str, 
                         log_dir: str = "logs") -> logging.Logger:
    """
    Get a logger for a specific experiment
    
    Args:
        experiment_name: Name of the experiment
        log_dir: Directory to store logs
        
    Returns:
        Logger instance
    """
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{experiment_name}_{timestamp}.log"
    
    # Setup logging
    setup_logging(log_file=str(log_file))
    
    # Return experiment-specific logger
    return logging.getLogger(experiment_name)
