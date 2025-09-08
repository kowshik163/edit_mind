"""
Configuration loader utility
"""

import yaml
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Union


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        DictConfig object with loaded configuration
    """
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load YAML config
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert to OmegaConf DictConfig
    config = OmegaConf.create(config_dict)
    
    # Resolve any variable interpolations
    config = OmegaConf.to_container(config, resolve=True)
    config = OmegaConf.create(config)
    
    return config


def save_config(config: DictConfig, save_path: Union[str, Path]):
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration to save
        save_path: Where to save the configuration
    """
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to regular dict for YAML serialization
    config_dict = OmegaConf.to_container(config, resolve=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)


def merge_configs(*configs: DictConfig) -> DictConfig:
    """
    Merge multiple configurations
    
    Args:
        *configs: Configuration objects to merge
        
    Returns:
        Merged configuration
    """
    
    merged = configs[0]
    
    for config in configs[1:]:
        merged = OmegaConf.merge(merged, config)
    
    return merged


def update_config_from_args(config: DictConfig, args: dict) -> DictConfig:
    """
    Update configuration with command line arguments
    
    Args:
        config: Base configuration
        args: Dictionary of arguments to override
        
    Returns:
        Updated configuration
    """
    
    # Create config from args
    args_config = OmegaConf.create(args)
    
    # Merge with base config (args take precedence)
    updated_config = OmegaConf.merge(config, args_config)
    
    return updated_config
