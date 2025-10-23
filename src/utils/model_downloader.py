"""
Model Auto-Downloader - Automatically downloads and caches all required models
"""

import os
import logging
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from omegaconf import DictConfig
from transformers import (
    AutoTokenizer, AutoModel, AutoProcessor,
    LlamaForCausalLM, CLIPModel, CLIPProcessor,
    WhisperModel, WhisperProcessor, WhisperForConditionalGeneration,
    pipeline
)
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ModelDownloader:
    """
    Automatically downloads and caches all required models for the video editor
    """
    
    def __init__(self, cache_dir: str = "models/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configurations with fallbacks
        self.model_configs = {
            "reasoning": {
                "primary": "codellama/CodeLlama-7b-Instruct-hf",
                "fallbacks": [
                    "microsoft/DialoGPT-medium",
                    "microsoft/DialoGPT-small"
                ],
                "type": "llm"
            },
            "vision": {
                "primary": "openai/clip-vit-large-patch14",
                "fallbacks": [
                    "openai/clip-vit-base-patch32",
                    "openai/clip-vit-base-patch16"
                ],
                "type": "vision"
            },
            "audio": {
                "primary": "openai/whisper-large-v2", 
                "fallbacks": [
                    "openai/whisper-base",
                    "openai/whisper-tiny"
                ],
                "type": "audio"
            },
            "vision_advanced": {
                "primary": "google/siglip-base-patch16-224",
                "fallbacks": [
                    "openai/clip-vit-base-patch32"
                ],
                "type": "vision"
            },
            "audio_speech": {
                "primary": "distil-whisper/distil-large-v2",
                "fallbacks": [
                    "openai/whisper-base"
                ],
                "type": "audio"
            },
            # New advanced video generation models
            "wan_video": {
                "primary": "alibaba-pai/wan-2.2",
                "fallbacks": [
                    "alibaba-pai/wan-2.1", 
                    "alibaba-pai/wan-2.0"
                ],
                "type": "video_generation",
                "description": "Alibaba's Wan series video generation models"
            },
            "mochi_video": {
                "primary": "genmo/mochi-1", 
                "fallbacks": [
                    "genmo/mochi-preview"
                ],
                "type": "video_generation",
                "description": "Genmo's Mochi video generation model"
            },
            "ltx_video": {
                "primary": "lightricks/ltx-video",
                "fallbacks": [
                    "lightricks/ltx-video-base"
                ],
                "type": "video_generation", 
                "description": "Lightricks LTX-Video model"
            },
            "hunyuan_video": {
                "primary": "tencent/hunyuan-video",
                "fallbacks": [
                    "tencent/hunyuan-video-base"
                ],
                "type": "video_generation",
                "description": "Tencent HunyuanVideo model"
            },
            "videocrafter": {
                "primary": "VideoCrafter/VideoCrafter1",
                "fallbacks": [
                    "VideoCrafter/VideoCrafter-base"
                ],
                "type": "video_generation",
                "description": "VideoCrafter video generation model"
            }
        }
        
        self.downloaded_models = {}
        
    def download_all_models(self, force_download: bool = False) -> Dict[str, Any]:
        """Download all required models with fallback handling"""
        
        logger.info("🤖 Starting automatic model download...")
        
        results = {}
        
        for model_key, config in self.model_configs.items():
            logger.info(f"📥 Downloading {model_key} models...")
            
            model_info = self._download_model_with_fallback(
                model_key, config, force_download
            )
            
            if model_info:
                results[model_key] = model_info
                logger.info(f"✅ {model_key}: {model_info['model_name']}")
            else:
                logger.error(f"❌ Failed to download {model_key}")
                
        logger.info(f"🎉 Model download complete: {len(results)}/{len(self.model_configs)} successful")
        
        # Save download manifest
        self._save_manifest(results)
        
        return results
    
    def _download_model_with_fallback(self, model_key: str, config: Dict[str, Any], 
                                    force_download: bool = False) -> Optional[Dict[str, Any]]:
        """Download a model with fallback options"""
        
        models_to_try = [config["primary"]] + config.get("fallbacks", [])
        
        for model_name in models_to_try:
            try:
                logger.info(f"  🔄 Trying {model_name}...")
                
                # Check if already cached
                if not force_download and self._is_model_cached(model_name):
                    logger.info(f"  📋 Using cached {model_name}")
                    return {
                        "model_name": model_name,
                        "type": config["type"],
                        "status": "cached",
                        "cache_path": str(self.cache_dir / model_name.replace("/", "_"))
                    }
                
                # Download model based on type
                if config["type"] == "llm":
                    model, tokenizer = self._download_llm(model_name)
                elif config["type"] == "vision":
                    model, processor = self._download_vision_model(model_name)
                elif config["type"] == "audio":
                    model, processor = self._download_audio_model(model_name)
                else:
                    continue
                
                # Cache the model
                cache_path = self._cache_model(model_name, model, tokenizer if config["type"] == "llm" else processor)
                
                return {
                    "model_name": model_name,
                    "type": config["type"],
                    "status": "downloaded",
                    "cache_path": cache_path,
                    "model": model,
                    "processor": tokenizer if config["type"] == "llm" else processor
                }
                
            except Exception as e:
                logger.warning(f"  ⚠️ Failed to download {model_name}: {e}")
                continue
        
        return None
    
    def _download_llm(self, model_name: str):
        """Download language model"""
        logger.info(f"    📚 Downloading LLM: {model_name}")
        
        # Determine dtype based on device availability
        model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        # Try LlamaForCausalLM first, fallback to AutoModel
        try:
            model = LlamaForCausalLM.from_pretrained(
                model_name,
                torch_dtype=model_dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                cache_dir=str(self.cache_dir)
            )
        except Exception as e:
            logger.info(f"    🔄 LlamaForCausalLM failed, trying AutoModel: {e}")
            model = AutoModel.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir)
            )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(self.cache_dir)
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer
    
    def _download_vision_model(self, model_name: str):
        """Download vision model"""
        logger.info(f"    👁️ Downloading Vision: {model_name}")
        
        model = CLIPModel.from_pretrained(
            model_name,
            cache_dir=str(self.cache_dir)
        )
        
        processor = CLIPProcessor.from_pretrained(
            model_name,
            cache_dir=str(self.cache_dir)
        )
        
        return model, processor
    
    def _download_audio_model(self, model_name: str):
        """Download audio model"""
        logger.info(f"    🎵 Downloading Audio: {model_name}")
        
        # Try WhisperForConditionalGeneration first
        try:
            model = WhisperForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir)
            )
        except Exception as e:
            logger.info(f"    🔄 WhisperForConditionalGeneration failed, trying WhisperModel: {e}")
            model = WhisperModel.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir)
            )
        
        processor = WhisperProcessor.from_pretrained(
            model_name,
            cache_dir=str(self.cache_dir)
        )
        
        return model, processor
    
    def _is_model_cached(self, model_name: str) -> bool:
        """Check if model is already cached"""
        cache_path = self.cache_dir / model_name.replace("/", "_")
        return cache_path.exists() and len(list(cache_path.glob("*"))) > 0
    
    def _cache_model(self, model_name: str, model, processor) -> str:
        """Cache model locally"""
        cache_path = self.cache_dir / model_name.replace("/", "_")
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Save model and processor
        try:
            model.save_pretrained(str(cache_path / "model"))
            processor.save_pretrained(str(cache_path / "processor"))
            
            # Save metadata
            metadata = {
                "model_name": model_name,
                "cached_at": str(Path().resolve()),
                "model_class": str(type(model)),
                "processor_class": str(type(processor))
            }
            
            import json
            with open(cache_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to cache {model_name}: {e}")
        
        return str(cache_path)
    
    def _save_manifest(self, results: Dict[str, Any]):
        """Save download manifest"""
        manifest = {
            "downloaded_models": results,
            "cache_dir": str(self.cache_dir),
            "total_models": len(results)
        }
        
        import json
        with open(self.cache_dir / "download_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
    
    def get_model(self, model_key: str) -> Optional[Dict[str, Any]]:
        """Get a downloaded model"""
        if model_key not in self.downloaded_models:
            # Try to load from cache
            manifest_path = self.cache_dir / "download_manifest.json"
            if manifest_path.exists():
                import json
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                    if model_key in manifest.get("downloaded_models", {}):
                        return manifest["downloaded_models"][model_key]
        
        return self.downloaded_models.get(model_key)
    
    def cleanup_cache(self, keep_recent: int = 2):
        """Clean up old cached models"""
        logger.info(f"🧹 Cleaning up model cache, keeping {keep_recent} most recent")
        
        # Implementation for cache cleanup
        cached_dirs = [d for d in self.cache_dir.iterdir() if d.is_dir()]
        
        if len(cached_dirs) > keep_recent:
            # Sort by modification time
            cached_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            for old_dir in cached_dirs[keep_recent:]:
                try:
                    import shutil
                    shutil.rmtree(old_dir)
                    logger.info(f"🗑️ Removed old cache: {old_dir.name}")
                except Exception as e:
                    logger.warning(f"Failed to remove {old_dir}: {e}")


def auto_download_models(config: Optional[DictConfig] = None, 
                        force_download: bool = False) -> Dict[str, Any]:
    """
    Convenience function to auto-download all models
    """
    cache_dir = "models/cache"
    if config and hasattr(config, 'model_cache_dir'):
        cache_dir = config.model_cache_dir
    
    downloader = ModelDownloader(cache_dir)
    return downloader.download_all_models(force_download)


if __name__ == "__main__":
    # CLI usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Download all required models")
    parser.add_argument("--cache-dir", default="models/cache", help="Model cache directory")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--cleanup", action="store_true", help="Cleanup old caches")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    downloader = ModelDownloader(args.cache_dir)
    
    if args.cleanup:
        downloader.cleanup_cache()
    
    results = downloader.download_all_models(args.force)
    
    print(f"\n🎉 Download complete!")
    print(f"📦 Downloaded {len(results)} model groups")
    print(f"💾 Cache location: {args.cache_dir}")
