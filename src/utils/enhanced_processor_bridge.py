"""
Enhanced Dataset Processor Bridge
Provides fallback methods for enhanced dataset processing functionality
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class EnhancedProcessorBridge:
    """Bridge class to provide enhanced dataset processing with fallbacks"""
    
    def __init__(self, downloader_instance):
        self.downloader = downloader_instance
        self._init_enhanced_processors()
    
    def _init_enhanced_processors(self):
        """Initialize enhanced dataset processors"""
        try:
            from .enhanced_dataset_processors import EnhancedDatasetProcessors
            enhanced = EnhancedDatasetProcessors()
            
            # Bind enhanced processors as methods
            self.process_ave_dataset = enhanced._process_ave_dataset.__get__(self.downloader, type(self.downloader))
            self.process_v3c1 = enhanced._process_v3c1.__get__(self.downloader, type(self.downloader))
            self.process_reddit_editors = enhanced._process_reddit_editors.__get__(self.downloader, type(self.downloader))
            self.process_youtube_tutorials = enhanced._process_youtube_tutorials.__get__(self.downloader, type(self.downloader))
            self.process_video_effects_code = enhanced._process_video_effects_code.__get__(self.downloader, type(self.downloader))
            self.process_kaggle_datasets = enhanced._process_kaggle_datasets.__get__(self.downloader, type(self.downloader))
            self.process_professional_editing = enhanced._process_professional_editing.__get__(self.downloader, type(self.downloader))
            
            logger.info("✅ Enhanced dataset processors loaded successfully")
            
        except ImportError as e:
            logger.warning(f"Enhanced processors not available, using fallbacks: {e}")
            # Use fallback processors
            self.process_ave_dataset = self._process_fallback
            self.process_v3c1 = self._process_fallback
            self.process_reddit_editors = self._process_fallback
            self.process_youtube_tutorials = self._process_fallback
            self.process_video_effects_code = self._process_fallback
            self.process_kaggle_datasets = self._process_kaggle_datasets_fallback
            self.process_professional_editing = self._process_professional_editing_fallback
    
    def _process_fallback(self, name: str, files: List[str], output_dir: Path, config: Dict) -> Dict[str, Any]:
        """Fallback processor for missing enhanced processors"""
        logger.warning(f"Using fallback processor for {name}")
        
        # Create minimal samples
        all_samples = []
        for i in range(min(config["sample_limit"], 100)):
            sample = {
                "sample_id": f"{name}_{i:04d}",
                "source": name,
                "generated": True,
                "fallback": True
            }
            all_samples.append(sample)
        
        samples_file = output_dir / "samples.json"
        with open(samples_file, 'w') as f:
            json.dump(all_samples, f, indent=2)
        
        return {
            "dataset": name,
            "samples": len(all_samples),
            "files": len(files),
            "samples_file": str(samples_file),
            "fallback": True
        }
    
    def _process_kaggle_datasets_fallback(self, name: str, files: List[str], output_dir: Path, config: Dict) -> Dict[str, Any]:
        """Fallback Kaggle dataset processor"""
        logger.info(f"    🔄 Processing Kaggle datasets (fallback)...")
        
        # Generate synthetic camera data
        camera_movements = ["pan", "tilt", "zoom", "dolly", "crane", "steadicam"]
        angles = ["wide", "medium", "close-up", "extreme-close-up", "bird's-eye", "low-angle"]
        
        all_samples = []
        for i in range(config["sample_limit"]):
            sample = {
                "sample_id": f"kaggle_{i:06d}",
                "camera_movement": camera_movements[i % len(camera_movements)],
                "camera_angle": angles[i % len(angles)],
                "scene_type": ["indoor", "outdoor", "studio"][i % 3],
                "lighting": ["natural", "artificial", "mixed"][i % 3],
                "quality_score": 0.5 + (i % 50) / 100.0,
                "source": "kaggle",
                "fallback": True
            }
            all_samples.append(sample)
        
        samples_file = output_dir / "samples.json"
        with open(samples_file, 'w') as f:
            json.dump(all_samples, f, indent=2)
        
        return {
            "dataset": name,
            "samples": len(all_samples),
            "camera_techniques": len(set(s["camera_movement"] for s in all_samples)),
            "samples_file": str(samples_file),
            "fallback": True
        }
    
    def _process_professional_editing_fallback(self, name: str, files: List[str], output_dir: Path, config: Dict) -> Dict[str, Any]:
        """Fallback professional editing processor"""
        logger.info(f"    🔄 Processing professional editing patterns (fallback)...")
        
        editing_patterns = [
            {"pattern_name": "match_cut", "description": "Cut that matches object or movement across scenes"},
            {"pattern_name": "j_cut", "description": "Audio continues from previous shot while video cuts to new shot"},
            {"pattern_name": "l_cut", "description": "Video continues while audio cuts to new source"},
            {"pattern_name": "cutaway", "description": "Brief shot of something other than main action"},
            {"pattern_name": "montage", "description": "Series of shots edited together to condense time"},
            {"pattern_name": "jump_cut", "description": "Cut between sequential shots of the same subject"},
            {"pattern_name": "cross_cut", "description": "Alternating between two or more scenes"},
            {"pattern_name": "fade_in_out", "description": "Gradual transition from/to black"},
            {"pattern_name": "dissolve", "description": "Gradual transition between two shots"},
            {"pattern_name": "wipe", "description": "One shot replaces another with a geometric pattern"}
        ]
        
        all_samples = []
        for i in range(config["sample_limit"]):
            pattern = editing_patterns[i % len(editing_patterns)]
            sample = {
                "pattern_id": f"pattern_{i:04d}",
                "name": pattern["pattern_name"],
                "description": pattern["description"],
                "category": "professional_technique",
                "difficulty": ["beginner", "intermediate", "advanced"][i % 3],
                "source": "professional_editing",
                "fallback": True
            }
            all_samples.append(sample)
        
        samples_file = output_dir / "samples.json"
        with open(samples_file, 'w') as f:
            json.dump(all_samples, f, indent=2)
        
        return {
            "dataset": name,
            "samples": len(all_samples),
            "technique_types": len(set(s["name"] for s in all_samples)),
            "samples_file": str(samples_file),
            "fallback": True
        }