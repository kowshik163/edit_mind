"""
Model Orchestrator - Coordinates different AI components
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ModelOrchestrator:
    """
    Orchestrates multiple AI models and components
    for the autonomous video editing pipeline
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.components = {}
        
    def register_component(self, name: str, component: Any):
        """Register a component with the orchestrator"""
        self.components[name] = component
        logger.info(f"Registered component: {name}")
        
    def get_component(self, name: str):
        """Get a registered component"""
        return self.components.get(name)
        
    def orchestrate_editing(self, video_path: str, prompt: str) -> str:
        """
        Orchestrate the full editing pipeline
        
        Args:
            video_path: Path to input video
            prompt: Editing instruction
            
        Returns:
            Path to edited video
        """
        
        logger.info(f"🎬 Orchestrating edit for: {video_path}")
        logger.info(f"💭 Prompt: {prompt}")
        
        try:
            # 1. Load and preprocess video
            vision_processor = self.components.get('vision_processor')
            audio_processor = self.components.get('audio_processor')
            
            if vision_processor and audio_processor:
                # Process video and audio
                frames = vision_processor.load_video(video_path)
                audio_data = audio_processor.load_audio(video_path)
                
                logger.info(f"📹 Loaded {len(frames)} frames")
                logger.info(f"🎵 Loaded {len(audio_data)} audio samples")
                
                # 2. Analyze content
                vision_features = vision_processor.analyze_scene(frames)
                audio_features = audio_processor._extract_audio_features(audio_data, 16000)
                
                # 3. Generate editing decisions using hybrid AI
                hybrid_ai = self.components.get('hybrid_ai')
                if hybrid_ai:
                    editing_decisions = hybrid_ai.autonomous_edit(
                        video_frames=frames,
                        audio_features=audio_features, 
                        editing_prompt=prompt
                    )
                    logger.info(f"🧠 Generated {len(editing_decisions.get('cuts', []))} editing decisions")
                
                # 4. Generate timeline
                timeline_generator = self.components.get('timeline_generator')
                if timeline_generator:
                    timeline = timeline_generator.generate_timeline(
                        frames=frames,
                        audio=audio_data,
                        decisions=editing_decisions,
                        prompt=prompt
                    )
                    
                    # 5. Render final video
                    output_path = timeline_generator.render_video(
                        timeline, 
                        output_path=f"orchestrated_{Path(video_path).stem}.mp4"
                    )
                    
                    logger.info(f"✅ Orchestration complete: {output_path}")
                    return output_path
            
            # Fallback if components not available
            logger.warning("Components not fully initialized, using fallback")
            output_path = f"orchestrated_{Path(video_path).stem}.mp4"
            
            # Create a simple copy as fallback
            import shutil
            shutil.copy2(video_path, output_path)
            logger.info(f"⚠️  Fallback orchestration: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Orchestration failed: {e}")
            # Return input path as ultimate fallback
            return video_path
