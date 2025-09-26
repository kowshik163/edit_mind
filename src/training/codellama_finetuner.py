"""
CodeLLaMA Fine-tuning Module for Video Effects
Specialized fine-tuning on video effect code datasets (FFmpeg, GLSL, MoviePy)
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset as HFDataset
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class VideoEffectsFinetuningConfig:
    """Configuration for CodeLLaMA fine-tuning on video effects"""
    
    # Model configuration
    model_name: str = "codellama/CodeLlama-7b-Python-hf"
    max_length: int = 1024
    
    # Training configuration
    output_dir: str = "checkpoints/codellama-video-effects"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    
    # Data configuration
    train_split: float = 0.8
    validation_split: float = 0.1
    test_split: float = 0.1
    
    # Effect types to include
    effect_types: List[str] = None
    
    def __post_init__(self):
        if self.effect_types is None:
            self.effect_types = [
                "ffmpeg", "moviepy", "opencv", "glsl", "shader",
                "transition", "filter", "effect", "composite"
            ]


class VideoEffectsDataset:
    """Dataset handler for video effects code fine-tuning"""
    
    def __init__(self, data_dir: str, tokenizer, config: VideoEffectsFinetuningConfig):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.config = config
        self.samples = []
        
    def load_effects_data(self) -> List[Dict[str, Any]]:
        """Load video effects code from various sources"""
        effects_data = []
        
        # Load from video_effects_scripts dataset
        effects_dir = self.data_dir / "video_effects_scripts"
        if effects_dir.exists():
            effects_data.extend(self._load_effects_scripts(effects_dir))
        
        # Load from professional editing patterns
        patterns_dir = self.data_dir / "professional_editing"
        if patterns_dir.exists():
            effects_data.extend(self._load_editing_patterns(patterns_dir))
        
        # Load curated effect examples
        examples_dir = self.data_dir / "effect_examples"
        if examples_dir.exists():
            effects_data.extend(self._load_curated_examples(examples_dir))
        
        logger.info(f"Loaded {len(effects_data)} video effects samples")
        return effects_data
    
    def _load_effects_scripts(self, effects_dir: Path) -> List[Dict[str, Any]]:
        """Load effects from video_effects_scripts dataset"""
        effects = []
        
        try:
            samples_file = effects_dir / "samples.json"
            if samples_file.exists():
                with open(samples_file) as f:
                    samples = json.load(f)
                
                for sample in samples:
                    if len(sample.get("code", "")) > 20:  # Filter out trivial code
                        effects.append({
                            "prompt": sample.get("description", ""),
                            "code": sample.get("code", ""),
                            "category": sample.get("category", "general"),
                            "language": sample.get("language", "python"),
                            "source": "video_effects_scripts"
                        })
        
        except Exception as e:
            logger.warning(f"Failed to load effects scripts: {e}")
        
        return effects
    
    def _load_editing_patterns(self, patterns_dir: Path) -> List[Dict[str, Any]]:
        """Load professional editing patterns"""
        patterns = []
        
        try:
            samples_file = patterns_dir / "samples.json"
            if samples_file.exists():
                with open(samples_file) as f:
                    samples = json.load(f)
                
                for sample in samples:
                    # Generate code examples for editing patterns
                    pattern_name = sample.get("name", "")
                    description = sample.get("description", "")
                    
                    if pattern_name and description:
                        code_example = self._generate_pattern_code(pattern_name, description)
                        patterns.append({
                            "prompt": f"Create a {pattern_name} effect: {description}",
                            "code": code_example,
                            "category": "editing_pattern",
                            "language": "python",
                            "source": "professional_editing"
                        })
        
        except Exception as e:
            logger.warning(f"Failed to load editing patterns: {e}")
        
        return patterns
    
    def _generate_pattern_code(self, pattern_name: str, description: str) -> str:
        """Generate code examples for editing patterns"""
        
        pattern_templates = {
            "match_cut": '''
# Match cut effect - matching object/movement across scenes
def create_match_cut(clip1, clip2, match_object="hand"):
    """Create a match cut between two clips"""
    import cv2
    import numpy as np
    
    # Find matching frames
    frame1 = clip1.get_frame(clip1.duration - 0.1)
    frame2 = clip2.get_frame(0.1)
    
    # Simple position-based matching
    cut_point = clip1.duration
    
    return concatenate_videoclips([clip1, clip2])
''',
            
            "j_cut": '''
# J-cut effect - audio continues from previous shot
def create_j_cut(clip1, clip2, overlap_duration=1.0):
    """Create a J-cut transition"""
    from moviepy.editor import CompositeVideoClip
    
    # Extract audio from first clip
    audio1 = clip1.audio
    
    # Create video transition
    video_cut = clip1.duration - overlap_duration
    video_part = clip1.subclip(0, video_cut)
    
    # Composite: video from clip1, then clip2 with extended audio
    result = concatenate_videoclips([video_part, clip2])
    final_audio = CompositeAudioClip([
        audio1.subclip(0, video_cut + overlap_duration),
        clip2.audio.set_start(video_cut)
    ])
    
    return result.set_audio(final_audio)
''',
            
            "fade_in_out": '''
# Fade in/out effect
def create_fade_transition(clip, fade_duration=1.0):
    """Create fade in/out effect"""
    from moviepy.editor import *
    
    # Fade in
    clip = clip.fadein(fade_duration)
    
    # Fade out  
    clip = clip.fadeout(fade_duration)
    
    return clip
''',
            
            "zoom": '''
# Dynamic zoom effect
def create_zoom_effect(clip, zoom_factor=1.5, center=(0.5, 0.5)):
    """Create dynamic zoom effect"""
    import numpy as np
    
    def zoom_func(get_frame, t):
        frame = get_frame(t)
        h, w = frame.shape[:2]
        
        # Calculate zoom for this time
        progress = t / clip.duration
        current_zoom = 1 + (zoom_factor - 1) * progress
        
        # Calculate crop region
        crop_w = int(w / current_zoom)
        crop_h = int(h / current_zoom)
        
        center_x = int(center[0] * w)
        center_y = int(center[1] * h)
        
        x1 = max(0, center_x - crop_w // 2)
        y1 = max(0, center_y - crop_h // 2)
        x2 = min(w, x1 + crop_w)
        y2 = min(h, y1 + crop_h)
        
        # Crop and resize
        cropped = frame[y1:y2, x1:x2]
        return cv2.resize(cropped, (w, h))
    
    return clip.fl(zoom_func)
'''
        }
        
        # Return template or generate basic template
        return pattern_templates.get(pattern_name.lower(), f'''
# {pattern_name} effect - {description}
def create_{pattern_name.lower().replace(' ', '_')}_effect(clip):
    """{description}"""
    # Implementation for {pattern_name}
    return clip
''')
    
    def _load_curated_examples(self, examples_dir: Path) -> List[Dict[str, Any]]:
        """Load curated video effect examples"""
        examples = []
        
        # Define curated examples for different effect types
        curated_effects = [
            {
                "prompt": "Create a smooth pan effect across the video",
                "code": '''
def create_pan_effect(clip, pan_direction="right", pan_speed=50):
    """Create smooth panning effect"""
    from moviepy.editor import *
    
    def pan_func(get_frame, t):
        frame = get_frame(t)
        h, w = frame.shape[:2]
        
        # Calculate pan offset
        offset = int(pan_speed * t)
        if pan_direction == "right":
            offset = -offset
        
        # Create panned frame
        panned = np.zeros_like(frame)
        if pan_direction in ["left", "right"]:
            if offset > 0:
                panned[:, offset:] = frame[:, :w-offset]
            else:
                panned[:, :w+offset] = frame[:, -offset:]
        
        return panned
    
    return clip.fl(pan_func)
''',
                "category": "camera_movement",
                "language": "python"
            },
            
            {
                "prompt": "Apply color grading with cinematic look",
                "code": '''
def apply_cinematic_grading(clip, temperature=0.1, tint=0.05, contrast=1.2):
    """Apply cinematic color grading"""
    import cv2
    import numpy as np
    
    def color_grade(get_frame, t):
        frame = get_frame(t)
        
        # Convert to float
        frame = frame.astype(np.float32) / 255.0
        
        # Adjust temperature (blue/orange)
        frame[:, :, 0] *= (1 - temperature)  # Blue
        frame[:, :, 2] *= (1 + temperature)  # Red
        
        # Adjust tint (green/magenta)  
        frame[:, :, 1] *= (1 + tint)
        
        # Adjust contrast
        frame = (frame - 0.5) * contrast + 0.5
        
        # Clamp and convert back
        frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
        return frame
    
    return clip.fl(color_grade)
''',
                "category": "color_grading", 
                "language": "python"
            },
            
            {
                "prompt": "Create glitch effect using FFmpeg",
                "code": '''
# FFmpeg command for glitch effect
ffmpeg_command = """
ffmpeg -i input.mp4 \\
  -vf "noise=alls=20:allf=t+u,eq=contrast=1.5:brightness=0.1" \\
  -c:v libx264 -crf 23 \\
  output_glitch.mp4
"""

def create_glitch_effect_ffmpeg(input_path, output_path):
    """Create glitch effect using FFmpeg"""
    import subprocess
    
    cmd = [
        "ffmpeg", "-i", input_path,
        "-vf", "noise=alls=20:allf=t+u,eq=contrast=1.5:brightness=0.1",
        "-c:v", "libx264", "-crf", "23",
        output_path
    ]
    
    subprocess.run(cmd, check=True)
    return output_path
''',
                "category": "glitch",
                "language": "ffmpeg"
            }
        ]
        
        for effect in curated_effects:
            effect["source"] = "curated_examples"
            examples.append(effect)
        
        return examples
    
    def prepare_training_data(self) -> Tuple[HFDataset, HFDataset]:
        """Prepare tokenized datasets for training"""
        
        # Load all effects data
        effects_data = self.load_effects_data()
        
        if not effects_data:
            raise ValueError("No effects data found for training")
        
        # Create training examples
        training_examples = []
        for effect in effects_data:
            # Create instruction-following format
            prompt = effect["prompt"]
            code = effect["code"]
            
            # Format as instruction-response pair
            instruction = f"### Instruction:\nCreate video effect: {prompt}\n\n### Response:\n{code}\n### End"
            
            training_examples.append({"text": instruction})
        
        # Convert to HuggingFace dataset
        dataset = HFDataset.from_list(training_examples)
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.config.max_length
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Split into train/validation
        train_size = int(self.config.train_split * len(tokenized_dataset))
        splits = tokenized_dataset.train_test_split(
            train_size=train_size,
            test_size=len(tokenized_dataset) - train_size
        )
        
        return splits["train"], splits["test"]


class CodeLLaMAVideoEffectsFinetuner:
    """Fine-tuner for CodeLLaMA on video effects datasets"""
    
    def __init__(self, config: VideoEffectsFinetuningConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer"""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        logger.info("Model and tokenizer loaded successfully")
    
    def fine_tune(self, data_dir: str) -> str:
        """Fine-tune CodeLLaMA on video effects data"""
        
        # Setup model
        self.setup_model_and_tokenizer()
        
        # Prepare data
        dataset_handler = VideoEffectsDataset(data_dir, self.tokenizer, self.config)
        train_dataset, eval_dataset = dataset_handler.prepare_training_data()
        
        logger.info(f"Training on {len(train_dataset)} examples, evaluating on {len(eval_dataset)}")
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=100,
            save_steps=500,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  # Disable wandb/tensorboard
            dataloader_pin_memory=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Fine-tune
        logger.info("Starting fine-tuning...")
        trainer.train()
        
        # Save final model
        final_model_path = os.path.join(self.config.output_dir, "final_model")
        trainer.save_model(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        
        logger.info(f"Fine-tuning completed. Model saved to: {final_model_path}")
        return final_model_path
    
    def evaluate_model(self, model_path: str, test_prompts: List[str]) -> Dict[str, Any]:
        """Evaluate the fine-tuned model on test prompts"""
        
        # Load fine-tuned model
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        results = []
        
        for prompt in test_prompts:
            # Format prompt
            formatted_prompt = f"### Instruction:\nCreate video effect: {prompt}\n\n### Response:\n"
            
            # Generate
            inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(formatted_prompt):].strip()
            
            results.append({
                "prompt": prompt,
                "generated_code": response
            })
        
        return {"test_results": results}


def create_video_effects_training_data(output_dir: str):
    """Create comprehensive training dataset for video effects"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create effect examples directory
    examples_dir = output_path / "effect_examples"
    examples_dir.mkdir(exist_ok=True)
    
    # This will be populated by the video_effects_scripts dataset downloader
    logger.info(f"Video effects training data directory created: {output_dir}")
    logger.info("Run dataset downloader with 'video_effects_scripts' to populate training data")


if __name__ == "__main__":
    # Example usage
    config = VideoEffectsFinetuningConfig(
        num_train_epochs=2,
        per_device_train_batch_size=2,
        learning_rate=1e-5
    )
    
    finetuner = CodeLLaMAVideoEffectsFinetuner(config)
    
    # Create training data directory
    data_dir = "data/datasets"
    create_video_effects_training_data(data_dir)
    
    # Fine-tune (requires actual video effects data)
    try:
        model_path = finetuner.fine_tune(data_dir)
        
        # Test the model
        test_prompts = [
            "Create a smooth zoom-in effect",
            "Add a vintage film grain filter",
            "Generate a color pop effect"
        ]
        
        results = finetuner.evaluate_model(model_path, test_prompts)
        print("Fine-tuning completed successfully!")
        print(f"Test results: {results}")
        
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")