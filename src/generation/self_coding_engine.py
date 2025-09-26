"""
Self-Coding Video Effects Engine - Dynamic effect generation using CodeLLaMA
"""

import os
import ast
import sys
import json
import logging
import tempfile
import subprocess
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, CodeLlamaTokenizer
from omegaconf import DictConfig
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SafeCodeExecutor:
    """Enhanced safe code execution environment with multi-step capabilities"""
    
    def __init__(self, timeout: int = 30):
        self.allowed_modules = {
            'numpy', 'cv2', 'PIL', 'moviepy.editor',
            'moviepy.video.fx', 'moviepy.audio.fx',
            'math', 'random', 'json', 'os.path',
            'subprocess', 'tempfile', 'shutil',
            'matplotlib.pyplot', 'scipy', 'skimage'
        }
        
        # For validation
        self.allowed_imports = self.allowed_modules
        self.forbidden_calls = {
            'eval', 'exec', 'compile', 'open', '__import__',
            'globals', 'locals', 'vars', 'dir', 'getattr', 'setattr'
        }
        self.restricted_functions = self.forbidden_calls  # Alias for backward compatibility
        
        # Multi-step execution context
        self.execution_context = {}
        self.temp_assets = []
        self.max_execution_time = timeout
        self.max_temp_files = 10
    
    def validate_code(self, code: str) -> tuple[bool, str]:
        """Validate that code is safe to execute"""
        
        try:
            # Parse AST to check for dangerous operations
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                # Check imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name not in self.allowed_imports:
                            return False, f"Forbidden import: {alias.name}"
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module not in self.allowed_imports:
                        return False, f"Forbidden module: {node.module}"
                
                # Check function calls
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in self.forbidden_calls:
                            return False, f"Forbidden function call: {node.func.id}"
                
                # Check attribute access
                elif isinstance(node, ast.Attribute):
                    if node.attr in self.forbidden_calls:
                        return False, f"Forbidden attribute access: {node.attr}"
            
            return True, "Code validation passed"
            
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def execute_effect(self, code: str, frame: np.ndarray, **kwargs) -> Optional[np.ndarray]:
        """Execute video effect code safely"""
        
        # Validate code first
        is_safe, message = self.validate_code(code)
        if not is_safe:
            logger.error(f"Code validation failed: {message}")
            return None
        
        try:
            # Create safe execution namespace
            safe_globals = {
                '__builtins__': {
                    'range': range, 'len': len, 'int': int, 'float': float,
                    'str': str, 'bool': bool, 'list': list, 'dict': dict,
                    'tuple': tuple, 'set': set, 'abs': abs, 'min': min, 'max': max,
                    'round': round, 'sum': sum, 'zip': zip, 'enumerate': enumerate
                },
                'cv2': cv2,
                'np': np,
                'numpy': np,
                'math': __import__('math'),
                'random': __import__('random'),
                'frame': frame,
                **kwargs
            }
            
            # Execute code
            exec(code, safe_globals)
            
            # Look for result function or variable
            if 'result' in safe_globals:
                return safe_globals['result']
            elif 'processed_frame' in safe_globals:
                return safe_globals['processed_frame']
            else:
                # Try to find a function and call it
                for name, obj in safe_globals.items():
                    if callable(obj) and not name.startswith('_'):
                        try:
                            return obj(frame, **kwargs)
                        except:
                            continue
            
            logger.warning("No result found in executed code")
            return None
            
        except Exception as e:
            logger.error(f"Code execution failed: {e}")
            return None
    
    def execute_safe(self, code: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute code safely with restricted environment and multi-step support"""
        
        is_safe, message = self.validate_code(code)
        if not is_safe:
            raise ValueError(f"Code validation failed: {message}")
        
        # Create safe execution environment
        safe_globals = self._create_safe_globals()
        
        # Add persistent execution context
        safe_globals.update(self.execution_context)
        
        # Add context if provided
        if context:
            safe_globals.update(context)
        
        try:
            # Execute with timeout
            result = self._execute_with_timeout(code, safe_globals)
            
            # Update persistent context with new variables
            self._update_execution_context(safe_globals)
            
            return result
            
        except Exception as e:
            logger.error(f"Code execution failed: {e}")
            raise RuntimeError(f"Execution failed: {e}")
    
    def _create_safe_globals(self) -> Dict[str, Any]:
        """Create safe global execution environment"""
        safe_globals = {
            '__builtins__': {
                'len': len, 'str': str, 'int': int, 'float': float,
                'list': list, 'dict': dict, 'tuple': tuple,
                'max': max, 'min': min, 'sum': sum, 'abs': abs,
                'round': round, 'range': range, 'enumerate': enumerate,
                'zip': zip, 'print': print, 'type': type, 'isinstance': isinstance
            }
        }
        
        # Add allowed modules
        for module in self.allowed_modules:
            try:
                if '.' in module:
                    # Handle submodules
                    parts = module.split('.')
                    imported = __import__(module, fromlist=[parts[-1]])
                    safe_globals[parts[-1]] = imported
                else:
                    safe_globals[module] = __import__(module)
            except ImportError:
                logger.warning(f"Module {module} not available")
                continue
        
        # Add utility functions for multi-step execution
        safe_globals.update({
            'create_temp_asset': self.create_temp_asset,
            'load_temp_asset': self.load_temp_asset,
            'execute_ffmpeg': self.execute_ffmpeg_safe,
            'create_intermediate_clip': self.create_intermediate_clip
        })
        
        return safe_globals
    
    def _execute_with_timeout(self, code: str, safe_globals: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code with timeout protection"""
        import threading
        
        result = {}
        exception = None
        
        def target():
            nonlocal result, exception
            try:
                exec(code, safe_globals)
                # Return any created variables (excluding built-ins)
                for key, value in safe_globals.items():
                    if (not key.startswith('__') and 
                        key not in self.allowed_modules and
                        key not in ['create_temp_asset', 'load_temp_asset', 'execute_ffmpeg', 'create_intermediate_clip']):
                        if callable(value) or key not in self.execution_context:
                            result[key] = value
            except Exception as e:
                exception = e
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout=self.max_execution_time)
        
        if thread.is_alive():
            raise RuntimeError(f"Code execution timed out after {self.max_execution_time} seconds")
        
        if exception:
            raise exception
        
        return result
    
    def _update_execution_context(self, safe_globals: Dict[str, Any]):
        """Update persistent execution context"""
        for key, value in safe_globals.items():
            if (not key.startswith('__') and 
                key not in self.allowed_modules and
                key not in ['create_temp_asset', 'load_temp_asset', 'execute_ffmpeg', 'create_intermediate_clip']):
                self.execution_context[key] = value
    
    def create_temp_asset(self, asset_type: str, data: Any = None) -> str:
        """Create temporary asset for multi-step processing"""
        import tempfile
        import uuid
        
        if len(self.temp_assets) >= self.max_temp_files:
            raise RuntimeError(f"Maximum temp files ({self.max_temp_files}) exceeded")
        
        asset_id = str(uuid.uuid4())[:8]
        
        if asset_type == "image":
            temp_file = tempfile.NamedTemporaryFile(suffix=f"_{asset_id}.png", delete=False)
            if data is not None and hasattr(data, 'save'):
                data.save(temp_file.name)
        elif asset_type == "video":
            temp_file = tempfile.NamedTemporaryFile(suffix=f"_{asset_id}.mp4", delete=False)
            if data is not None and hasattr(data, 'write_videofile'):
                data.write_videofile(temp_file.name, verbose=False, logger=None)
        elif asset_type == "audio":
            temp_file = tempfile.NamedTemporaryFile(suffix=f"_{asset_id}.wav", delete=False)
            if data is not None and hasattr(data, 'write_audiofile'):
                data.write_audiofile(temp_file.name, verbose=False, logger=None)
        else:
            temp_file = tempfile.NamedTemporaryFile(suffix=f"_{asset_id}.tmp", delete=False)
        
        temp_file.close()
        self.temp_assets.append(temp_file.name)
        
        logger.info(f"Created temp asset: {temp_file.name}")
        return temp_file.name
    
    def load_temp_asset(self, asset_path: str, asset_type: str = "auto"):
        """Load temporary asset for processing"""
        if asset_path not in self.temp_assets:
            raise ValueError("Asset not in temp assets list")
        
        if not os.path.exists(asset_path):
            raise RuntimeError(f"Temp asset not found: {asset_path}")
        
        try:
            if asset_type == "auto":
                # Auto-detect based on extension
                ext = os.path.splitext(asset_path)[1].lower()
                if ext in ['.png', '.jpg', '.jpeg']:
                    asset_type = "image"
                elif ext in ['.mp4', '.avi', '.mov']:
                    asset_type = "video"
                elif ext in ['.wav', '.mp3', '.aac']:
                    asset_type = "audio"
            
            if asset_type == "image":
                from PIL import Image
                return Image.open(asset_path)
            elif asset_type == "video":
                from moviepy.editor import VideoFileClip
                return VideoFileClip(asset_path)
            elif asset_type == "audio":
                from moviepy.editor import AudioFileClip
                return AudioFileClip(asset_path)
            else:
                # Return path for custom handling
                return asset_path
                
        except Exception as e:
            raise RuntimeError(f"Failed to load temp asset: {e}")
    
    def execute_ffmpeg_safe(self, command_args: list, input_file: str = None, output_file: str = None):
        """Execute FFmpeg command safely"""
        import subprocess
        
        # Validate command
        if not command_args or command_args[0] != "ffmpeg":
            raise ValueError("Only ffmpeg commands allowed")
        
        # Restrict to safe FFmpeg operations
        safe_filters = [
            'scale', 'crop', 'rotate', 'hflip', 'vflip',
            'brightness', 'contrast', 'saturation', 'hue',
            'fade', 'overlay', 'concat', 'fps', 'loop',
            'noise', 'blur', 'sharpen', 'eq', 'colorkey'
        ]
        
        command_str = ' '.join(command_args)
        has_safe_filter = any(f in command_str for f in safe_filters)
        
        if not has_safe_filter and '-vf' in command_args:
            raise ValueError("Unsafe FFmpeg filter detected")
        
        # Create output file if not specified
        if output_file is None:
            output_file = self.create_temp_asset("video")
        
        # Execute with timeout
        try:
            result = subprocess.run(
                command_args,
                timeout=self.max_execution_time,
                capture_output=True,
                text=True,
                check=True
            )
            return output_file
        except subprocess.TimeoutExpired:
            raise RuntimeError("FFmpeg command timed out")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg failed: {e.stderr}")
    
    def create_intermediate_clip(self, clip_data: Any, clip_type: str = "video"):
        """Create intermediate clip for complex multi-step processing"""
        if clip_type == "video" and hasattr(clip_data, 'write_videofile'):
            temp_path = self.create_temp_asset("video", clip_data)
            return self.load_temp_asset(temp_path, "video")
        elif clip_type == "audio" and hasattr(clip_data, 'write_audiofile'):
            temp_path = self.create_temp_asset("audio", clip_data)
            return self.load_temp_asset(temp_path, "audio")
        else:
            raise RuntimeError(f"Unsupported clip type: {clip_type}")
    
    def cleanup_temp_assets(self):
        """Clean up temporary assets"""
        import os
        
        for asset_path in self.temp_assets:
            try:
                if os.path.exists(asset_path):
                    os.unlink(asset_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp asset {asset_path}: {e}")
        
        self.temp_assets.clear()
    
    def reset_context(self):
        """Reset execution context and clean up"""
        self.execution_context.clear()
        self.cleanup_temp_assets()


class VideoEffectCodeGenerator:
    """Generate video effect code using CodeLLaMA"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Initialize CodeLLaMA model
        model_name = config.get('codellama_model', 'codellama/CodeLlama-7b-Python-hf')
        try:
            self.tokenizer = CodeLlamaTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                device_map="auto" if self.device.type == 'cuda' else None
            )
            
            if self.device.type != 'cuda':
                self.model = self.model.to(self.device)
                
            logger.info(f"CodeLLaMA model loaded: {model_name}")
            
        except Exception as e:
            logger.warning(f"Failed to load CodeLLaMA: {e}")
            self.model = None
            self.tokenizer = None
    
    def generate_effect_code(self, effect_description: str, 
                           reference_effects: List[Dict] = None) -> Optional[str]:
        """Generate video effect code from description"""
        
        if self.model is None:
            logger.warning("CodeLLaMA not available, using template fallback")
            return self._generate_template_code(effect_description)
        
        try:
            # Create prompt with examples and description
            prompt = self._create_code_generation_prompt(effect_description, reference_effects)
            
            # Tokenize and generate
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode generated code
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract code from generation
            code = self._extract_code_from_generation(generated_text, prompt)
            
            return code
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return self._generate_template_code(effect_description)
    
    def _create_code_generation_prompt(self, description: str, 
                                     reference_effects: List[Dict] = None) -> str:
        """Create prompt for code generation"""
        
        prompt = """# Video Effect Code Generator
# Task: Generate Python code for video effects using OpenCV and NumPy

# Example 1: Fade Effect
def fade_effect(frame, alpha=0.5):
    '''Apply fade effect to frame'''
    return cv2.addWeighted(frame, alpha, np.zeros_like(frame), 1-alpha, 0)

# Example 2: Blur Effect
def blur_effect(frame, blur_strength=15):
    '''Apply blur effect to frame'''
    return cv2.GaussianBlur(frame, (blur_strength, blur_strength), 0)

"""
        
        # Add reference effects if provided
        if reference_effects:
            prompt += "# Reference Effects:\n"
            for effect in reference_effects[:3]:  # Limit to 3 examples
                if 'code' in effect:
                    prompt += f"# {effect.get('name', 'Effect')}: {effect.get('description', '')}\n"
                    prompt += effect['code'] + "\n\n"
        
        prompt += f"""
# Task: Generate code for the following effect:
# Description: {description}
# Requirements:
# - Use cv2 and numpy (imported as np)
# - Function should take 'frame' as first parameter
# - Return processed frame as numpy array
# - Include docstring explaining the effect

def generated_effect(frame, **kwargs):
    '''
    {description}
    '''
"""
        
        return prompt
    
    def _extract_code_from_generation(self, generated_text: str, prompt: str) -> str:
        """Extract the generated code from the full generation"""
        
        # Remove the original prompt
        if prompt in generated_text:
            code_part = generated_text.replace(prompt, "").strip()
        else:
            code_part = generated_text.strip()
        
        # Extract function definition
        lines = code_part.split('\n')
        function_lines = []
        in_function = False
        
        for line in lines:
            if line.strip().startswith('def ') and 'frame' in line:
                in_function = True
                function_lines.append(line)
            elif in_function:
                if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                    # End of function
                    break
                function_lines.append(line)
        
        if function_lines:
            return '\n'.join(function_lines)
        else:
            # Fallback: return the whole code part
            return code_part
    
    def _generate_template_code(self, description: str) -> str:
        """Generate template code when CodeLLaMA is not available"""
        
        # Simple template-based generation
        effect_type = self._classify_effect_type(description.lower())
        
        templates = {
            'blur': '''
def generated_effect(frame, strength=15, **kwargs):
    """Apply blur effect to frame"""
    kernel_size = max(1, int(strength))
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
''',
            'color': '''
def generated_effect(frame, hue_shift=0, saturation=1.0, brightness=1.0, **kwargs):
    """Apply color adjustment to frame"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    hsv[:, :, 1] *= saturation
    hsv[:, :, 2] *= brightness
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
''',
            'distortion': '''
def generated_effect(frame, intensity=0.1, **kwargs):
    """Apply distortion effect to frame"""
    height, width = frame.shape[:2]
    map_x = np.zeros((height, width), dtype=np.float32)
    map_y = np.zeros((height, width), dtype=np.float32)
    
    for y in range(height):
        for x in range(width):
            map_x[y, x] = x + intensity * np.sin(y * 0.1)
            map_y[y, x] = y + intensity * np.cos(x * 0.1)
    
    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
''',
            'default': '''
def generated_effect(frame, **kwargs):
    """Apply custom effect to frame"""
    # Default: slight brightness adjustment
    return cv2.addWeighted(frame, 1.1, np.zeros_like(frame), 0, 10)
'''
        }
        
        return templates.get(effect_type, templates['default'])
    
    def _classify_effect_type(self, description: str) -> str:
        """Classify effect type from description"""
        
        if any(word in description for word in ['blur', 'smooth', 'soften']):
            return 'blur'
        elif any(word in description for word in ['color', 'hue', 'saturation', 'brightness']):
            return 'color'
        elif any(word in description for word in ['distort', 'wave', 'ripple', 'bend']):
            return 'distortion'
        else:
            return 'default'


class SelfCodingVideoEditor:
    """Main self-coding video editor that generates effects on demand"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.code_generator = VideoEffectCodeGenerator(config)
        self.code_executor = SafeCodeExecutor(timeout=config.get('execution_timeout', 30))
        
        # Load effect code database
        self.effect_database = self._load_effect_database()
        
        # Cache for generated effects
        self.generated_effects_cache = {}
        
        logger.info("Self-coding video editor initialized")
    
    def _load_effect_database(self) -> List[Dict]:
        """Load database of existing video effects code"""
        
        try:
            # Load from processed video effects dataset
            effects_file = Path(self.config.get('effects_database', 'data/datasets/video_effects_scripts/samples.json'))
            
            if effects_file.exists():
                with open(effects_file, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Effects database not found: {effects_file}")
                return []
                
        except Exception as e:
            logger.warning(f"Failed to load effects database: {e}")
            return []
    
    def generate_effect(self, description: str) -> str:
        """Generate effect code from description (for testing purposes)"""
        # Find relevant reference effects
        reference_effects = self._find_similar_effects(description)
        
        # Generate code
        generated_code = self.code_generator.generate_effect_code(description, reference_effects)
        
        if not generated_code:
            logger.warning(f"Failed to generate code for: {description}")
            # Return a basic fallback code
            return '''
def generated_effect(frame, **kwargs):
    """Fallback effect - basic brightness adjustment"""
    import cv2
    import numpy as np
    return cv2.addWeighted(frame, 1.1, np.zeros_like(frame), 0, 10)
'''
        
        return generated_code

    def create_custom_effect(self, description: str, test_frame: np.ndarray = None) -> Optional[Callable]:
        """Create a custom video effect from natural language description"""
        
        # Check cache first
        cache_key = description.lower().strip()
        if cache_key in self.generated_effects_cache:
            logger.info(f"Using cached effect for: {description}")
            return self.generated_effects_cache[cache_key]
        
        logger.info(f"Generating custom effect: {description}")
        
        # Generate code using the generate_effect method
        generated_code = self.generate_effect(description)
        
        # Test the generated effect
        if test_frame is not None:
            test_result = self.code_executor.execute_effect(generated_code, test_frame)
            if test_result is None:
                logger.error(f"Generated effect failed testing: {description}")
                return None
        
        # Create callable effect function
        def custom_effect(frame: np.ndarray, **kwargs) -> np.ndarray:
            result = self.code_executor.execute_effect(generated_code, frame, **kwargs)
            return result if result is not None else frame
        
        # Cache the effect
        self.generated_effects_cache[cache_key] = custom_effect
        
        # Save generated effect for future use
        self._save_generated_effect(description, generated_code)
        
        logger.info(f"Successfully created custom effect: {description}")
        return custom_effect
    
    def _find_similar_effects(self, description: str) -> List[Dict]:
        """Find similar effects in the database for reference"""
        
        description_words = set(description.lower().split())
        similar_effects = []
        
        for effect in self.effect_database:
            effect_words = set((effect.get('description', '') + ' ' + effect.get('name', '')).lower().split())
            
            # Calculate similarity
            overlap = len(description_words & effect_words)
            if overlap > 0:
                effect['similarity_score'] = overlap / len(description_words | effect_words)
                similar_effects.append(effect)
        
        # Sort by similarity and return top matches
        similar_effects.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        return similar_effects[:5]
    
    def _save_generated_effect(self, description: str, code: str):
        """Save generated effect to database for future reference"""
        
        try:
            generated_effects_dir = Path("data/generated_effects")
            generated_effects_dir.mkdir(parents=True, exist_ok=True)
            
            effect_data = {
                "description": description,
                "code": code,
                "generated_timestamp": __import__('time').time(),
                "source": "self_coding"
            }
            
            # Save individual effect
            effect_file = generated_effects_dir / f"effect_{len(list(generated_effects_dir.glob('*.json')))}.json"
            with open(effect_file, 'w') as f:
                json.dump(effect_data, f, indent=2)
            
            # Add to database
            self.effect_database.append(effect_data)
            
            logger.info(f"Saved generated effect: {effect_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save generated effect: {e}")
    
    def apply_generated_effect(self, frames: List[np.ndarray], 
                             effect_description: str, **effect_params) -> List[np.ndarray]:
        """Apply a generated effect to a list of frames"""
        
        # Create the custom effect
        effect_function = self.create_custom_effect(effect_description, frames[0] if frames else None)
        
        if effect_function is None:
            logger.error(f"Could not create effect: {effect_description}")
            return frames
        
        # Apply effect to all frames
        processed_frames = []
        for i, frame in enumerate(frames):
            try:
                processed_frame = effect_function(frame, **effect_params)
                processed_frames.append(processed_frame)
                
                if i % 10 == 0:  # Log progress
                    logger.info(f"Processed {i+1}/{len(frames)} frames")
                    
            except Exception as e:
                logger.warning(f"Effect failed on frame {i}: {e}")
                processed_frames.append(frame)  # Use original frame
        
        logger.info(f"Applied generated effect '{effect_description}' to {len(frames)} frames")
        return processed_frames
    
    def list_available_effects(self) -> List[Dict]:
        """List all available effects (database + generated)"""
        
        effects = []
        
        # Add database effects
        for effect in self.effect_database:
            effects.append({
                "name": effect.get("name", "Unknown"),
                "description": effect.get("description", ""),
                "category": effect.get("category", "unknown"),
                "source": effect.get("source", "database")
            })
        
        # Add cached generated effects
        for description in self.generated_effects_cache:
            effects.append({
                "name": f"Generated: {description}",
                "description": description,
                "category": "generated",
                "source": "self_coding"
            })
        
        return effects


def create_self_coding_editor(config: DictConfig) -> SelfCodingVideoEditor:
    """Factory function to create self-coding video editor"""
    return SelfCodingVideoEditor(config)