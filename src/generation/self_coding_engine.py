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
        """
        Production-ready video effect code generation using fine-tuned CodeLLaMA with advanced strategies.
        Implements multiple generation approaches, sophisticated validation, and robust fallback systems.
        """
        
        # Initialize model if needed
        if self.model is None:
            logger.info("🤖 CodeLLaMA not loaded, attempting initialization...")
            self._initialize_model()
            
            if self.model is None:
                logger.warning("CodeLLaMA initialization failed, using sophisticated template system")
                return self._generate_production_ready_template_code(effect_description, reference_effects)
        
        try:
            # Multi-stage generation process for optimal results
            logger.info(f"🎬 Generating video effect code for: '{effect_description}'")
            
            # Stage 1: Comprehensive prompt engineering
            generation_context = self._analyze_effect_requirements(effect_description)
            sophisticated_prompt = self._create_production_grade_prompt(
                effect_description, reference_effects, generation_context
            )
            
            # Stage 2: Advanced tokenization with context preservation
            inputs = self.tokenizer(
                sophisticated_prompt,
                return_tensors="pt",
                max_length=4096,  # Large context for complex effects
                truncation=True,
                padding=True,
                add_special_tokens=True
            )
            
            # Move to appropriate device
            device = next(self.model.parameters()).device if self.model else 'cpu'
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Stage 3: Multi-strategy code generation
            generated_codes = []
            
            # Strategy A: High-precision generation for technical effects
            if generation_context['technical_complexity'] == 'high':
                precision_codes = self._generate_precision_code(inputs, effect_description)
                generated_codes.extend(precision_codes)
            
            # Strategy B: Creative generation for artistic effects  
            if generation_context['creative_complexity'] == 'high':
                creative_codes = self._generate_creative_code(inputs, effect_description)
                generated_codes.extend(creative_codes)
            
            # Strategy C: Balanced generation for general effects
            if not generated_codes or generation_context['complexity_level'] == 'moderate':
                balanced_codes = self._generate_balanced_code(inputs, effect_description)
                generated_codes.extend(balanced_codes)
            
            # Stage 4: Advanced code validation and selection
            if generated_codes:
                validated_codes = []
                for code in generated_codes:
                    if self._comprehensive_code_validation(code, effect_description):
                        validated_codes.append(code)
                
                if validated_codes:
                    best_code = self._intelligent_code_selection(
                        validated_codes, effect_description, generation_context
                    )
                    
                    # Final optimization and cleanup
                    optimized_code = self._optimize_and_finalize_code(best_code, effect_description)
                    
                    logger.info(f"✅ Successfully generated production-ready effect code")
                    logger.info(f"   Code length: {len(optimized_code)} characters")
                    logger.info(f"   Complexity: {generation_context['complexity_level']}")
                    
                    return optimized_code
            
            # Stage 5: Fallback to advanced template system
            logger.warning("Code generation validation failed, using advanced template system")
            return self._generate_production_ready_template_code(effect_description, reference_effects)
                
        except Exception as e:
            logger.error(f"Code generation pipeline failed: {e}")
            return self._generate_production_ready_template_code(effect_description, reference_effects)
    
    def _analyze_effect_requirements(self, effect_description: str) -> Dict[str, Any]:
        """Analyze effect requirements for optimal generation strategy"""
        
        context = {
            'complexity_level': 'moderate',
            'technical_complexity': 'moderate', 
            'creative_complexity': 'moderate',
            'performance_requirements': 'standard',
            'library_preferences': [],
            'effect_category': 'general'
        }
        
        desc_lower = effect_description.lower()
        
        # Analyze technical complexity
        high_tech_indicators = ['algorithm', 'mathematical', 'transformation', 'matrix', 'computation']
        tech_score = sum(1 for indicator in high_tech_indicators if indicator in desc_lower)
        
        if tech_score >= 2:
            context['technical_complexity'] = 'high'
        elif tech_score == 1:
            context['technical_complexity'] = 'moderate'
        else:
            context['technical_complexity'] = 'low'
        
        # Analyze creative complexity
        creative_indicators = ['artistic', 'creative', 'unique', 'innovative', 'stylized', 'abstract']
        creative_score = sum(1 for indicator in creative_indicators if indicator in desc_lower)
        
        if creative_score >= 2:
            context['creative_complexity'] = 'high'
        elif creative_score == 1:
            context['creative_complexity'] = 'moderate'
        else:
            context['creative_complexity'] = 'low'
        
        # Overall complexity
        total_complexity = tech_score + creative_score
        if total_complexity >= 3:
            context['complexity_level'] = 'high'
        elif total_complexity >= 1:
            context['complexity_level'] = 'moderate'
        else:
            context['complexity_level'] = 'low'
        
        # Performance analysis
        realtime_indicators = ['real-time', 'fast', 'quick', 'instant', 'live']
        if any(indicator in desc_lower for indicator in realtime_indicators):
            context['performance_requirements'] = 'high_speed'
        elif any(indicator in ['complex', 'detailed', 'high-quality'] for indicator in desc_lower.split()):
            context['performance_requirements'] = 'high_quality'
        
        # Library preferences
        if any(term in desc_lower for term in ['opencv', 'cv2', 'computer vision']):
            context['library_preferences'].append('opencv')
        if any(term in desc_lower for term in ['numpy', 'numerical', 'mathematical']):
            context['library_preferences'].append('numpy')
        if any(term in desc_lower for term in ['pillow', 'pil', 'image']):
            context['library_preferences'].append('pillow')
        
        # Effect categorization
        categories = {
            'glitch': ['glitch', 'corruption', 'digital', 'noise'],
            'color': ['color', 'hue', 'saturation', 'brightness', 'contrast'],
            'motion': ['motion', 'movement', 'tracking', 'stabilization'],
            'particle': ['particle', 'spark', 'dust', 'fire', 'smoke'],
            'distortion': ['distortion', 'warp', 'ripple', 'wave'],
            'composite': ['composite', 'blend', 'mask', 'overlay']
        }
        
        for category, keywords in categories.items():
            if any(keyword in desc_lower for keyword in keywords):
                context['effect_category'] = category
                break
        
        return context
    
    def _generate_precision_code(self, inputs: Dict, description: str) -> List[str]:
        """Generate high-precision code for technical effects"""
        codes = []
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.1,  # Very low temperature for precision
                    top_p=0.8,
                    do_sample=True,
                    num_return_sequences=2,
                    pad_token_id=self.tokenizer.eos_token_id or self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.05,
                    no_repeat_ngram_size=3
                )
            
            for output in outputs:
                generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
                code = self._extract_and_validate_code(generated_text, inputs)
                if code:
                    codes.append(code)
                    
        except Exception as e:
            logger.warning(f"Precision code generation failed: {e}")
        
        return codes
    
    def _generate_creative_code(self, inputs: Dict, description: str) -> List[str]:
        """Generate creative code for artistic effects"""
        codes = []
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=896,
                    temperature=0.8,  # Higher temperature for creativity
                    top_p=0.95,
                    top_k=50,
                    do_sample=True,
                    num_return_sequences=2,
                    pad_token_id=self.tokenizer.eos_token_id or self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.15
                )
            
            for output in outputs:
                generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
                code = self._extract_and_validate_code(generated_text, inputs)
                if code:
                    codes.append(code)
                    
        except Exception as e:
            logger.warning(f"Creative code generation failed: {e}")
        
        return codes
    
    def _generate_balanced_code(self, inputs: Dict, description: str) -> List[str]:
        """Generate balanced code for general effects"""
        codes = []
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=768,
                    temperature=0.4,  # Balanced temperature
                    top_p=0.9,
                    do_sample=True,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id or self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            code = self._extract_and_validate_code(generated_text, inputs)
            if code:
                codes.append(code)
                    
        except Exception as e:
            logger.warning(f"Balanced code generation failed: {e}")
        
        return codes
    
    def _comprehensive_code_validation(self, code: str, description: str) -> bool:
        """Comprehensive validation of generated code"""
        
        try:
            # Basic syntax validation
            ast.parse(code)
            
            # Function definition check
            if not any(line.strip().startswith('def ') for line in code.split('\n')):
                return False
            
            # Security validation
            forbidden_patterns = [
                'exec', 'eval', '__import__', 'open(', 'file(',
                'subprocess', 'os.system', 'os.popen'
            ]
            
            if any(pattern in code for pattern in forbidden_patterns):
                return False
            
            # Video processing validation
            required_elements = ['frame', 'return']
            if not all(element in code for element in required_elements):
                return False
            
            # Library usage validation
            expected_libraries = ['cv2', 'np', 'numpy']
            if not any(lib in code for lib in expected_libraries):
                return False
            
            # Relevance check
            desc_keywords = description.lower().split()
            code_lower = code.lower()
            relevance_score = sum(1 for keyword in desc_keywords if keyword in code_lower)
            
            if relevance_score < len(desc_keywords) * 0.3:  # At least 30% relevance
                return False
            
            return True
            
        except (SyntaxError, ValueError):
            return False
        except Exception:
            return False
    
    def _intelligent_code_selection(self, codes: List[str], description: str, 
                                  context: Dict[str, Any]) -> str:
        """Intelligently select the best code based on multiple criteria"""
        
        if len(codes) == 1:
            return codes[0]
        
        scored_codes = []
        
        for code in codes:
            score = 0
            
            # Complexity appropriateness (30 points max)
            lines = [l for l in code.split('\n') if l.strip() and not l.strip().startswith('#')]
            code_complexity = len(lines)
            
            if context['complexity_level'] == 'high':
                score += min(30, code_complexity * 1.5)
            elif context['complexity_level'] == 'moderate':
                score += max(0, 30 - abs(code_complexity - 15) * 2)
            else:
                score += max(0, 30 - code_complexity)
            
            # Relevance score (25 points max)
            desc_words = set(description.lower().split())
            code_words = set(code.lower().split())
            relevance = len(desc_words.intersection(code_words)) / len(desc_words) if desc_words else 0
            score += relevance * 25
            
            # Library usage score (20 points max)
            preferred_libs = context.get('library_preferences', [])
            for lib in preferred_libs:
                if lib == 'opencv' and 'cv2.' in code:
                    score += 8
                elif lib == 'numpy' and ('np.' in code or 'numpy.' in code):
                    score += 6
                elif lib == 'pillow' and 'PIL' in code:
                    score += 6
            
            # Documentation quality (15 points max)
            if '"""' in code or "'''" in code:
                score += 10
            if '#' in code:
                comment_count = code.count('#')
                score += min(5, comment_count)
            
            # Performance considerations (10 points max)
            if context['performance_requirements'] == 'high_speed':
                if any(term in code for term in ['copy()', '.copy()', 'dtype=']):
                    score += 5
                if 'for' in code and 'range' in code:
                    score -= 3  # Penalize explicit loops for performance
            
            scored_codes.append((score, code))
        
        # Return highest scoring code
        scored_codes.sort(key=lambda x: x[0], reverse=True)
        best_score, best_code = scored_codes[0]
        
        logger.debug(f"Selected code with score: {best_score:.1f}")
        return best_code
    
    def _optimize_and_finalize_code(self, code: str, description: str) -> str:
        """Optimize and finalize the generated code"""
        
        # Add comprehensive documentation if missing
        if '"""' not in code and "'''" not in code:
            # Extract function name
            import re
            func_match = re.search(r'def\s+(\w+)\s*\(', code)
            func_name = func_match.group(1) if func_match else 'generated_effect'
            
            doc_string = f'    """\n    {description}\n    \n    Args:\n        frame: Input video frame (numpy array)\n        \n    Returns:\n        numpy array: Processed video frame\n    """'
            
            # Insert documentation after function definition
            lines = code.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('def '):
                    lines.insert(i + 1, doc_string)
                    break
            
            code = '\n'.join(lines)
        
        # Ensure proper imports at the top
        required_imports = []
        if 'cv2.' in code and 'import cv2' not in code:
            required_imports.append('import cv2')
        if ('np.' in code or 'numpy.' in code) and 'import numpy' not in code:
            required_imports.append('import numpy as np')
        if 'PIL' in code and 'from PIL' not in code:
            required_imports.append('from PIL import Image')
        
        if required_imports:
            import_block = '\n'.join(required_imports) + '\n\n'
            code = import_block + code
        
        # Add type hints if missing
        if 'def ' in code and '->' not in code:
            code = code.replace('def ', 'def ', 1).replace('(frame', '(frame: np.ndarray').replace('):', ') -> np.ndarray:', 1)
        
        return code
    
    def _generate_production_ready_template_code(self, description: str, 
                                               reference_effects: List[Dict] = None) -> str:
        """Generate production-ready code using sophisticated template system"""
        
        # Analyze description for template selection
        context = self._analyze_effect_requirements(description)
        template_type = self._select_optimal_template(description, context)
        
        # Generate code using selected template
        if template_type == 'glitch':
            return self._generate_glitch_template(description, context)
        elif template_type == 'color':
            return self._generate_color_template(description, context)
        elif template_type == 'motion':
            return self._generate_motion_template(description, context)
        elif template_type == 'particle':
            return self._generate_particle_template(description, context)
        elif template_type == 'distortion':
            return self._generate_distortion_template(description, context)
        elif template_type == 'composite':
            return self._generate_composite_template(description, context)
        else:
            return self._generate_universal_template(description, context)
    
    def _select_optimal_template(self, description: str, context: Dict[str, Any]) -> str:
        """Select the most appropriate template based on description analysis"""
        
        # Use effect category from context analysis
        category = context.get('effect_category', 'general')
        
        # Template mapping with confidence scores
        template_scores = {
            'glitch': 0,
            'color': 0,
            'motion': 0,
            'particle': 0,
            'distortion': 0,
            'composite': 0,
            'universal': 0.1  # Always a viable option
        }
        
        desc_lower = description.lower()
        
        # Score each template type
        glitch_keywords = ['glitch', 'digital', 'corruption', 'noise', 'static', 'rgb shift']
        template_scores['glitch'] = sum(0.2 for keyword in glitch_keywords if keyword in desc_lower)
        
        color_keywords = ['color', 'hue', 'saturation', 'brightness', 'contrast', 'tint', 'grade']
        template_scores['color'] = sum(0.15 for keyword in color_keywords if keyword in desc_lower)
        
        motion_keywords = ['motion', 'movement', 'tracking', 'stabilize', 'pan', 'zoom']
        template_scores['motion'] = sum(0.2 for keyword in motion_keywords if keyword in desc_lower)
        
        particle_keywords = ['particle', 'spark', 'dust', 'fire', 'smoke', 'stars', 'snow']
        template_scores['particle'] = sum(0.25 for keyword in particle_keywords if keyword in desc_lower)
        
        distortion_keywords = ['distortion', 'warp', 'ripple', 'wave', 'bend', 'twist']
        template_scores['distortion'] = sum(0.2 for keyword in distortion_keywords if keyword in desc_lower)
        
        composite_keywords = ['composite', 'blend', 'mask', 'overlay', 'green screen', 'chroma']
        template_scores['composite'] = sum(0.2 for keyword in composite_keywords if keyword in desc_lower)
        
        # Select highest scoring template
        best_template = max(template_scores.items(), key=lambda x: x[1])
        return best_template[0] if best_template[1] > 0.3 else 'universal'
    
    def _generate_glitch_template(self, description: str, context: Dict[str, Any]) -> str:
        """Generate advanced glitch effect template"""
        
        intensity = 0.5  # Default intensity
        if 'subtle' in description.lower():
            intensity = 0.2
        elif 'intense' in description.lower() or 'heavy' in description.lower():
            intensity = 0.8
        
        return f'''import cv2
import numpy as np
from typing import Tuple, Optional

def glitch_effect(frame: np.ndarray, intensity: float = {intensity}, 
                 scan_lines: bool = True, rgb_shift: bool = True,
                 digital_noise: bool = True) -> np.ndarray:
    """
    Advanced digital glitch effect with multiple distortion types.
    
    {description}
    
    Args:
        frame: Input video frame (numpy array)
        intensity: Effect intensity (0.0 to 1.0)
        scan_lines: Enable horizontal scan line glitches
        rgb_shift: Enable RGB channel shifting
        digital_noise: Enable digital noise overlay
        
    Returns:
        numpy array: Processed frame with glitch effects
    """
    height, width = frame.shape[:2]
    result = frame.copy().astype(np.float32)
    
    if rgb_shift:
        # RGB channel shifting for chromatic aberration
        shift_amount = int(intensity * 15)
        if shift_amount > 0:
            # Red channel shift
            result[:-shift_amount, :, 0] = frame[shift_amount:, :, 0]
            # Blue channel shift  
            result[shift_amount:, :, 2] = frame[:-shift_amount, :, 2]
    
    if scan_lines:
        # Horizontal scan line corruption
        num_lines = int(height * intensity * 0.1)
        for _ in range(num_lines):
            y = np.random.randint(0, height - 5)
            line_height = np.random.randint(1, 4)
            
            # Create noise for scan line
            noise = np.random.randint(0, 255, (line_height, width, 3))
            blend_factor = np.random.uniform(0.3, 0.7)
            
            result[y:y+line_height] = (
                result[y:y+line_height] * (1 - blend_factor) + 
                noise * blend_factor
            )
    
    if digital_noise:
        # Digital pixel corruption
        noise_pixels = int(width * height * intensity * 0.02)
        for _ in range(noise_pixels):
            x, y = np.random.randint(0, width), np.random.randint(0, height)
            result[y, x] = np.random.randint(0, 255, 3)
    
    # Data moshing effect for high intensity
    if intensity > 0.6:
        block_size = 8
        num_blocks = int(intensity * 20)
        
        for _ in range(num_blocks):
            x = np.random.randint(0, width - block_size)
            y = np.random.randint(0, height - block_size)
            
            # Duplicate or corrupt block
            if np.random.random() < 0.5:
                # Duplicate from random location
                src_x = np.random.randint(0, width - block_size)
                src_y = np.random.randint(0, height - block_size)
                result[y:y+block_size, x:x+block_size] = result[src_y:src_y+block_size, src_x:src_x+block_size]
            else:
                # Add noise block
                result[y:y+block_size, x:x+block_size] = np.random.randint(0, 255, (block_size, block_size, 3))
    
    return np.clip(result, 0, 255).astype(np.uint8)
'''
    
    def _generate_color_template(self, description: str, context: Dict[str, Any]) -> str:
        """Generate advanced color manipulation template"""
        
        return f'''import cv2
import numpy as np
from typing import Tuple, Optional

def color_effect(frame: np.ndarray, hue_shift: float = 0.1, 
                saturation_factor: float = 1.2, brightness_offset: int = 10,
                contrast_factor: float = 1.1, color_temperature: float = 0.0) -> np.ndarray:
    """
    Advanced color manipulation and grading effect.
    
    {description}
    
    Args:
        frame: Input video frame (numpy array)
        hue_shift: Hue adjustment (-0.5 to 0.5)
        saturation_factor: Saturation multiplier (0.0 to 2.0)
        brightness_offset: Brightness adjustment (-100 to 100)
        contrast_factor: Contrast multiplier (0.5 to 2.0)
        color_temperature: Color temperature shift (-1.0 to 1.0)
        
    Returns:
        numpy array: Color-processed frame
    """
    # Convert to HSV for hue/saturation adjustments
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Hue shift
    if abs(hue_shift) > 0.001:
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift * 180) % 180
    
    # Saturation adjustment
    if abs(saturation_factor - 1.0) > 0.001:
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
    
    # Convert back to BGR
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    
    # Brightness and contrast adjustments
    if abs(brightness_offset) > 0 or abs(contrast_factor - 1.0) > 0.001:
        result = result * contrast_factor + brightness_offset
    
    # Color temperature adjustment
    if abs(color_temperature) > 0.001:
        if color_temperature > 0:  # Warmer (more red/yellow)
            result[:, :, 0] *= (1 - color_temperature * 0.2)  # Reduce blue
            result[:, :, 2] *= (1 + color_temperature * 0.1)  # Increase red
        else:  # Cooler (more blue)
            result[:, :, 0] *= (1 + abs(color_temperature) * 0.2)  # Increase blue
            result[:, :, 2] *= (1 - abs(color_temperature) * 0.1)  # Reduce red
    
    # Professional color curve adjustment (S-curve for cinematic look)
    normalized = result / 255.0
    # Apply subtle S-curve
    s_curved = 3 * normalized**2 - 2 * normalized**3
    result = s_curved * 255
    
    return np.clip(result, 0, 255).astype(np.uint8)
'''

    def _generate_universal_template(self, description: str, context: Dict[str, Any]) -> str:
        """Generate universal effect template for general use"""
        
        return f'''import cv2
import numpy as np
from typing import Tuple, Optional, Any

def custom_effect(frame: np.ndarray, **kwargs) -> np.ndarray:
    """
    Universal video effect processor.
    
    {description}
    
    Args:
        frame: Input video frame (numpy array)
        **kwargs: Effect parameters
        
    Returns:
        numpy array: Processed video frame
    """
    height, width = frame.shape[:2]
    result = frame.copy()
    
    # Extract common parameters
    intensity = kwargs.get('intensity', 0.5)
    blend_mode = kwargs.get('blend_mode', 'normal')
    
    # Basic processing pipeline
    if intensity > 0.1:
        # Apply primary effect based on description analysis
        processed = frame.astype(np.float32)
        
        # Example processing - adapt based on specific needs
        if 'blur' in "{description.lower()}":
            kernel_size = int(intensity * 20) + 1
            if kernel_size % 2 == 0:
                kernel_size += 1
            processed = cv2.GaussianBlur(processed, (kernel_size, kernel_size), 0)
        
        elif 'sharpen' in "{description.lower()}":
            # Sharpening kernel
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(processed, -1, kernel)
            processed = cv2.addWeighted(processed, 1 - intensity, sharpened, intensity, 0)
        
        elif 'vintage' in "{description.lower()}":
            # Vintage film effect
            # Reduce saturation
            hsv = cv2.cvtColor(processed.astype(np.uint8), cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = hsv[:, :, 1] * 0.7  # Reduce saturation
            processed = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)
            
            # Add warm tint
            processed[:, :, 0] *= 0.9  # Reduce blue
            processed[:, :, 2] *= 1.1  # Increase red
        
        else:
            # Generic enhancement
            # Slight contrast and brightness adjustment
            processed = processed * (1 + intensity * 0.3) + (intensity * 20)
        
        # Blend with original based on intensity
        result = cv2.addWeighted(
            frame.astype(np.float32), 1 - intensity,
            processed, intensity, 0
        )
    
    return np.clip(result, 0, 255).astype(np.uint8)
'''

    def _create_production_grade_prompt(self, description: str, reference_effects: List[Dict] = None,
                                      context: Dict[str, Any] = None) -> str:
        """Create production-grade prompt for CodeLLaMA"""
        
        # Build comprehensive context
        context_str = f"""
# Professional Video Effects Code Generator
# Expertise: Computer Vision, Image Processing, Video Effects Programming
# Libraries: OpenCV, NumPy, Advanced Mathematics
# Target: Production-ready, optimized video processing code

## Effect Request Analysis:
- Description: {description}
- Complexity Level: {context.get('complexity_level', 'moderate') if context else 'moderate'}
- Performance Requirements: {context.get('performance_requirements', 'standard') if context else 'standard'}
- Technical Focus: {context.get('technical_complexity', 'moderate') if context else 'moderate'}
- Creative Focus: {context.get('creative_complexity', 'moderate') if context else 'moderate'}

## Code Generation Guidelines:
1. Write professional, production-ready Python code
2. Use OpenCV (cv2) and NumPy for video processing
3. Include comprehensive docstrings and type hints
4. Implement proper error handling and bounds checking
5. Optimize for performance and memory efficiency
6. Follow PEP 8 coding standards

## Reference Examples:
"""
        
        # Add reference effects if provided
        if reference_effects:
            context_str += "### Similar Effects:\n"
            for i, ref in enumerate(reference_effects[:2], 1):
                context_str += f"{i}. {ref.get('description', 'Unknown effect')}\n"
        
        # Build the specific generation request
        generation_request = f"""
## Generate Effect Code:
Create a complete Python function that implements: "{description}"

Requirements:
- Function name should be descriptive (e.g., 'glitch_effect', 'color_grade_effect')
- Accept frame (numpy array) as primary parameter
- Include adjustable parameters for effect intensity and variations
- Return processed frame as numpy array
- Include comprehensive docstring with parameter descriptions
- Use type hints for all parameters and return values
- Implement proper input validation and error handling
- Optimize for real-time video processing where possible

def """
        
        return context_str + generation_request

    def _initialize_model(self):
        """Try to initialize CodeLLaMA model on-demand"""
        try:
            model_name = self.config.get('model_name', 'codellama/CodeLlama-7b-Python-hf')
            
            logger.info(f"Loading CodeLLaMA model: {model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=self.config.get('cache_dir', 'models/cache')
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimization
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto" if torch.cuda.is_available() else None,
                load_in_8bit=True,  # Enable quantization for memory efficiency
                trust_remote_code=True,
                cache_dir=self.config.get('cache_dir', 'models/cache')
            )
            
            self.model.eval()
            logger.info("✅ CodeLLaMA model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize CodeLLaMA: {e}")
            self.model = None
            self.tokenizer = None
    
    def _extract_and_validate_code(self, generated_text: str, prompt: str) -> Optional[str]:
        """Extract and validate generated code with enhanced parsing"""
        
        # Remove the original prompt
        if prompt in generated_text:
            code_part = generated_text.replace(prompt, "").strip()
        else:
            code_part = generated_text.strip()
        
        # Enhanced code extraction
        import re
        
        # Look for function definitions
        function_patterns = [
            r'def\s+\w+\(.*?\):.*?(?=\n\ndef|\n\n#|\Z)',
            r'def\s+generated_effect\(.*?\):.*?(?=\n\ndef|\n\n#|\Z)',
            r'def\s+\w+_effect\(.*?\):.*?(?=\n\ndef|\n\n#|\Z)'
        ]
        
        extracted_code = None
        
        for pattern in function_patterns:
            matches = re.findall(pattern, code_part, re.DOTALL)
            if matches:
                extracted_code = matches[0].strip()
                break
        
        if not extracted_code:
            # Fallback: extract everything that looks like code
            lines = code_part.split('\n')
            code_lines = []
            in_function = False
            
            for line in lines:
                if line.strip().startswith('def '):
                    in_function = True
                    code_lines.append(line)
                elif in_function:
                    if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                        break  # End of function
                    code_lines.append(line)
            
            if code_lines:
                extracted_code = '\n'.join(code_lines)
        
        # Validate the extracted code
        if extracted_code and self._validate_generated_code(extracted_code):
            return extracted_code
        
        return None
    
    def _validate_generated_code(self, code: str) -> bool:
        """Enhanced validation for generated code"""
        
        try:
            # Parse AST to check syntax
            ast.parse(code)
            
            # Check that it has a function definition
            if 'def ' not in code:
                return False
            
            # Check that it uses allowed modules
            forbidden = ['exec', 'eval', '__import__', 'open', 'file']
            if any(f in code for f in forbidden):
                return False
            
            # Check for basic video processing logic
            video_keywords = ['frame', 'cv2', 'np.', 'numpy', 'image']
            if not any(keyword in code for keyword in video_keywords):
                return False
            
            return True
            
        except SyntaxError:
            return False
        except Exception:
            return False
    
    def _select_best_code(self, codes: List[str], description: str) -> str:
        """Select the best generated code based on multiple criteria"""
        
        if len(codes) == 1:
            return codes[0]
        
        scored_codes = []
        
        for code in codes:
            score = 0
            
            # Complexity score (moderate complexity is better)
            lines = code.split('\n')
            complexity = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
            if 5 <= complexity <= 20:
                score += 2
            elif complexity > 20:
                score += 1
            
            # Keyword relevance score
            desc_words = description.lower().split()
            code_lower = code.lower()
            keyword_matches = sum(1 for word in desc_words if word in code_lower)
            score += keyword_matches
            
            # Library usage score
            if 'cv2.' in code:
                score += 3
            if 'np.' in code or 'numpy.' in code:
                score += 2
            if 'PIL' in code:
                score += 1
            
            # Documentation score
            if '"""' in code or "'''" in code:
                score += 1
            
            scored_codes.append((score, code))
        
        # Return highest scoring code
        scored_codes.sort(key=lambda x: x[0], reverse=True)
        return scored_codes[0][1]
    
    def _create_enhanced_code_generation_prompt(self, description: str, 
                                              reference_effects: List[Dict] = None) -> str:
        """Create enhanced prompt for sophisticated code generation"""
        
        prompt = """# Advanced Video Effect Code Generator
# Expert-level Python code generation for video effects using OpenCV, NumPy, and advanced techniques

# Example 1: Advanced Glitch Effect
def glitch_effect(frame, intensity=0.3, scan_lines=True, rgb_shift=True):
    '''Create digital glitch effect with multiple distortion types'''
    import cv2
    import numpy as np
    
    height, width = frame.shape[:2]
    result = frame.copy()
    
    if rgb_shift:
        # RGB channel shifting for glitch effect
        shift = int(intensity * 10)
        result[:, shift:, 0] = frame[:, :-shift, 0]  # Red shift
        result[:, :-shift, 2] = frame[:, shift:, 2]  # Blue shift
    
    if scan_lines:
        # Add horizontal scan line glitches
        for i in range(0, height, int(20 / intensity)):
            if np.random.random() < intensity:
                line_height = np.random.randint(1, 5)
                noise = np.random.randint(0, 255, (line_height, width, 3), dtype=np.uint8)
                result[i:i+line_height] = cv2.addWeighted(result[i:i+line_height], 0.7, noise, 0.3, 0)
    
    return result

# Example 2: Particle System Effect
def particle_effect(frame, particle_count=100, trail_length=10):
    '''Create dynamic particle overlay effect'''
    import cv2
    import numpy as np
    
    height, width = frame.shape[:2]
    result = frame.copy()
    
    # Generate random particle positions and velocities
    particles = np.random.randint(0, min(width, height), (particle_count, 2))
    velocities = np.random.randn(particle_count, 2) * 2
    
    # Draw particles with trails
    overlay = np.zeros_like(frame)
    for i, (pos, vel) in enumerate(zip(particles, velocities)):
        # Draw particle trail
        for t in range(trail_length):
            alpha = (trail_length - t) / trail_length
            trail_pos = pos - vel * t
            if 0 <= trail_pos[0] < width and 0 <= trail_pos[1] < height:
                cv2.circle(overlay, tuple(trail_pos.astype(int)), 2, (255, 255, 255), -1)
    
    # Blend with original frame
    result = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
    return result

# Example 3: Advanced Color Grading
def cinematic_grade(frame, mood='warm', intensity=0.5):
    '''Professional cinematic color grading'''
    import cv2
    import numpy as np
    
    result = frame.astype(np.float32) / 255.0
    
    if mood == 'warm':
        # Warm color grade - enhance oranges and reduce blues
        result[:,:,0] *= (1 - intensity * 0.2)  # Reduce blue
        result[:,:,2] *= (1 + intensity * 0.3)  # Enhance red
    elif mood == 'cool':
        # Cool color grade - enhance blues and reduce reds
        result[:,:,0] *= (1 + intensity * 0.3)  # Enhance blue
        result[:,:,2] *= (1 - intensity * 0.2)  # Reduce red
    elif mood == 'cyberpunk':
        # Cyberpunk grade - high contrast with cyan/magenta
        result = cv2.convertScaleAbs(result, alpha=1.2, beta=-20)
        result[:,:,1] *= (1 + intensity * 0.4)  # Enhance green for cyan
    
    # Apply S-curve for contrast
    result = np.power(result, 1.0 / (1 + intensity))
    
    return (np.clip(result, 0, 1) * 255).astype(np.uint8)

"""
        
        # Add sophisticated reference effects if provided
        if reference_effects:
            prompt += "# Advanced Reference Effects:\n"
            for i, effect in enumerate(reference_effects[:3]):
                if 'code' in effect:
                    prompt += f"# Reference {i+1}: {effect.get('name', 'Advanced Effect')}\n"
                    prompt += f"# Description: {effect.get('description', 'Professional video effect')}\n"
                    prompt += effect['code'] + "\n\n"
        
        # Enhanced task specification
        prompt += f"""
# TASK: Generate sophisticated, production-quality code for the following effect:
# DESCRIPTION: {description}

# REQUIREMENTS:
# - Use advanced OpenCV and NumPy techniques
# - Implement professional-grade video processing
# - Handle edge cases and parameter validation
# - Include comprehensive docstrings
# - Use efficient algorithms suitable for real-time processing
# - Support customizable parameters via **kwargs
# - Return high-quality processed frame

# ADVANCED GUIDELINES:
# - For glitch effects: Use channel shifting, noise injection, scan line distortion
# - For particle effects: Implement physics-based movement and trails
# - For color effects: Use proper color space conversions and LUTs
# - For geometric effects: Apply proper transformations and interpolation
# - For temporal effects: Consider frame sequence and motion

def generated_effect(frame, **kwargs):
    '''
    {description}
    
    Advanced implementation with professional video processing techniques.
    
    Args:
        frame: Input video frame (numpy array)
        **kwargs: Effect parameters for customization
        
    Returns:
        numpy.ndarray: Processed frame with applied effect
    '''
    import cv2
    import numpy as np
    
    # Parameter extraction with defaults
    intensity = kwargs.get('intensity', 0.5)
    
    # Input validation
    if frame is None or frame.size == 0:
        return frame
    
    height, width = frame.shape[:2]
    result = frame.copy()
    
"""
        
        return prompt
    
    def _generate_advanced_template_code(self, description: str) -> str:
        """Generate sophisticated template-based code when CodeLLaMA is unavailable"""
        
        # Analyze description for effect type
        desc_lower = description.lower()
        
        # Advanced effect templates
        if any(word in desc_lower for word in ['glitch', 'digital', 'corruption', 'noise']):
            return self._generate_glitch_template(description)
        elif any(word in desc_lower for word in ['particle', 'spark', 'dust', 'fire', 'energy']):
            return self._generate_particle_template(description)
        elif any(word in desc_lower for word in ['color', 'grade', 'cinematic', 'mood', 'tint']):
            return self._generate_color_template(description)
        elif any(word in desc_lower for word in ['blur', 'focus', 'depth', 'bokeh']):
            return self._generate_blur_template(description)
        elif any(word in desc_lower for word in ['distort', 'warp', 'bend', 'ripple']):
            return self._generate_distortion_template(description)
        else:
            return self._generate_generic_template(description)
    
    def _generate_glitch_template(self, description: str) -> str:
        """Generate advanced glitch effect template"""
        return f'''
def generated_effect(frame, **kwargs):
    """
    {description}
    Advanced glitch effect with multiple distortion types.
    """
    import cv2
    import numpy as np
    
    intensity = kwargs.get('intensity', 0.3)
    rgb_shift = kwargs.get('rgb_shift', True)
    scan_lines = kwargs.get('scan_lines', True)
    
    height, width = frame.shape[:2]
    result = frame.copy()
    
    if rgb_shift:
        # RGB channel shifting
        shift = int(intensity * 15)
        if shift > 0:
            result[:, shift:, 0] = frame[:, :-shift, 0]  # Red shift
            result[:, :-shift, 2] = frame[:, shift:, 2]  # Blue shift
    
    if scan_lines:
        # Digital scan line artifacts
        for i in range(0, height, max(1, int(30 / (intensity + 0.1)))):
            if np.random.random() < intensity:
                line_height = np.random.randint(1, 6)
                end_row = min(i + line_height, height)
                # Add digital noise
                noise = np.random.randint(0, 255, (end_row - i, width, 3), dtype=np.uint8)
                alpha = 0.3 + intensity * 0.5
                result[i:end_row] = cv2.addWeighted(result[i:end_row], 1-alpha, noise, alpha, 0)
    
    # Add pixel corruption
    if intensity > 0.4:
        corruption_count = int(width * height * intensity * 0.001)
        for _ in range(corruption_count):
            x, y = np.random.randint(0, width), np.random.randint(0, height)
            result[y, x] = np.random.randint(0, 255, 3)
    
    return result
'''
    
    def _generate_particle_template(self, description: str) -> str:
        """Generate advanced particle system template"""
        return f'''
def generated_effect(frame, **kwargs):
    """
    {description}
    Advanced particle system with physics-based movement.
    """
    import cv2
    import numpy as np
    
    particle_count = kwargs.get('particle_count', 150)
    particle_size = kwargs.get('particle_size', 3)
    intensity = kwargs.get('intensity', 0.4)
    
    height, width = frame.shape[:2]
    result = frame.copy()
    
    # Create particle overlay
    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Generate particles with physics
    for _ in range(particle_count):
        # Random position and properties
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        
        # Particle color based on underlying pixel
        if 0 <= y < height and 0 <= x < width:
            base_color = frame[y, x]
            # Enhance brightness for particle effect
            particle_color = np.clip(base_color.astype(np.float32) * 1.5 + 50, 0, 255).astype(np.uint8)
        else:
            particle_color = (255, 255, 255)
        
        # Draw particle with glow effect
        cv2.circle(overlay, (x, y), particle_size + 2, particle_color // 3, -1)  # Outer glow
        cv2.circle(overlay, (x, y), particle_size, particle_color, -1)  # Core
        cv2.circle(overlay, (x, y), max(1, particle_size // 2), (255, 255, 255), -1)  # Highlight
    
    # Apply gaussian blur for glow effect
    overlay = cv2.GaussianBlur(overlay, (15, 15), 0)
    
    # Blend with original frame
    result = cv2.addWeighted(frame, 1 - intensity, overlay, intensity, 0)
    
    return result
'''
    
    def _generate_color_template(self, description: str) -> str:
        """Generate advanced color grading template"""
        return f'''
def generated_effect(frame, **kwargs):
    """
    {description}
    Professional color grading with multiple adjustment layers.
    """
    import cv2
    import numpy as np
    
    intensity = kwargs.get('intensity', 0.5)
    mood = kwargs.get('mood', 'cinematic')
    
    # Convert to float for precise calculations
    result = frame.astype(np.float32) / 255.0
    
    # Apply color grading based on mood
    if 'warm' in description.lower() or mood == 'warm':
        # Warm color grade
        result[:,:,2] *= (1 + intensity * 0.2)  # Enhance red/yellow
        result[:,:,0] *= (1 - intensity * 0.1)  # Reduce blue
    elif 'cool' in description.lower() or mood == 'cool':
        # Cool color grade
        result[:,:,0] *= (1 + intensity * 0.2)  # Enhance blue
        result[:,:,2] *= (1 - intensity * 0.1)  # Reduce red
    elif 'vintage' in description.lower():
        # Vintage look
        result[:,:,1] *= (1 + intensity * 0.1)  # Slight green tint
        result = np.power(result, 0.9)  # Reduce contrast slightly
    
    # Apply S-curve for professional contrast
    result = np.where(result < 0.5, 2 * result * result, 1 - 2 * (1 - result) * (1 - result))
    
    # Subtle vignette effect
    center_x, center_y = result.shape[1] // 2, result.shape[0] // 2
    Y, X = np.ogrid[:result.shape[0], :result.shape[1]]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    vignette = 1 - (dist_from_center / max_dist) * intensity * 0.3
    result *= vignette[:,:,np.newaxis]
    
    return (np.clip(result, 0, 1) * 255).astype(np.uint8)
'''
    
    def _generate_blur_template(self, description: str) -> str:
        """Generate advanced blur effect template"""
        return f'''
def generated_effect(frame, **kwargs):
    """
    {description}
    Advanced blur with depth and selective focus options.
    """
    import cv2
    import numpy as np
    
    blur_strength = kwargs.get('blur_strength', 15)
    focus_point = kwargs.get('focus_point', None)  # (x, y) tuple
    intensity = kwargs.get('intensity', 1.0)
    
    height, width = frame.shape[:2]
    
    if focus_point is None:
        # Uniform blur
        if 'motion' in description.lower():
            # Motion blur
            kernel = np.zeros((blur_strength, blur_strength))
            kernel[blur_strength//2, :] = np.ones(blur_strength)
            kernel = kernel / blur_strength
            result = cv2.filter2D(frame, -1, kernel)
        else:
            # Gaussian blur
            result = cv2.GaussianBlur(frame, (blur_strength*2+1, blur_strength*2+1), 0)
    else:
        # Radial blur with focus point
        center_x, center_y = focus_point
        Y, X = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(width**2 + height**2) / 4
        
        # Create distance-based blur map
        blur_map = np.clip(dist_from_center / max_dist, 0, 1) * intensity
        
        result = frame.copy()
        # Apply varying blur based on distance from focus point
        for i in range(1, blur_strength):
            mask = (blur_map >= i / blur_strength) & (blur_map < (i + 1) / blur_strength)
            if np.any(mask):
                blurred = cv2.GaussianBlur(frame, (i*2+1, i*2+1), 0)
                result[mask] = blurred[mask]
    
    return result
'''
    
    def _generate_distortion_template(self, description: str) -> str:
        """Generate advanced distortion effect template"""
        return f'''
def generated_effect(frame, **kwargs):
    """
    {description}
    Advanced geometric distortion with wave and ripple effects.
    """
    import cv2
    import numpy as np
    
    intensity = kwargs.get('intensity', 0.3)
    wave_length = kwargs.get('wave_length', 50)
    
    height, width = frame.shape[:2]
    
    # Create coordinate matrices
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    if 'ripple' in description.lower():
        # Ripple effect from center
        center_x, center_y = width // 2, height // 2
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Create ripple distortion
        displacement = np.sin(dist / wave_length) * intensity * 20
        new_x = np.clip(x + displacement, 0, width - 1).astype(np.float32)
        new_y = np.clip(y + displacement, 0, height - 1).astype(np.float32)
        
    elif 'wave' in description.lower():
        # Horizontal wave distortion
        wave_offset = np.sin(y / wave_length) * intensity * 30
        new_x = np.clip(x + wave_offset, 0, width - 1).astype(np.float32)
        new_y = y.astype(np.float32)
        
    else:
        # Generic distortion
        distort_x = np.sin(y / wave_length) * intensity * 10
        distort_y = np.cos(x / wave_length) * intensity * 10
        new_x = np.clip(x + distort_x, 0, width - 1).astype(np.float32)
        new_y = np.clip(y + distort_y, 0, height - 1).astype(np.float32)
    
    # Apply distortion using remap
    result = cv2.remap(frame, new_x, new_y, cv2.INTER_LINEAR)
    
    return result
'''
    
    def _generate_generic_template(self, description: str) -> str:
        """Generate generic advanced effect template"""
        return f'''
def generated_effect(frame, **kwargs):
    """
    {description}
    Advanced video effect with customizable parameters.
    """
    import cv2
    import numpy as np
    
    intensity = kwargs.get('intensity', 0.5)
    
    # Apply sophisticated processing based on description keywords
    result = frame.copy()
    
    # Enhance image processing
    if intensity > 0:
        # Apply unsharp masking for enhanced details
        gaussian_3 = cv2.GaussianBlur(result, (9, 9), 2.0)
        unsharp_image = cv2.addWeighted(result, 1.5 + intensity, gaussian_3, -0.5 - intensity, 0)
        result = unsharp_image
        
        # Apply subtle color enhancement
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Enhance luminance
        l = cv2.createCLAHE(clipLimit=2.0 + intensity, tileGridSize=(8,8)).apply(l)
        
        enhanced_lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return result
'''
    
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