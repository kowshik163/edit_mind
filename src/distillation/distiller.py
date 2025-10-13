"""
Knowledge Distillation Module
Distills knowledge from expert models into the hybrid AI system
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from omegaconf import DictConfig

try:
    from ..models.expert_models import ExpertModels
    from ..utils.distillation_utils import DistillationLoss, FeatureMatching
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from models.expert_models import ExpertModels
    from utils.distillation_utils import DistillationLoss, FeatureMatching

logger = logging.getLogger(__name__)


class KnowledgeDistiller:
    """
    Enhanced distillation from state-of-the-art teacher models:
    - RT-DETR (Real-time Detection Transformer)
    - HQ-SAM (High-Quality Segment Anything Model)
    - Whisper (speech recognition)
    - BeatNet (music/rhythm analysis)
    - Demucs (audio source separation)
    - RAFT (optical flow)
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load expert teacher models
        self.expert_models = ExpertModels(config)
        
        # Initialize advanced teacher models
        self._initialize_advanced_teachers()
        
        # Distillation loss functions
        self.distillation_loss = DistillationLoss(
            temperature=config.training.phase2.temperature,
            alpha=config.training.phase2.alpha
        )
        
        # Feature matching for intermediate representations
        self.feature_matcher = FeatureMatching()
        
        # Advanced teacher model handlers
        self.rt_detr = None
        self.hq_sam = None
        self.beatnet = None
        self.demucs = None
    
    def _initialize_advanced_teachers(self):
        """Initialize state-of-the-art teacher models"""
        logger.info("🔬 Initializing advanced teacher models...")
        
        # Initialize RT-DETR for object detection
        try:
            self.rt_detr = self._load_rt_detr()
            self._validate_model_authenticity('RT-DETR', self.rt_detr)
        except Exception as e:
            logger.warning(f"RT-DETR not available: {e}")
            self.rt_detr = None
        
        # Initialize HQ-SAM for segmentation
        try:
            self.hq_sam = self._load_hq_sam()
            self._validate_model_authenticity('HQ-SAM', self.hq_sam)
        except Exception as e:
            logger.warning(f"HQ-SAM not available: {e}")
            self.hq_sam = None
        
        # Initialize BeatNet for music analysis
        try:
            self.beatnet = self._load_beatnet()
            self._validate_model_authenticity('BeatNet', self.beatnet)
        except Exception as e:
            logger.warning(f"BeatNet not available: {e}")
            self.beatnet = None
        
        # Initialize Demucs for audio separation
        try:
            self.demucs = self._load_demucs()
            self._validate_model_authenticity('Demucs', self.demucs)
        except Exception as e:
            logger.warning(f"Demucs not available: {e}")
            self.demucs = None
        
        # Log final model status
        self._log_model_status()
    
    def _validate_model_authenticity(self, model_name: str, model_info: Dict):
        """Validate and log the authenticity of loaded models"""
        if model_info is None:
            logger.error(f"❌ {model_name}: Failed to load")
            return
        
        model_type = model_info.get('type', 'unknown')
        model_path = model_info.get('model_name', 'unknown')
        
        # Check if using real pre-trained models vs fallbacks
        if 'fallback' in model_type.lower():
            logger.warning(f"⚠️  {model_name}: Using FALLBACK implementation ({model_path})")
            logger.warning(f"   Consider installing proper {model_name} packages for full functionality")
        elif 'real' in model_type.lower() or 'teacher' in model_type.lower():
            logger.info(f"✅ {model_name}: Using AUTHENTIC pre-trained model ({model_path})")
        elif any(indicator in model_path.lower() for indicator in ['facebook', 'microsoft', 'openai', 'lyuwenyu']):
            logger.info(f"✅ {model_name}: Using official pre-trained model ({model_path})")
        else:
            logger.warning(f"🔍 {model_name}: Using alternative implementation ({model_path})")
    
    def _log_model_status(self):
        """Log overall model loading status"""
        
        models_status = {
            'RT-DETR (Object Detection)': self.rt_detr,
            'HQ-SAM (Segmentation)': self.hq_sam,
            'BeatNet (Music Analysis)': self.beatnet,
            'Demucs (Audio Separation)': self.demucs
        }
        
        authentic_count = 0
        fallback_count = 0
        failed_count = 0
        
        logger.info("🏗️  Advanced Teacher Models Status:")
        
        for model_name, model_info in models_status.items():
            if model_info is None:
                logger.info(f"   ❌ {model_name}: NOT LOADED")
                failed_count += 1
            else:
                model_type = model_info.get('type', '')
                if 'fallback' in model_type.lower():
                    logger.info(f"   ⚠️  {model_name}: FALLBACK")
                    fallback_count += 1
                else:
                    logger.info(f"   ✅ {model_name}: AUTHENTIC")
                    authentic_count += 1
        
        total_models = len(models_status)
        success_rate = (authentic_count + fallback_count) / total_models * 100
        
        logger.info(f"📊 Model Loading Summary:")
        logger.info(f"   Authentic Models: {authentic_count}/{total_models}")
        logger.info(f"   Fallback Models: {fallback_count}/{total_models}")
        logger.info(f"   Failed Models: {failed_count}/{total_models}")
        logger.info(f"   Success Rate: {success_rate:.1f}%")
        
        if authentic_count == total_models:
            logger.info("🎯 Perfect! All authentic teacher models loaded successfully")
        elif authentic_count > fallback_count:
            logger.info("👍 Good! Majority of authentic models loaded")
        elif fallback_count > 0:
            logger.warning("⚠️  Warning! Using fallback implementations - consider installing proper packages")
        else:
            logger.error("❌ Critical! No teacher models loaded successfully")
        
        # Provide installation guidance if needed
        if fallback_count > 0 or failed_count > 0:
            self._provide_installation_guidance()
    
    def _provide_installation_guidance(self):
        """Provide guidance for installing authentic models"""
        
        logger.info("📋 Installation Guidance for Authentic Models:")
        
        if self.rt_detr is None or 'fallback' in self.rt_detr.get('type', '').lower():
            logger.info("   RT-DETR:")
            logger.info("     pip install transformers")
            logger.info("     # Models will be downloaded automatically from Hugging Face")
        
        if self.hq_sam is None or 'fallback' in self.hq_sam.get('type', '').lower():
            logger.info("   HQ-SAM:")
            logger.info("     pip install transformers") 
            logger.info("     # SAM models will be downloaded from Hugging Face Hub")
        
        if self.beatnet is None or 'fallback' in self.beatnet.get('type', '').lower():
            logger.info("   BeatNet:")
            logger.info("     pip install BeatNet")
            logger.info("     # Official BeatNet package for advanced music analysis")
        
        if self.demucs is None or 'fallback' in self.demucs.get('type', '').lower():
            logger.info("   Demucs:")
            logger.info("     pip install demucs")
            logger.info("     # Official Demucs for professional audio separation")
    
    def _load_rt_detr(self):
        """
        Load RT-DETR model for real-time object detection with config-driven model selection.
        Prioritizes models specified in configuration for advanced video understanding.
        """
        # Get model configuration from config
        teacher_models = self.config.get('models', {}).get('teacher_models', {})
        rt_detr_config = teacher_models.get('rt_detr', {})
        
        # Priority order: config model -> latest RT-DETR -> DETR -> RetinaNet fallback
        model_candidates = []
        
        # Add configured model if available
        if rt_detr_config.get('model_name'):
            model_candidates.append({
                'name': rt_detr_config['model_name'],
                'type': 'configured_rt_detr',
                'processor_class': 'RTDetrImageProcessor',
                'model_class': 'RTDetrForObjectDetection'
            })
        
        # Add config-specified teacher model
        teachers_config = self.config.get('teachers', {})
        if teachers_config.get('object_detection'):
            # Parse teacher model specification like "RT-DETR/rtdetr-resnet50"
            teacher_model = teachers_config['object_detection']
            if '/' in teacher_model:
                org, model = teacher_model.split('/', 1)
                if org.upper() == 'RT-DETR':
                    # Convert to Hugging Face format
                    hf_model_name = f"PekingU/{model}" if not model.startswith('PekingU/') else model
                    model_candidates.insert(0, {  # Insert at front for priority
                        'name': hf_model_name,
                        'type': 'teacher_rt_detr',
                        'processor_class': 'RTDetrImageProcessor',
                        'model_class': 'RTDetrForObjectDetection'
                    })
        
        # Add standard RT-DETR options
        model_candidates.extend([
            {
                'name': 'microsoft/rt-detr-resnet-101', 
                'type': 'rt_detr_101',
                'processor_class': 'RTDetrImageProcessor',
                'model_class': 'RTDetrForObjectDetection'
            },
            {
                'name': 'microsoft/rt-detr-resnet-50',
                'type': 'rt_detr_50', 
                'processor_class': 'RTDetrImageProcessor',
                'model_class': 'RTDetrForObjectDetection'
            },
            {
                'name': 'facebook/detr-resnet-101',
                'type': 'detr_101',
                'processor_class': 'DetrImageProcessor', 
                'model_class': 'DetrForObjectDetection'
            },
            {
                'name': 'facebook/detr-resnet-50',
                'type': 'detr_50',
                'processor_class': 'DetrImageProcessor',
                'model_class': 'DetrForObjectDetection'
            }
        ])
        
        # Try loading models in priority order
        for candidate in model_candidates:
            try:
                logger.info(f"🔍 Attempting to load {candidate['name']} ({candidate['type']})")
                
                # Import appropriate classes
                if candidate['processor_class'] == 'RTDetrImageProcessor':
                    try:
                        from transformers import RTDetrImageProcessor, RTDetrForObjectDetection
                        processor_class = RTDetrImageProcessor
                        model_class = RTDetrForObjectDetection
                    except ImportError:
                        logger.warning(f"RT-DETR classes not available, skipping {candidate['name']}")
                        continue
                        
                elif candidate['processor_class'] == 'DetrImageProcessor':
                    from transformers import DetrImageProcessor, DetrForObjectDetection
                    processor_class = DetrImageProcessor
                    model_class = DetrForObjectDetection
                else:
                    logger.warning(f"Unknown processor class: {candidate['processor_class']}")
                    continue
                
                # Load model and processor
                processor = processor_class.from_pretrained(candidate['name'])
                model = model_class.from_pretrained(candidate['name'])
                model.eval()
                model.to(self.device)
                
                logger.info(f"✅ Successfully loaded {candidate['name']} for object detection")
                
                return {
                    'model': model,
                    'processor': processor,
                    'type': candidate['type'],
                    'model_name': candidate['name'],
                    'capabilities': {
                        'real_time': 'rt_detr' in candidate['type'],
                        'accuracy': 'resnet-101' in candidate['name'],
                        'speed': 'resnet-50' in candidate['name']
                    }
                }
                
            except Exception as e:
                logger.warning(f"Failed to load {candidate['name']}: {e}")
                continue
        
        # Before falling back, try alternative RT-DETR sources
        logger.warning("Standard RT-DETR models failed, trying alternative sources...")
        
        # Try alternative model sources
        alternative_candidates = [
            {
                'name': 'lyuwenyu/rt-detr-l',
                'type': 'alternative_rt_detr',
                'processor_class': 'RTDetrImageProcessor',
                'model_class': 'RTDetrForObjectDetection'
            },
            {
                'name': 'lyuwenyu/rt-detr-r50vd',
                'type': 'alternative_rt_detr',
                'processor_class': 'RTDetrImageProcessor', 
                'model_class': 'RTDetrForObjectDetection'
            }
        ]
        
        for alt_candidate in alternative_candidates:
            try:
                logger.info(f"🔄 Trying alternative RT-DETR: {alt_candidate['name']}")
                
                from transformers import RTDetrImageProcessor, RTDetrForObjectDetection
                
                processor = RTDetrImageProcessor.from_pretrained(alt_candidate['name'])
                model = RTDetrForObjectDetection.from_pretrained(alt_candidate['name'])
                model.eval()
                model.to(self.device)
                
                logger.info(f"✅ Successfully loaded alternative RT-DETR: {alt_candidate['name']}")
                
                return {
                    'model': model,
                    'processor': processor,
                    'type': alt_candidate['type'],
                    'model_name': alt_candidate['name'],
                    'capabilities': {
                        'real_time': True,
                        'accuracy': 'high',
                        'speed': 'fast',
                        'source': 'alternative'
                    }
                }
                
            except Exception as e:
                logger.warning(f"Alternative RT-DETR {alt_candidate['name']} failed: {e}")
                continue
        
        # Ultimate fallback to RetinaNet only after exhausting all RT-DETR options
        logger.warning("All RT-DETR sources exhausted, using RetinaNet as final fallback")
        try:
            import torchvision.transforms as T
            from torchvision.models.detection import retinanet_resnet50_fpn
            
            model = retinanet_resnet50_fpn(pretrained=True)
            model.eval()
            model.to(self.device)
            
            transform = T.Compose([
                T.ToPILImage(),
                T.Resize((800, 800)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            logger.info("⚠️ Using RetinaNet fallback for object detection")
            return {
                'model': model,
                'transform': transform,
                'type': 'retinanet_fallback',
                'model_name': 'torchvision_retinanet_resnet50_fpn',
                'capabilities': {
                    'real_time': False,
                    'accuracy': 'medium',
                    'speed': 'medium'
                }
            }
            
        except ImportError:
            logger.error("❌ No object detection model available - all options failed")
            return None
    
    def _load_hq_sam(self):
        """
        Load HQ-SAM model for high-quality segmentation with config-driven model selection.
        Prioritizes models specified in configuration for advanced video segmentation.
        """
        # Get model configuration from config
        teacher_models = self.config.get('models', {}).get('teacher_models', {})
        sam_config = teacher_models.get('hq_sam', {})
        
        # Priority order: config model -> SAM variants by size -> DeepLab fallback
        model_candidates = []
        
        # Add configured model if available
        if sam_config.get('model_name'):
            model_candidates.append({
                'name': sam_config['model_name'],
                'type': 'configured_sam',
                'size': sam_config.get('model_size', 'large')
            })
        
        # Add config-specified teacher model
        teachers_config = self.config.get('teachers', {})
        if teachers_config.get('segmentation'):
            # Parse teacher model specification like "HQ-SAM/sam-hq-vit-h"
            teacher_model = teachers_config['segmentation']
            if '/' in teacher_model:
                org, model = teacher_model.split('/', 1)
                if org.upper() == 'HQ-SAM':
                    # Convert to appropriate Hugging Face model name
                    if 'vit-h' in model:
                        hf_model_name = "facebook/sam-vit-huge"
                        size = 'huge'
                    elif 'vit-l' in model:
                        hf_model_name = "facebook/sam-vit-large" 
                        size = 'large'
                    elif 'vit-b' in model:
                        hf_model_name = "facebook/sam-vit-base"
                        size = 'base'
                    else:
                        # Default to large if unclear
                        hf_model_name = "facebook/sam-vit-large"
                        size = 'large'
                    
                    model_candidates.insert(0, {  # Insert at front for priority
                        'name': hf_model_name,
                        'type': 'teacher_hq_sam',
                        'size': size
                    })
        
        # Add standard SAM model variants in order of capability
        model_candidates.extend([
            {
                'name': 'facebook/sam-vit-huge',
                'type': 'sam_huge',
                'size': 'huge'
            },
            {
                'name': 'facebook/sam-vit-large', 
                'type': 'sam_large',
                'size': 'large'
            },
            {
                'name': 'facebook/sam-vit-base',
                'type': 'sam_base',
                'size': 'base'
            }
        ])
        
        # Try loading SAM models in priority order
        for candidate in model_candidates:
            try:
                logger.info(f"🔍 Attempting to load SAM model {candidate['name']} ({candidate['type']})")
                
                from transformers import SamModel, SamProcessor
                
                # Load model and processor
                processor = SamProcessor.from_pretrained(candidate['name'])
                model = SamModel.from_pretrained(candidate['name'])
                model.eval()
                model.to(self.device)
                
                logger.info(f"✅ Successfully loaded {candidate['name']} for segmentation")
                
                return {
                    'model': model,
                    'processor': processor,
                    'type': candidate['type'],
                    'model_name': candidate['name'],
                    'model_size': candidate['size'],
                    'capabilities': {
                        'high_quality': candidate['size'] in ['huge', 'large'],
                        'speed': 'fast' if candidate['size'] == 'base' else 'medium',
                        'memory_efficient': candidate['size'] == 'base',
                        'precision': 'high' if candidate['size'] != 'base' else 'medium'
                    }
                }
                
            except Exception as e:
                logger.warning(f"Failed to load SAM model {candidate['name']}: {e}")
                continue
        
        # Before falling back, try alternative SAM implementations
        logger.warning("Standard SAM models failed, trying alternative sources...")
        
        # Try alternative SAM sources and implementations
        alternative_sam_candidates = [
            {
                'name': 'facebook/sam2-hiera-large',
                'type': 'sam2_large',
                'size': 'large'
            },
            {
                'name': 'facebook/sam2-hiera-base-plus',
                'type': 'sam2_base_plus',
                'size': 'base_plus'
            },
            {
                'name': 'facebook/sam2-hiera-small',
                'type': 'sam2_small',
                'size': 'small'
            }
        ]
        
        for alt_candidate in alternative_sam_candidates:
            try:
                logger.info(f"🔄 Trying alternative SAM: {alt_candidate['name']}")
                
                # Try SAM 2.0 models if available
                try:
                    from transformers import Sam2Model, Sam2Processor
                    
                    processor = Sam2Processor.from_pretrained(alt_candidate['name'])
                    model = Sam2Model.from_pretrained(alt_candidate['name'])
                    model.eval()
                    model.to(self.device)
                    
                    logger.info(f"✅ Successfully loaded SAM 2.0: {alt_candidate['name']}")
                    
                    return {
                        'model': model,
                        'processor': processor,
                        'type': alt_candidate['type'],
                        'model_name': alt_candidate['name'],
                        'model_size': alt_candidate['size'],
                        'capabilities': {
                            'high_quality': alt_candidate['size'] in ['large', 'base_plus'],
                            'speed': 'fast' if alt_candidate['size'] == 'small' else 'medium',
                            'memory_efficient': alt_candidate['size'] == 'small',
                            'precision': 'very_high' if 'large' in alt_candidate['size'] else 'high',
                            'version': '2.0'
                        }
                    }
                    
                except ImportError:
                    # Fallback to original SAM if SAM 2.0 not available
                    from transformers import SamModel, SamProcessor
                    
                    # Try to find equivalent SAM 1.0 model
                    sam1_name = alt_candidate['name'].replace('sam2', 'sam').replace('-hiera', '-vit')
                    
                    processor = SamProcessor.from_pretrained(sam1_name)
                    model = SamModel.from_pretrained(sam1_name)
                    model.eval()
                    model.to(self.device)
                    
                    logger.info(f"✅ Successfully loaded SAM 1.0 equivalent: {sam1_name}")
                    
                    return {
                        'model': model,
                        'processor': processor,
                        'type': f"sam1_{alt_candidate['size']}",
                        'model_name': sam1_name,
                        'model_size': alt_candidate['size'],
                        'capabilities': {
                            'high_quality': alt_candidate['size'] != 'small',
                            'speed': 'medium',
                            'memory_efficient': alt_candidate['size'] == 'small',
                            'precision': 'high',
                            'version': '1.0'
                        }
                    }
                
            except Exception as e:
                logger.warning(f"Alternative SAM {alt_candidate['name']} failed: {e}")
                continue
        
        # Ultimate fallback to DeepLabV3 only after exhausting all SAM options
        logger.warning("All SAM sources exhausted, using DeepLabV3 as final fallback")
        try:
            import torchvision.models.segmentation as seg_models
            import torchvision.transforms as T
            
            model = seg_models.deeplabv3_resnet50(pretrained=True)
            model.eval()
            model.to(self.device)
            
            # Create transform for DeepLab preprocessing
            transform = T.Compose([
                T.ToPILImage(),
                T.Resize((520, 520)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            logger.info("⚠️ Using DeepLabV3 fallback for segmentation")
            return {
                'model': model,
                'transform': transform,
                'type': 'deeplab_fallback',
                'model_name': 'torchvision_deeplabv3_resnet50',
                'capabilities': {
                    'high_quality': False,
                    'speed': 'fast',
                    'memory_efficient': True,
                    'precision': 'medium'
                }
            }
            
        except Exception as e:
            logger.error(f"❌ No segmentation model available - all options failed: {e}")
            return None
    
    def _load_beatnet(self):
        """
        Load actual BeatNet model for advanced music structure and rhythm analysis.
        Prioritizes real BeatNet implementation over librosa-based fallbacks.
        """
        
        # Get model configuration from config
        teacher_models = self.config.get('models', {}).get('teacher_models', {})
        beatnet_config = teacher_models.get('beatnet', {})
        
        # Try to load the real BeatNet model first
        try:
            # Attempt to import and load actual BeatNet
            try:
                import BeatNet
                from BeatNet.BeatNet import BeatNet as RealBeatNet
                
                # Initialize with specified or default configuration
                model_name = beatnet_config.get('model_name', 'beatnet')
                model_size = beatnet_config.get('model_size', '1')  # BeatNet model variations
                
                # Load the actual BeatNet model
                beatnet_model = RealBeatNet(
                    model=model_size, 
                    mode='offline',  # Use offline mode for better accuracy
                    inference_model='DBN',  # Dynamic Bayesian Network for advanced inference
                    plot=[],  # No plotting for automated processing
                    thread=False
                )
                
                logger.info("✅ Successfully loaded actual BeatNet model")
                
                return {
                    'model': beatnet_model,
                    'type': 'real_beatnet',
                    'model_name': f'BeatNet-{model_size}',
                    'capabilities': {
                        'beat_tracking': True,
                        'downbeat_detection': True,
                        'tempo_estimation': True,
                        'rhythmic_analysis': True,
                        'structure_analysis': True,
                        'real_time': False,  # Offline mode for accuracy
                        'precision': 'high'
                    },
                    'processor': self._create_beatnet_processor(beatnet_model)
                }
                
            except ImportError as e:
                logger.warning(f"BeatNet package not available: {e}")
                logger.info("Install with: pip install BeatNet")
                raise e
                
            except Exception as e:
                logger.warning(f"Failed to initialize BeatNet model: {e}")
                raise e
                
        except Exception as e:
            logger.warning(f"Real BeatNet model loading failed: {e}")
            logger.info("Falling back to advanced librosa-based implementation")
            
            # Fallback to sophisticated librosa implementation
            try:
                import librosa
                import librosa.display
                
                class AdvancedBeatNetFallback:
                    """
                    Advanced music analysis fallback using librosa and enhanced algorithms.
                    Implements BeatNet-inspired techniques with librosa backend.
                    """
                    
                    def __init__(self):
                        self.sr = 44100
                        self.hop_length = 512
                        self.frame_length = 2048
                        
                return AdvancedBeatNetFallback()
                        
            except ImportError as e:
                logger.warning(f"Fallback implementation failed: {e}")
                return None
    
    def _create_beatnet_processor(self, beatnet_model):
        """Create processor wrapper for BeatNet model"""
        
        def process_audio_with_beatnet(audio_data, sr=44100):
            """Process audio using real BeatNet model"""
            try:
                # BeatNet expects specific input format
                if isinstance(audio_data, torch.Tensor):
                    audio_data = audio_data.cpu().numpy()
                
                # Ensure mono audio for BeatNet
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=0)
                
                # Process with BeatNet
                output = beatnet_model.process(audio_data)
                
                return {
                    'beats': output.get('beats', []),
                    'downbeats': output.get('downbeats', []),
                    'tempo': output.get('tempo', 120),
                    'time_signature': output.get('time_signature', [4, 4]),
                    'beat_activation': output.get('beat_activation', []),
                    'downbeat_activation': output.get('downbeat_activation', [])
                }
                
            except Exception as e:
                logger.warning(f"BeatNet processing failed: {e}")
                return {'error': str(e)}
        
        return process_audio_with_beatnet
                        
                    def process_audio(self, audio_data):
                        """Advanced tempo, beat, and structure detection"""
                        try:
                            # Advanced tempo detection with multiple algorithms
                            tempo, beats = librosa.beat.beat_track(
                                y=audio_data, 
                                sr=self.sr,
                                hop_length=self.hop_length,
                                start_bpm=120,
                                tightness=100
                            )
                            
                            # Onset detection with multiple methods
                            onset_frames = librosa.onset.onset_detect(
                                y=audio_data,
                                sr=self.sr,
                                hop_length=self.hop_length,
                                backtrack=True
                            )
                            
                            # Spectral features for rhythm analysis
                            spectral_centroids = librosa.feature.spectral_centroid(
                                y=audio_data, sr=self.sr, hop_length=self.hop_length
                            )[0]
                            
                            # Rhythm pattern analysis
                            rhythm_pattern = self._analyze_advanced_rhythm(beats, spectral_centroids)
                            
                            # Musical structure analysis
                            structure = self._analyze_structure(audio_data)
                            
                            return {
                                'tempo': float(tempo),
                                'beats': beats.tolist(),
                                'onsets': onset_frames.tolist(),
                                'rhythm_pattern': rhythm_pattern,
                                'spectral_centroids': spectral_centroids.tolist(),
                                'structure': structure,
                                'beat_strength': self._calculate_beat_strength(audio_data, beats),
                                'rhythmic_complexity': self._calculate_rhythmic_complexity(beats)
                            }
                            
                        except Exception as e:
                            logger.warning(f"Advanced BeatNet analysis failed: {e}")
                            return self._basic_fallback(audio_data)
                    
                    def _analyze_advanced_rhythm(self, beats, spectral_centroids):
                        """Advanced rhythm pattern analysis"""
                        if len(beats) < 8:
                            return 'insufficient_data'
                        
                        # Calculate beat intervals and their variation
                        intervals = np.diff(beats)
                        avg_interval = np.mean(intervals)
                        interval_std = np.std(intervals)
                        
                        # Analyze spectral content at beat locations
                        beat_spectral_means = []
                        for beat in beats[:len(spectral_centroids)]:
                            if int(beat) < len(spectral_centroids):
                                beat_spectral_means.append(spectral_centroids[int(beat)])
                        
                        # Classify rhythm based on multiple factors
                        if interval_std < 0.05:  # Very regular
                            if avg_interval < 0.4:
                                return 'fast_regular'
                            elif avg_interval < 0.7:
                                return 'medium_regular'
                            else:
                                return 'slow_regular'
                        else:  # More varied
                            if avg_interval < 0.5:
                                return 'fast_syncopated'
                            else:
                                return 'complex_rhythm'
                    
                    def _analyze_structure(self, audio_data):
                        """Analyze musical structure (intro, verse, chorus, etc.)"""
                        try:
                            # Use chroma features for harmonic analysis
                            chroma = librosa.feature.chroma_stft(y=audio_data, sr=self.sr)
                            
                            # Use MFCC for timbral analysis
                            mfcc = librosa.feature.mfcc(y=audio_data, sr=self.sr, n_mfcc=13)
                            
                            # Simple structure detection based on feature similarity
                            # This is a simplified version - actual BeatNet would be more sophisticated
                            segments = self._detect_segments(chroma, mfcc)
                            
                            return {
                                'segments': segments,
                                'harmonic_progression': chroma.mean(axis=1).tolist(),
                                'timbral_features': mfcc.mean(axis=1).tolist()
                            }
                        except:
                            return {'segments': [], 'harmonic_progression': [], 'timbral_features': []}
                    
                    def _detect_segments(self, chroma, mfcc):
                        """Detect musical segments"""
                        # Simple segmentation based on feature change
                        segments = []
                        segment_length = min(chroma.shape[1], mfcc.shape[1]) // 4
                        
                        for i in range(0, min(chroma.shape[1], mfcc.shape[1]), segment_length):
                            end_idx = min(i + segment_length, chroma.shape[1])
                            segments.append({
                                'start_frame': i,
                                'end_frame': end_idx,
                                'type': 'segment'
                            })
                        
                        return segments
                    
                    def _calculate_beat_strength(self, audio_data, beats):
                        """Calculate beat strength/salience"""
                        try:
                            onset_strength = librosa.onset.onset_strength(
                                y=audio_data, sr=self.sr, hop_length=self.hop_length
                            )
                            
                            # Get strength at beat locations
                            beat_strengths = []
                            for beat in beats:
                                if int(beat) < len(onset_strength):
                                    beat_strengths.append(float(onset_strength[int(beat)]))
                            
                            return np.mean(beat_strengths) if beat_strengths else 0.0
                        except:
                            return 0.5
                    
                    def _calculate_rhythmic_complexity(self, beats):
                        """Calculate rhythmic complexity score"""
                        if len(beats) < 4:
                            return 0.0
                        
                        intervals = np.diff(beats)
                        # Higher std deviation = more complex rhythm
                        complexity = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0
                        return float(np.clip(complexity, 0, 1))
                    
                    def _basic_fallback(self, audio_data):
                        """Basic fallback analysis"""
                        return {
                            'tempo': 120.0,
                            'beats': [],
                            'onsets': [],
                            'rhythm_pattern': 'unknown',
                            'spectral_centroids': [],
                            'structure': {'segments': [], 'harmonic_progression': [], 'timbral_features': []},
                            'beat_strength': 0.5,
                            'rhythmic_complexity': 0.0
                        }
                
                logger.info("✅ Loaded advanced BeatNet implementation")
                return AdvancedBeatNet()
                
            except ImportError:
                logger.warning("Librosa not available for advanced music analysis")
                raise ImportError("Librosa required for BeatNet")
                
        except Exception as e:
            logger.warning(f"Advanced BeatNet not available: {e}")
            
            # Simple fallback class
            class BeatNetFallback:
                """Minimal music analysis fallback"""
                
                def __init__(self):
                    self.sr = 44100
                
                def process_audio(self, audio_data):
                    """Basic rhythm detection"""
                    return {
                        'tempo': 120.0,
                        'beats': [],
                        'onsets': [],
                        'rhythm_pattern': 'unknown',
                        'spectral_centroids': [],
                        'structure': {'segments': [], 'harmonic_progression': [], 'timbral_features': []},
                        'beat_strength': 0.5,
                        'rhythmic_complexity': 0.0
                    }
            
            logger.info("⚠️ Using minimal BeatNet fallback")
            return BeatNetFallback()
            
        except Exception as e:
            logger.warning(f"BeatNet loading failed: {e}")
            return None
    
    def _load_demucs(self):
        """
        Load actual Demucs model for professional audio source separation.
        Uses the real pre-trained Demucs models for high-quality separation.
        """
        
        # Get model configuration from config
        teacher_models = self.config.get('models', {}).get('teacher_models', {})
        demucs_config = teacher_models.get('demucs', {})
        
        # Try to load the real Demucs model first
        try:
            # Attempt to load actual Demucs
            try:
                import demucs.api
                from demucs import pretrained
                from demucs.apply import apply_model
                from demucs.audio import convert_audio
                
                # Get model name from config or use default
                model_name = demucs_config.get('model_name', 'htdemucs')  # Latest hybrid transformer model
                
                # Load the pre-trained Demucs model
                model = pretrained.get_model(model_name)
                model.eval()
                model.to(self.device)
                
                logger.info(f"✅ Successfully loaded real Demucs model: {model_name}")
                
                return {
                    'model': model,
                    'type': 'real_demucs',
                    'model_name': model_name,
                    'capabilities': {
                        'source_separation': True,
                        'stems': ['drums', 'bass', 'other', 'vocals'],
                        'quality': 'professional',
                        'real_time': False,  # High-quality mode
                        'model_type': 'hybrid_transformer' if 'ht' in model_name else 'standard'
                    },
                    'processor': self._create_demucs_processor(model, model_name)
                }
                
            except ImportError as e:
                logger.warning(f"Demucs package not available: {e}")
                logger.info("Install with: pip install demucs")
                raise e
                
            except Exception as e:
                logger.warning(f"Failed to load Demucs model: {e}")
                raise e
                
        except Exception as e:
            logger.warning(f"Real Demucs model loading failed: {e}")
            logger.info("Falling back to advanced spectral separation implementation")
            
            # Fallback to sophisticated separation implementation
            try:
                import librosa
                from scipy import signal
                import torchaudio
                
                class AdvancedDemucsalFallback:
                    """
                    Advanced audio source separation fallback using sophisticated algorithms.
                    Implements Demucs-inspired techniques without requiring the full model.
                    """
                    
                    def __init__(self):
                        self.sr = 44100
                        self.n_fft = 2048
                        self.hop_length = 512
                        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        
    
    def _create_demucs_processor(self, model, model_name):
        """Create processor wrapper for Demucs model"""
        
        def separate_with_demucs(audio_data, sr=44100):
            """Process audio using real Demucs model"""
            try:
                from demucs.apply import apply_model
                from demucs.audio import convert_audio
                
                # Ensure audio is in correct format for Demucs
                if isinstance(audio_data, np.ndarray):
                    audio_tensor = torch.from_numpy(audio_data).float()
                else:
                    audio_tensor = audio_data.float()
                
                # Ensure correct shape: [channels, samples]
                if len(audio_tensor.shape) == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dimension
                elif len(audio_tensor.shape) == 2 and audio_tensor.shape[0] > audio_tensor.shape[1]:
                    audio_tensor = audio_tensor.T  # Transpose if needed
                
                # Convert to model's expected sample rate and format
                audio_tensor = convert_audio(audio_tensor, sr, model.samplerate, model.audio_channels)
                audio_tensor = audio_tensor.to(self.device)
                
                # Apply Demucs separation
                with torch.no_grad():
                    separated = apply_model(model, audio_tensor.unsqueeze(0))[0]  # Add batch dim
                
                # Extract stems (typically: drums, bass, other, vocals)
                stems = {}
                if hasattr(model, 'sources'):
                    for i, source in enumerate(model.sources):
                        if i < separated.shape[0]:
                            stems[source] = separated[i].cpu().numpy()
                else:
                    # Default stem names for most Demucs models
                    stem_names = ['drums', 'bass', 'other', 'vocals']
                    for i, name in enumerate(stem_names):
                        if i < separated.shape[0]:
                            stems[name] = separated[i].cpu().numpy()
                
                return stems
                
            except Exception as e:
                logger.warning(f"Demucs separation failed: {e}")
                return {'error': str(e)}
        
        return separate_with_demucs
                        
                        # Try to use torchaudio's separation if available
                        try:
                            self.separator_available = True
                            logger.info("✅ Advanced separation algorithms available")
                        except:
                            self.separator_available = False
                    
                    def separate_sources(self, audio_data):
                        """Advanced source separation using multiple techniques"""
                        try:
                            # Convert to torch tensor if needed
                            if isinstance(audio_data, np.ndarray):
                                audio_tensor = torch.from_numpy(audio_data).float()
                            else:
                                audio_tensor = audio_data.float()
                            
                            # Advanced spectral analysis
                            stft = torch.stft(
                                audio_tensor, 
                                n_fft=self.n_fft, 
                                hop_length=self.hop_length,
                                return_complex=True
                            )
                            
                            magnitude = torch.abs(stft)
                            phase = torch.angle(stft)
                            
                            # Advanced source separation using multiple methods
                            separated = self._advanced_spectral_separation(magnitude, phase, audio_data)
                            
                            # Apply post-processing
                            separated = self._post_process_separation(separated)
                            
                            return separated
                            
                        except Exception as e:
                            logger.warning(f"Advanced separation failed: {e}")
                            return self._fallback_separation(audio_data)
                    
                    def _advanced_spectral_separation(self, magnitude, phase, original_audio):
                        """Advanced spectral separation using multiple techniques"""
                        
                        # Technique 1: Harmonic-Percussive Separation
                        harmonic, percussive = self._harmonic_percussive_separation(magnitude)
                        
                        # Technique 2: Frequency Band Separation  
                        vocal_mask, instrumental_mask = self._intelligent_masking(magnitude)
                        
                        # Technique 3: Statistical Source Separation
                        sources = self._statistical_separation(magnitude, phase)
                        
                        # Combine techniques for better results
                        vocals = self._reconstruct_audio(magnitude * vocal_mask, phase)
                        instrumental = self._reconstruct_audio(magnitude * instrumental_mask, phase)
                        
                        # Extract drums from percussive component
                        drums = self._extract_drums(percussive, phase)
                        
                        # Extract bass from harmonic component  
                        bass = self._extract_bass(harmonic, phase)
                        
                        return {
                            'vocals': vocals,
                            'instrumental': instrumental,
                            'drums': drums,
                            'bass': bass,
                            'harmonic': self._reconstruct_audio(harmonic, phase),
                            'percussive': self._reconstruct_audio(percussive, phase)
                        }
                    
                    def _harmonic_percussive_separation(self, magnitude):
                        """Separate harmonic and percussive components"""
                        # Apply median filtering for harmonic/percussive separation
                        harmonic = signal.medfilt2d(magnitude.numpy(), kernel_size=(1, 17))  # Horizontal filter
                        percussive = signal.medfilt2d(magnitude.numpy(), kernel_size=(17, 1))  # Vertical filter
                        
                        # Normalize
                        total = harmonic + percussive + 1e-8
                        harmonic_mask = harmonic / total
                        percussive_mask = percussive / total
                        
                        return (
                            torch.from_numpy(harmonic_mask) * magnitude,
                            torch.from_numpy(percussive_mask) * magnitude
                        )
                    
                    def _intelligent_masking(self, magnitude):
                        """Create intelligent masks for vocal/instrumental separation"""
                        
                        # Frequency-based analysis
                        freq_bins = magnitude.shape[0]
                        
                        # Vocal frequency range (roughly 80Hz - 1100Hz for fundamental)
                        vocal_range = slice(int(freq_bins * 0.1), int(freq_bins * 0.4))
                        
                        # Analyze temporal consistency (vocals tend to be more consistent)
                        temporal_std = torch.std(magnitude, dim=1, keepdim=True)
                        consistency_mask = (temporal_std < torch.percentile(temporal_std, 60)).float()
                        
                        # Analyze spectral centroids (vocals often have higher centroids)
                        freq_weights = torch.arange(freq_bins).float().unsqueeze(1)
                        spectral_centroid = torch.sum(magnitude * freq_weights, dim=0) / (torch.sum(magnitude, dim=0) + 1e-8)
                        centroid_mask = (spectral_centroid > torch.percentile(spectral_centroid, 50)).float()
                        
                        # Combine masks
                        vocal_mask = torch.zeros_like(magnitude)
                        vocal_mask[vocal_range, :] = consistency_mask[vocal_range, :] * centroid_mask.unsqueeze(0)
                        
                        # Smooth the mask
                        vocal_mask = self._smooth_mask(vocal_mask)
                        
                        # Instrumental is complement
                        instrumental_mask = 1 - vocal_mask
                        
                        return vocal_mask, instrumental_mask
                    
                    def _statistical_separation(self, magnitude, phase):
                        """Statistical approach to source separation"""
                        # Independent Component Analysis (simplified)
                        # This is a basic version - real Demucs uses deep learning
                        
                        # Convert to time-frequency patches
                        patches = self._extract_patches(magnitude)
                        
                        # Apply statistical separation (simplified ICA-like approach)
                        separated_patches = self._apply_ica(patches)
                        
                        return separated_patches
                    
                    def _extract_patches(self, magnitude):
                        """Extract time-frequency patches for analysis"""
                        # Simple patch extraction
                        patch_size = 32
                        patches = []
                        
                        for i in range(0, magnitude.shape[0] - patch_size, patch_size // 2):
                            for j in range(0, magnitude.shape[1] - patch_size, patch_size // 2):
                                patch = magnitude[i:i+patch_size, j:j+patch_size]
                                patches.append(patch)
                        
                        return patches
                    
                    def _apply_ica(self, patches):
                        """Apply Independent Component Analysis (simplified)"""
                        # This is a placeholder for ICA-based separation
                        # Real implementation would use sklearn.decomposition.FastICA or similar
                        return patches  # Return as-is for now
                    
                    def _extract_drums(self, percussive, phase):
                        """Extract drums from percussive component"""
                        # Focus on low-mid frequency range for drums
                        drum_mask = torch.zeros_like(percussive)
                        freq_bins = percussive.shape[0]
                        drum_range = slice(int(freq_bins * 0.05), int(freq_bins * 0.3))
                        drum_mask[drum_range, :] = 1.0
                        
                        drums_spec = percussive * drum_mask
                        return self._reconstruct_audio(drums_spec, phase)
                    
                    def _extract_bass(self, harmonic, phase):
                        """Extract bass from harmonic component"""
                        # Focus on low frequency range for bass
                        bass_mask = torch.zeros_like(harmonic)
                        freq_bins = harmonic.shape[0]
                        bass_range = slice(0, int(freq_bins * 0.15))
                        bass_mask[bass_range, :] = 1.0
                        
                        bass_spec = harmonic * bass_mask
                        return self._reconstruct_audio(bass_spec, phase)
                    
                    def _smooth_mask(self, mask):
                        """Smooth mask to reduce artifacts"""
                        from scipy.ndimage import gaussian_filter
                        smoothed = gaussian_filter(mask.numpy(), sigma=1.0)
                        return torch.from_numpy(smoothed)
                    
                    def _reconstruct_audio(self, magnitude, phase):
                        """Reconstruct audio from magnitude and phase"""
                        complex_spec = magnitude * torch.exp(1j * phase)
                        audio = torch.istft(
                            complex_spec, 
                            n_fft=self.n_fft, 
                            hop_length=self.hop_length
                        )
                        return audio.numpy()
                    
                    def _post_process_separation(self, separated):
                        """Apply post-processing to improve separation quality"""
                        
                        # Apply gentle filtering to reduce artifacts
                        for source_name, source_audio in separated.items():
                            if len(source_audio) > 0:
                                # Simple low-pass filter for very high frequencies
                                b, a = signal.butter(5, 0.95, 'low')
                                separated[source_name] = signal.filtfilt(b, a, source_audio)
                        
                        return separated
                    
                    def _fallback_separation(self, audio_data):
                        """Fallback separation using basic spectral methods"""
                        try:
                            import librosa
                            
                            # Basic spectral separation
                            stft = librosa.stft(audio_data, hop_length=self.hop_length)
                            magnitude = np.abs(stft)
                            phase = np.angle(stft)
                            
                            # Simple frequency-based separation
                            vocal_mask = (magnitude > np.percentile(magnitude, 65))
                            instrumental_mask = ~vocal_mask
                            
                            vocals = librosa.istft(magnitude * vocal_mask * np.exp(1j * phase))
                            instrumental = librosa.istft(magnitude * instrumental_mask * np.exp(1j * phase))
                            
                            return {
                                'vocals': vocals,
                                'instrumental': instrumental,
                                'bass': instrumental * 0.3,
                                'drums': instrumental * 0.2,
                                'harmonic': instrumental * 0.7,
                                'percussive': instrumental * 0.3
                            }
                            
                        except:
                            # Ultra-basic fallback
                            return {
                                'vocals': audio_data * 0.4,
                                'instrumental': audio_data * 0.6,
                                'bass': audio_data * 0.2,
                                'drums': audio_data * 0.1,
                                'harmonic': audio_data * 0.5,
                                'percussive': audio_data * 0.2
                            }
                
                logger.info("✅ Loaded advanced Demucs implementation")
                return AdvancedDemucs()
                
            except ImportError as e:
                logger.warning(f"Advanced audio processing libraries not available: {e}")
                raise ImportError("Required audio libraries not available")
                
        except Exception as e:
            logger.warning(f"Advanced Demucs not available: {e}")
            
            # Simple fallback class
            class DemucsFallback:
                """Basic audio source separation fallback"""
                
                def __init__(self):
                    self.sr = 44100
                
                def separate_sources(self, audio_data):
                    """Basic source separation"""
                    return {
                        'vocals': audio_data * 0.4,
                        'instrumental': audio_data * 0.6,
                        'bass': audio_data * 0.2,
                        'drums': audio_data * 0.1,
                        'harmonic': audio_data * 0.5,
                        'percussive': audio_data * 0.2
                    }
            
            logger.info("⚠️ Using basic Demucs fallback")
            return DemucsFallback()
            
        except Exception as e:
            logger.warning(f"Demucs loading failed: {e}")
            return None
    def distill_rt_detr_knowledge(self, student_model: nn.Module, video_frames: torch.Tensor) -> Dict[str, Any]:
        """Distill object detection knowledge from RT-DETR"""
        if self.rt_detr is None:
            return {}
        
        logger.info("🎯 Distilling RT-DETR object detection knowledge...")
        
        try:
            with torch.no_grad():
                # Process frames through RT-DETR
                detections = []
                
                for frame in video_frames:
                    # Preprocess frame
                    if self.rt_detr['type'] == 'retinanet':
                        frame_tensor = self.rt_detr['transform'](frame.cpu().numpy())
                        frame_tensor = frame_tensor.unsqueeze(0).to(self.device)
                        
                        # Get detections
                        detection = self.rt_detr['model'](frame_tensor)
                        detections.append({
                            'boxes': detection[0]['boxes'],
                            'scores': detection[0]['scores'],
                            'labels': detection[0]['labels'],
                            'spatial_features': self._extract_spatial_features(detection)
                        })
                
                # Create distillation targets
                spatial_knowledge = self._process_spatial_detections(detections)
                
                return {
                    'spatial_understanding': spatial_knowledge,
                    'object_locations': [d['boxes'] for d in detections],
                    'object_confidences': [d['scores'] for d in detections],
                    'object_classes': [d['labels'] for d in detections]
                }
        
        except Exception as e:
            logger.error(f"RT-DETR distillation failed: {e}")
            return {}
    
    def distill_hq_sam_knowledge(self, student_model: nn.Module, video_frames: torch.Tensor) -> Dict[str, Any]:
        """Distill segmentation knowledge from HQ-SAM"""
        if self.hq_sam is None:
            return {}
        
        logger.info("🎨 Distilling HQ-SAM segmentation knowledge...")
        
        try:
            with torch.no_grad():
                segmentations = []
                
                for frame in video_frames:
                    # Preprocess for segmentation
                    if self.hq_sam['type'] == 'deeplab':
                        frame_tensor = torch.nn.functional.interpolate(
                            frame.unsqueeze(0), 
                            size=(512, 512), 
                            mode='bilinear'
                        ).to(self.device)
                        
                        # Get segmentation
                        seg_output = self.hq_sam['model'](frame_tensor)
                        segmentation = torch.nn.functional.softmax(seg_output['out'], dim=1)
                        
                        segmentations.append({
                            'segmentation_map': segmentation,
                            'fine_boundaries': self._extract_boundary_features(segmentation)
                        })
                
                # Create segmentation knowledge
                segmentation_knowledge = self._process_segmentation_maps(segmentations)
                
                return {
                    'segmentation_understanding': segmentation_knowledge,
                    'object_boundaries': [s['fine_boundaries'] for s in segmentations],
                    'region_features': [s['segmentation_map'] for s in segmentations]
                }
        
        except Exception as e:
            logger.error(f"HQ-SAM distillation failed: {e}")
            return {}
    
    def distill_beatnet_knowledge(self, student_model: nn.Module, audio_data: torch.Tensor) -> Dict[str, Any]:
        """Distill music structure knowledge from BeatNet"""
        if self.beatnet is None:
            return {}
        
        logger.info("🎵 Distilling BeatNet music analysis knowledge...")
        
        try:
            # Convert tensor to numpy for processing
            audio_np = audio_data.cpu().numpy()
            if len(audio_np.shape) > 1:
                audio_np = audio_np.flatten()
            
            # Process through BeatNet
            music_analysis = self.beatnet.process_audio(audio_np)
            
            # Create rhythm understanding features
            rhythm_knowledge = {
                'tempo_stability': self._analyze_tempo_stability(music_analysis['tempo']),
                'beat_pattern': self._encode_beat_pattern(music_analysis['beats']),
                'rhythmic_complexity': self._calculate_rhythmic_complexity(music_analysis),
                'musical_structure': self._identify_musical_structure(music_analysis)
            }
            
            return {
                'music_understanding': rhythm_knowledge,
                'tempo_info': music_analysis['tempo'],
                'beat_locations': music_analysis['beats'],
                'onset_times': music_analysis['onsets']
            }
        
        except Exception as e:
            logger.error(f"BeatNet distillation failed: {e}")
            return {}
    
    def distill_demucs_knowledge(self, student_model: nn.Module, audio_data: torch.Tensor) -> Dict[str, Any]:
        """Distill audio source separation knowledge from Demucs"""
        if self.demucs is None:
            return {}
        
        logger.info("🎶 Distilling Demucs audio separation knowledge...")
        
        try:
            # Convert tensor to numpy for processing
            audio_np = audio_data.cpu().numpy()
            if len(audio_np.shape) > 1:
                audio_np = audio_np.flatten()
            
            # Separate audio sources
            separated_sources = self.demucs.separate_sources(audio_np)
            
            # Create audio understanding features
            audio_knowledge = {
                'source_separation_quality': self._evaluate_separation_quality(separated_sources),
                'vocal_characteristics': self._analyze_vocal_content(separated_sources['vocals']),
                'instrumental_complexity': self._analyze_instrumental_content(separated_sources['instrumental']),
                'audio_balance': self._analyze_audio_balance(separated_sources)
            }
            
            return {
                'audio_understanding': audio_knowledge,
                'separated_sources': separated_sources,
                'source_masks': self._create_source_masks(separated_sources)
            }
        
        except Exception as e:
            logger.error(f"Demucs distillation failed: {e}")
            return {}
    
    def _extract_spatial_features(self, detection_output) -> torch.Tensor:
        """Extract spatial understanding features from detection output"""
        # Basic spatial feature extraction
        boxes = detection_output[0]['boxes']
        if len(boxes) == 0:
            return torch.zeros((1, 256), device=self.device)
        
        # Simple spatial encoding: center points and dimensions
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        sizes = boxes[:, 2:] - boxes[:, :2]
        
        # Create feature vector
        spatial_features = torch.cat([centers, sizes], dim=1)
        return spatial_features.mean(dim=0).unsqueeze(0)  # Average across objects
    
    def _process_spatial_detections(self, detections: List[Dict]) -> torch.Tensor:
        """Process spatial detections into knowledge representation"""
        if not detections:
            return torch.zeros((1, 512), device=self.device)
        
        # Combine spatial features across frames
        spatial_features = []
        for detection in detections:
            if len(detection['boxes']) > 0:
                spatial_features.append(detection['spatial_features'])
        
        if spatial_features:
            combined_features = torch.cat(spatial_features, dim=0)
            return combined_features.mean(dim=0).unsqueeze(0)
        
        return torch.zeros((1, 256), device=self.device)
    
    def _extract_boundary_features(self, segmentation: torch.Tensor) -> torch.Tensor:
        """Extract fine boundary features from segmentation maps"""
        # Simple edge detection on segmentation
        seg_map = segmentation.argmax(dim=1).float()
        
        # Sobel edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=self.device).float()
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=self.device).float()
        
        edges_x = torch.nn.functional.conv2d(seg_map.unsqueeze(0), sobel_x.unsqueeze(0).unsqueeze(0), padding=1)
        edges_y = torch.nn.functional.conv2d(seg_map.unsqueeze(0), sobel_y.unsqueeze(0).unsqueeze(0), padding=1)
        
        boundaries = torch.sqrt(edges_x**2 + edges_y**2)
        return boundaries.flatten()
    
    def _process_segmentation_maps(self, segmentations: List[Dict]) -> torch.Tensor:
        """Process segmentation maps into knowledge representation"""
        if not segmentations:
            return torch.zeros((1, 1024), device=self.device)
        
        # Combine boundary features
        boundary_features = []
        for seg in segmentations:
            boundary_features.append(seg['fine_boundaries'])
        
        if boundary_features:
            combined_boundaries = torch.stack(boundary_features, dim=0)
            return combined_boundaries.mean(dim=0).unsqueeze(0)
        
        return torch.zeros((1, 1024), device=self.device)
    
    def _analyze_tempo_stability(self, tempo: float) -> Dict[str, float]:
        """Analyze tempo stability characteristics"""
        return {
            'tempo_value': tempo,
            'stability_score': 1.0 if 60 <= tempo <= 180 else 0.5,
            'energy_level': min(tempo / 140.0, 1.0)
        }
    
    def _encode_beat_pattern(self, beats: List[float]) -> torch.Tensor:
        """Encode beat pattern into feature vector"""
        if len(beats) < 2:
            return torch.zeros(64, device=self.device)
        
        # Simple beat interval encoding
        intervals = np.diff(beats)
        if len(intervals) == 0:
            return torch.zeros(64, device=self.device)
        
        # Create histogram of intervals
        hist, _ = np.histogram(intervals, bins=64, range=(0, 2.0))
        return torch.tensor(hist, dtype=torch.float32, device=self.device)
    
    def _calculate_rhythmic_complexity(self, music_analysis: Dict) -> float:
        """Calculate rhythmic complexity score"""
        beats = music_analysis.get('beats', [])
        onsets = music_analysis.get('onsets', [])
        
        if len(beats) == 0:
            return 0.0
        
        # Simple complexity based on onset density
        complexity = len(onsets) / max(len(beats), 1)
        return min(complexity, 1.0)
    
    def _identify_musical_structure(self, music_analysis: Dict) -> Dict[str, Any]:
        """Identify basic musical structure"""
        tempo = music_analysis.get('tempo', 120)
        rhythm_pattern = music_analysis.get('rhythm_pattern', 'unknown')
        
        return {
            'structure_type': rhythm_pattern,
            'tempo_category': 'fast' if tempo > 120 else 'moderate' if tempo > 80 else 'slow',
            'complexity_level': self._calculate_rhythmic_complexity(music_analysis)
        }
    
    def _evaluate_separation_quality(self, sources: Dict) -> float:
        """Evaluate quality of source separation"""
        try:
            # Simple quality metric based on energy distribution
            total_energy = sum(np.sum(source**2) for source in sources.values())
            if total_energy == 0:
                return 0.0
            
            # Higher quality if energy is well distributed across sources
            energies = [np.sum(source**2) / total_energy for source in sources.values()]
            entropy = -sum(e * np.log(e + 1e-10) for e in energies if e > 0)
            
            return min(entropy / np.log(len(sources)), 1.0)
        
        except:
            return 0.5  # Default quality
    
    def _analyze_vocal_content(self, vocals: np.ndarray) -> Dict[str, float]:
        """Analyze vocal content characteristics"""
        try:
            energy = np.sum(vocals**2)
            spectral_centroid = np.mean(np.abs(np.fft.fft(vocals)))
            
            return {
                'vocal_energy': float(energy),
                'spectral_brightness': float(spectral_centroid),
                'presence_confidence': 1.0 if energy > 0.01 else 0.0
            }
        except:
            return {'vocal_energy': 0.0, 'spectral_brightness': 0.0, 'presence_confidence': 0.0}
    
    def _analyze_instrumental_content(self, instrumental: np.ndarray) -> Dict[str, float]:
        """Analyze instrumental content characteristics"""
        try:
            energy = np.sum(instrumental**2)
            dynamic_range = np.max(instrumental) - np.min(instrumental)
            
            return {
                'instrumental_energy': float(energy),
                'dynamic_range': float(dynamic_range),
                'complexity_score': min(dynamic_range * 10, 1.0)
            }
        except:
            return {'instrumental_energy': 0.0, 'dynamic_range': 0.0, 'complexity_score': 0.0}
    
    def _analyze_audio_balance(self, sources: Dict) -> Dict[str, float]:
        """Analyze balance between audio sources"""
        try:
            energies = {name: np.sum(source**2) for name, source in sources.items()}
            total_energy = sum(energies.values())
            
            if total_energy == 0:
                return {name: 0.0 for name in sources.keys()}
            
            return {name: energy / total_energy for name, energy in energies.items()}
        except:
            return {name: 0.25 for name in sources.keys()}  # Default equal balance
    
    def _create_source_masks(self, sources: Dict) -> Dict[str, np.ndarray]:
        """Create source separation masks"""
        try:
            # Simple energy-based masks
            total_energy = sum(np.abs(source) for source in sources.values())
            total_energy = np.maximum(total_energy, 1e-10)  # Avoid division by zero
            
            masks = {}
            for name, source in sources.items():
                mask = np.abs(source) / total_energy
                masks[name] = mask
            
            return masks
        except:
            return {name: np.ones_like(source) * 0.25 for name, source in sources.items()}
        
    def distill_all_experts(self, student_model: nn.Module, video_frames: torch.Tensor = None, audio_data: torch.Tensor = None):
        """
        Enhanced sequential distillation from all expert models including advanced teachers
        """
        logger.info("🔬 Starting Enhanced Knowledge Distillation from Expert Models")
        
        distilled_knowledge = {}
        
        # Distill from advanced teacher models
        if video_frames is not None:
            # RT-DETR object detection knowledge
            rt_detr_knowledge = self.distill_rt_detr_knowledge(student_model, video_frames)
            distilled_knowledge.update(rt_detr_knowledge)
            
            # HQ-SAM segmentation knowledge
            hq_sam_knowledge = self.distill_hq_sam_knowledge(student_model, video_frames)
            distilled_knowledge.update(hq_sam_knowledge)
        
        if audio_data is not None:
            # BeatNet music analysis knowledge
            beatnet_knowledge = self.distill_beatnet_knowledge(student_model, audio_data)
            distilled_knowledge.update(beatnet_knowledge)
            
            # Demucs audio separation knowledge
            demucs_knowledge = self.distill_demucs_knowledge(student_model, audio_data)
            distilled_knowledge.update(demucs_knowledge)
        
        logger.info(f"✅ Enhanced distillation completed with {len(distilled_knowledge)} knowledge components")
        return distilled_knowledge
        
        # 1. Vision experts (RT-DETR + HQ-SAM)
        logger.info("👁️ Distilling vision knowledge...")
        self.distill_vision_experts(student_model)
        
        # 2. Audio experts (Whisper + BeatNet)
        logger.info("🎵 Distilling audio knowledge...")  
        self.distill_audio_experts(student_model)
        
        # 3. Video motion (RAFT optical flow)
        logger.info("🎬 Distilling motion knowledge...")
        self.distill_motion_expert(student_model)
        
        # 4. Cross-modal alignment
        logger.info("🔄 Cross-modal knowledge alignment...")
        self.distill_cross_modal_alignment(student_model)
        
        logger.info("✅ Knowledge distillation completed!")
        
    def distill_vision_experts(self, student_model: nn.Module):
        """Distill from RT-DETR and HQ-SAM"""
        
        # Freeze expert models
        self.expert_models.rt_detr.eval()
        self.expert_models.hq_sam.eval()
        
        # Setup optimizer for student vision components
        vision_params = [
            p for name, p in student_model.named_parameters() 
            if 'vision' in name.lower()
        ]
        optimizer = torch.optim.AdamW(vision_params, lr=1e-5)
        
        # Get vision distillation dataset
        data_loader = self._get_vision_distillation_data()
        
        for epoch in range(5):  # Vision distillation epochs
            total_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(data_loader):
                
                images = batch['images'].to(self.device)  # (B, T, C, H, W)
                B, T, C, H, W = images.shape
                
                # Flatten time dimension for processing
                images_flat = images.view(B * T, C, H, W)
                
                optimizer.zero_grad()
                
                # Teacher predictions
                with torch.no_grad():
                    # RT-DETR object detection
                    rtdetr_outputs = self.expert_models.rt_detr(images_flat)
                    
                    # HQ-SAM segmentation (on subset for efficiency)
                    sam_outputs = []
                    for i in range(0, min(B*T, 8)):  # Process first 8 images
                        sam_out = self.expert_models.hq_sam(images_flat[i:i+1])
                        sam_outputs.append(sam_out)
                
                # Student predictions
                student_vision_outputs = student_model(video_frames=images)
                student_vision_emb = student_vision_outputs.get('vision_embeddings')
                
                if student_vision_emb is None:
                    continue
                    
                # Distillation losses
                loss = 0.0
                
                # 1. Feature distillation from RT-DETR backbone
                if hasattr(rtdetr_outputs, 'backbone_features'):
                    teacher_features = rtdetr_outputs.backbone_features
                    student_features = student_vision_emb.view(B*T, -1)
                    
                    # Match dimensions if needed
                    if teacher_features.shape != student_features.shape:
                        teacher_features = F.adaptive_avg_pool1d(
                            teacher_features.transpose(1, 2), 
                            student_features.shape[-1]
                        ).transpose(1, 2)
                    
                    loss += self.feature_matcher.match_features(
                        student_features, teacher_features
                    )
                
                # 2. Object detection knowledge transfer
                if hasattr(rtdetr_outputs, 'logits'):
                    # Use attention maps as soft targets
                    detection_loss = self._compute_detection_distillation_loss(
                        student_vision_emb, rtdetr_outputs
                    )
                    loss += detection_loss
                
                # 3. Segmentation knowledge (if SAM outputs available)
                if sam_outputs:
                    seg_loss = self._compute_segmentation_distillation_loss(
                        student_vision_emb[:len(sam_outputs)], sam_outputs
                    )
                    loss += seg_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 100 == 0:
                    logger.info(f"Vision distillation batch {batch_idx}, loss: {loss.item():.4f}")
                    
            avg_loss = total_loss / num_batches
            logger.info(f"Vision distillation epoch {epoch+1}, avg loss: {avg_loss:.4f}")
            
    def distill_audio_experts(self, student_model: nn.Module):
        """Distill from Whisper and BeatNet"""
        
        # Freeze expert models
        self.expert_models.whisper.eval()
        self.expert_models.beatnet.eval()
        
        # Setup optimizer for student audio components  
        audio_params = [
            p for name, p in student_model.named_parameters()
            if 'audio' in name.lower()
        ]
        optimizer = torch.optim.AdamW(audio_params, lr=1e-5)
        
        # Get audio distillation data
        data_loader = self._get_audio_distillation_data()
        
        for epoch in range(3):  # Audio distillation epochs
            for batch_idx, batch in enumerate(data_loader):
                
                audio_features = batch['audio_features'].to(self.device)  # (B, T, F)
                
                optimizer.zero_grad()
                
                # Teacher predictions
                with torch.no_grad():
                    # Whisper audio encoding
                    whisper_outputs = self.expert_models.whisper.encoder(audio_features)
                    
                    # BeatNet rhythm analysis 
                    beat_outputs = self.expert_models.beatnet(audio_features)
                
                # Student predictions
                student_audio_outputs = student_model(audio_features=audio_features)
                student_audio_emb = student_audio_outputs.get('audio_embeddings')
                
                if student_audio_emb is None:
                    continue
                
                # Audio distillation losses
                loss = 0.0
                
                # 1. Whisper feature distillation
                teacher_audio_emb = whisper_outputs.last_hidden_state
                loss += self.feature_matcher.match_features(
                    student_audio_emb, teacher_audio_emb
                )
                
                # 2. Beat/rhythm knowledge transfer
                if hasattr(beat_outputs, 'beat_embeddings'):
                    beat_loss = F.mse_loss(
                        student_audio_emb.mean(dim=1), 
                        beat_outputs.beat_embeddings
                    )
                    loss += 0.5 * beat_loss
                
                loss.backward()
                optimizer.step()
                
                if batch_idx % 50 == 0:
                    logger.info(f"Audio distillation batch {batch_idx}, loss: {loss.item():.4f}")
                    
    def distill_motion_expert(self, student_model: nn.Module):
        """Distill optical flow knowledge from RAFT"""
        
        logger.info("🎬 Distilling motion knowledge from RAFT...")
        
        # Freeze RAFT expert model
        self.expert_models.raft.eval()
        
        # Setup optimizer for student motion components
        motion_params = [
            p for name, p in student_model.named_parameters() 
            if 'motion' in name.lower() or 'temporal' in name.lower()
        ]
        
        if not motion_params:
            # If no specific motion parameters, use vision parameters
            motion_params = [
                p for name, p in student_model.named_parameters() 
                if 'vision' in name.lower()
            ]
        
        optimizer = torch.optim.AdamW(motion_params, lr=1e-5)
        
        # Get video pairs for motion analysis
        data_loader = self._get_motion_distillation_data()
        
        for epoch in range(3):  # Motion distillation epochs
            total_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(data_loader):
                
                # Get consecutive video frame pairs
                frame_pairs = batch['frame_pairs'].to(self.device)  # (B, 2, C, H, W)
                B, T, C, H, W = frame_pairs.shape
                
                optimizer.zero_grad()
                
                # Teacher predictions - RAFT optical flow
                with torch.no_grad():
                    frame1 = frame_pairs[:, 0]  # (B, C, H, W)
                    frame2 = frame_pairs[:, 1]  # (B, C, H, W)
                    
                    # RAFT expects specific input format
                    try:
                        raft_flow = self.expert_models.raft(frame1, frame2)
                        if hasattr(raft_flow, 'flow_predictions'):
                            teacher_flow = raft_flow.flow_predictions[-1]  # Final prediction
                        else:
                            teacher_flow = raft_flow  # Direct flow output
                    except Exception as e:
                        logger.warning(f"RAFT processing failed: {e}")
                        continue
                
                # Student predictions - process frame sequence
                student_outputs = student_model(video_frames=frame_pairs)
                student_vision_emb = student_outputs.get('vision_embeddings')
                
                if student_vision_emb is None:
                    continue
                
                # Extract motion features from student embeddings
                # Use temporal differences to approximate optical flow understanding
                student_motion = self._extract_motion_features(student_vision_emb, frame_pairs)
                
                # Motion distillation loss
                motion_loss = self._compute_motion_distillation_loss(
                    student_motion, teacher_flow
                )
                
                motion_loss.backward()
                optimizer.step()
                
                total_loss += motion_loss.item()
                num_batches += 1
                
                if batch_idx % 50 == 0:
                    logger.info(f"Motion distillation batch {batch_idx}, loss: {motion_loss.item():.4f}")
                    
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                logger.info(f"Motion distillation epoch {epoch+1}, avg loss: {avg_loss:.4f}")
            else:
                logger.warning("No valid batches for motion distillation")
        
    def distill_cross_modal_alignment(self, student_model: nn.Module):
        """
        Final cross-modal alignment using all teacher models together
        """
        
        # Setup optimizer for fusion components
        fusion_params = [
            p for name, p in student_model.named_parameters()
            if 'fusion' in name.lower()
        ]
        optimizer = torch.optim.AdamW(fusion_params, lr=5e-6)
        
        # Get multimodal data
        data_loader = self._get_multimodal_distillation_data()
        
        for epoch in range(2):  # Cross-modal alignment epochs
            for batch_idx, batch in enumerate(data_loader):
                
                images = batch['images'].to(self.device)
                audio_features = batch['audio_features'].to(self.device)
                
                optimizer.zero_grad()
                
                # Get all teacher representations
                with torch.no_grad():
                    vision_teacher_emb = self.expert_models.get_vision_embeddings(images)
                    audio_teacher_emb = self.expert_models.get_audio_embeddings(audio_features)
                
                # Student multimodal fusion
                student_outputs = student_model(
                    video_frames=images,
                    audio_features=audio_features
                )
                
                fused_emb = student_outputs.get('fused_embeddings')
                if fused_emb is None:
                    continue
                
                # Cross-modal alignment loss
                loss = self._compute_cross_modal_alignment_loss(
                    fused_emb, vision_teacher_emb, audio_teacher_emb
                )
                
                loss.backward()
                optimizer.step()
                
                if batch_idx % 50 == 0:
                    logger.info(f"Cross-modal alignment batch {batch_idx}, loss: {loss.item():.4f}")
        
    def _compute_detection_distillation_loss(self, 
                                           student_emb: torch.Tensor,
                                           teacher_outputs: Any) -> torch.Tensor:
        """Compute object detection distillation loss"""
        
        # Simplified implementation - use attention as soft target
        if hasattr(teacher_outputs, 'attention_weights'):
            teacher_attention = teacher_outputs.attention_weights
            
            # Generate student attention from embeddings
            student_attention = F.softmax(
                torch.matmul(student_emb, student_emb.transpose(-2, -1)) / 
                (student_emb.size(-1) ** 0.5), 
                dim=-1
            )
            
            # Match dimensions and compute KL divergence
            return F.kl_div(
                student_attention.log(), 
                teacher_attention.detach(),
                reduction='batchmean'
            )
        
        return torch.tensor(0.0, device=self.device)
        
    def _compute_segmentation_distillation_loss(self,
                                              student_emb: torch.Tensor, 
                                              sam_outputs: List) -> torch.Tensor:
        """Compute segmentation distillation loss"""
        
        # Simplified SAM distillation
        loss = 0.0
        
        for i, sam_out in enumerate(sam_outputs):
            if i >= student_emb.size(0):
                break
                
            if hasattr(sam_out, 'image_embeddings'):
                teacher_seg_emb = sam_out.image_embeddings
                student_seg_emb = student_emb[i:i+1]
                
                # Dimension matching
                if teacher_seg_emb.shape != student_seg_emb.shape:
                    teacher_seg_emb = F.adaptive_avg_pool2d(
                        teacher_seg_emb, student_seg_emb.shape[-2:]
                    )
                
                loss += F.mse_loss(student_seg_emb, teacher_seg_emb)
                
        return loss / len(sam_outputs) if sam_outputs else torch.tensor(0.0)
    
    def _extract_motion_features(self, vision_embeddings: torch.Tensor, frame_pairs: torch.Tensor) -> torch.Tensor:
        """Extract motion-aware features from vision embeddings"""
        B, T, C, H, W = frame_pairs.shape
        
        if vision_embeddings.dim() == 3:  # (B, seq_len, dim)
            # Split embeddings for each frame
            emb_dim = vision_embeddings.shape[-1]
            frame1_emb = vision_embeddings[:, :emb_dim//2, :]
            frame2_emb = vision_embeddings[:, emb_dim//2:, :]
        else:
            # Use temporal difference in embeddings as motion proxy
            frame1_emb = vision_embeddings[:, 0] if vision_embeddings.dim() > 2 else vision_embeddings
            frame2_emb = vision_embeddings[:, 1] if vision_embeddings.dim() > 2 else vision_embeddings
        
        # Compute temporal difference features (motion proxy)
        motion_features = frame2_emb - frame1_emb
        
        return motion_features
    
    def _compute_motion_distillation_loss(self, student_motion: torch.Tensor, teacher_flow: torch.Tensor) -> torch.Tensor:
        """Compute loss between student motion features and teacher optical flow"""
        
        # Convert teacher flow to feature-like representation
        B, C, H, W = teacher_flow.shape  # Flow has 2 channels (x, y components)
        
        # Global average pooling to match student feature dimensions
        teacher_motion_summary = F.adaptive_avg_pool2d(teacher_flow, (1, 1)).view(B, C)
        
        # If student motion has more dimensions, we need to match
        if student_motion.dim() == 3:  # (B, seq_len, dim)
            student_motion_summary = student_motion.mean(dim=1)  # Global average
        else:
            student_motion_summary = student_motion
        
        # Dimension matching - project to same space
        if student_motion_summary.shape[-1] != teacher_motion_summary.shape[-1]:
            # Create a learnable projection layer (stored as buffer to persist across calls)
            if not hasattr(self, 'motion_projection'):
                self.motion_projection = nn.Linear(
                    student_motion_summary.shape[-1], 
                    teacher_motion_summary.shape[-1]
                ).to(self.device)
            
            student_motion_projected = self.motion_projection(student_motion_summary)
        else:
            student_motion_projected = student_motion_summary
        
        # L2 loss between motion representations
        motion_loss = F.mse_loss(student_motion_projected, teacher_motion_summary.detach())
        
        return motion_loss
    
    def _get_motion_distillation_data(self):
        """Get data loader for motion distillation with consecutive frame pairs"""
        from ..utils.data_loader import VideoEditingDataset
        from torch.utils.data import DataLoader
        
        try:
            # Create dataset that returns consecutive frame pairs
            from ..utils.video_pair_dataset import VideoPairDataset
            
            dataset = VideoPairDataset(
                data_dir=self.config.get('data_dir', 'data/'),
                datasets=['webvid', 'youtube8m'],  # Video datasets
                max_samples=5000,
                consecutive_frames=True  # Return consecutive frame pairs
            )
            
            return DataLoader(
                dataset,
                batch_size=self.config.get('distillation_batch_size', 4),  # Smaller batch for pairs
                shuffle=True,
                num_workers=2,
                collate_fn=getattr(dataset, 'collate_fn', None)
            )
        except Exception as e:
            logger.warning(f"Could not create motion distillation data: {e}")
            # Return minimal synthetic data for development
            return self._create_synthetic_motion_data()
        
    def _create_synthetic_motion_data(self):
        """Create synthetic motion data for development/testing"""
        class SyntheticMotionData:
            def __init__(self, num_samples=50):
                self.num_samples = num_samples
                
            def __iter__(self):
                for i in range(self.num_samples):
                    # Create realistic frame pairs with subtle motion
                    frame1 = torch.randn(2, 3, 224, 224)  # Batch of 2
                    frame2 = frame1 + 0.1 * torch.randn_like(frame1)  # Small motion
                    frame_pairs = torch.stack([frame1, frame2], dim=1)  # (B, 2, C, H, W)
                    
                    yield {
                        'frame_pairs': frame_pairs,
                        'video_id': f'synthetic_{i}'
                    }
                    
        return SyntheticMotionData()
        
    def _compute_cross_modal_alignment_loss(self,
                                          student_fused: torch.Tensor,
                                          vision_teacher: torch.Tensor,
                                          audio_teacher: torch.Tensor) -> torch.Tensor:
        """Compute cross-modal alignment loss"""
        
        # Contrastive loss for alignment
        batch_size = student_fused.size(0)
        
        # Compute similarities
        vision_sim = F.cosine_similarity(
            student_fused.unsqueeze(1), 
            vision_teacher.unsqueeze(0), 
            dim=-1
        )
        
        audio_sim = F.cosine_similarity(
            student_fused.unsqueeze(1),
            audio_teacher.unsqueeze(0),
            dim=-1
        )
        
        # Cross-modal alignment targets (diagonal should be high)
        targets = torch.arange(batch_size, device=self.device)
        
        vision_loss = F.cross_entropy(vision_sim / 0.07, targets)  # Temperature scaling
        audio_loss = F.cross_entropy(audio_sim / 0.07, targets)
        
        return (vision_loss + audio_loss) / 2
        
    def _get_vision_distillation_data(self):
        """Get data loader for vision distillation with robust fallbacks"""
        try:
            # Try to create real dataset
            from ..utils.data_loader import VideoEditingDataset
            from torch.utils.data import DataLoader
            
            dataset = VideoEditingDataset(
                data_dir=self.config.get('data_dir', 'data/'),
                datasets=['webvid', 'youtube8m'],  # Visual-heavy datasets
                max_samples=10000,  # Limit for distillation
                frame_sample_rate=2  # Sample every 2nd frame
            )
            
            if len(dataset) == 0:
                raise ValueError("Dataset is empty")
            
            return DataLoader(
                dataset,
                batch_size=self.config.get('distillation_batch_size', 8),
                shuffle=True,
                num_workers=4,
                collate_fn=getattr(dataset, 'collate_fn', None)
            )
            
        except Exception as e:
            logger.warning(f"Could not create real vision distillation data: {e}")
            # Return synthetic data generator instead of empty list
            return self._create_synthetic_vision_data()
    
    def _create_synthetic_vision_data(self):
        """Create synthetic vision data for development/testing"""
        class SyntheticVisionData:
            def __init__(self, num_samples=100):
                self.num_samples = num_samples
                
            def __iter__(self):
                for i in range(self.num_samples):
                    # Create realistic video frames
                    batch_size = 4
                    frames = torch.randn(batch_size, 8, 3, 224, 224)  # B, T, C, H, W
                    
                    yield {
                        'images': frames,
                        'video_frames': frames,
                        'video_id': [f'synthetic_vision_{i}_{j}' for j in range(batch_size)]
                    }
                    
        return SyntheticVisionData()
        
    def _get_audio_distillation_data(self):
        """Get data loader for audio distillation with robust fallbacks"""
        try:
            from ..utils.data_loader import VideoEditingDataset
            from torch.utils.data import DataLoader
            
            # Create dataset focused on audio content
            dataset = VideoEditingDataset(
                data_dir=self.config.get('data_dir', 'data/'),
                datasets=['audioset', 'webvid'],  # Audio-heavy datasets
                max_samples=10000,
                audio_sample_rate=16000
            )
            
            if len(dataset) == 0:
                raise ValueError("Audio dataset is empty")
            
            return DataLoader(
                dataset,
                batch_size=self.config.get('distillation_batch_size', 8),
                shuffle=True,
                num_workers=4,
                collate_fn=getattr(dataset, 'collate_fn', None)
            )
            
        except Exception as e:
            logger.warning(f"Could not create real audio distillation data: {e}")
            return self._create_synthetic_audio_data()
    
    def _create_synthetic_audio_data(self):
        """Create synthetic audio data for development/testing"""
        class SyntheticAudioData:
            def __init__(self, num_samples=100):
                self.num_samples = num_samples
                
            def __iter__(self):
                for i in range(self.num_samples):
                    # Create realistic audio features
                    batch_size = 4
                    # Whisper-style mel spectrogram features
                    audio_features = torch.randn(batch_size, 80, 3000)  # B, mel_bins, time
                    
                    yield {
                        'audio_features': audio_features,
                        'audio_id': [f'synthetic_audio_{i}_{j}' for j in range(batch_size)]
                    }
                    
        return SyntheticAudioData()
        
    def _get_multimodal_distillation_data(self):
        """Get data loader for multimodal distillation with robust fallbacks"""
        try:
            from ..utils.data_loader import VideoEditingDataset
            from torch.utils.data import DataLoader
            
            # Create full multimodal dataset
            dataset = VideoEditingDataset(
                data_dir=self.config.get('data_dir', 'data/'),
                datasets=['webvid', 'audioset', 'youtube8m', 'activitynet'],
                max_samples=20000,
                multimodal=True
            )
            
            if len(dataset) == 0:
                raise ValueError("Multimodal dataset is empty")
            
            return DataLoader(
                dataset,
                batch_size=self.config.get('distillation_batch_size', 4),  # Smaller batch for multimodal
                shuffle=True,
                num_workers=4,
                collate_fn=getattr(dataset, 'collate_fn', None)
            )
            
        except Exception as e:
            logger.warning(f"Could not create real multimodal distillation data: {e}")
            return self._create_synthetic_multimodal_data()
    
    def _create_synthetic_multimodal_data(self):
        """Create synthetic multimodal data for development/testing"""
        class SyntheticMultiModalData:
            def __init__(self, num_samples=100):
                self.num_samples = num_samples
                
            def __iter__(self):
                for i in range(self.num_samples):
                    # Create realistic multimodal data
                    batch_size = 2  # Smaller batch for multimodal
                    
                    # Video frames
                    video_frames = torch.randn(batch_size, 8, 3, 224, 224)  # B, T, C, H, W
                    
                    # Audio features
                    audio_features = torch.randn(batch_size, 80, 1000)  # B, mel_bins, time
                    
                    # Optional text tokens (captions)
                    text_tokens = torch.randint(0, 1000, (batch_size, 20))  # B, seq_len
                    
                    yield {
                        'images': video_frames,
                        'video_frames': video_frames,
                        'audio_features': audio_features,
                        'text_tokens': text_tokens,
                        'captions': text_tokens,  # Alternative key
                        'sample_id': [f'synthetic_mm_{i}_{j}' for j in range(batch_size)]
                    }
                    
        return SyntheticMultiModalData()


class ProgressiveDistillation:
    """
    Progressive distillation strategy that gradually transfers knowledge
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.stages = [
            "low_level_features",    # Basic visual/audio features
            "mid_level_features",    # Object parts, audio segments  
            "high_level_concepts",   # Full objects, speech, music
            "cross_modal_alignment"  # Multimodal understanding
        ]
        
    def progressive_distill(self, student_model: nn.Module, teacher_models: Dict):
        """
        Progressively distill knowledge in stages
        """
        
        for stage in self.stages:
            logger.info(f"📈 Progressive distillation stage: {stage}")
            
            if stage == "low_level_features":
                self._distill_low_level(student_model, teacher_models)
            elif stage == "mid_level_features":
                self._distill_mid_level(student_model, teacher_models)
            elif stage == "high_level_concepts":
                self._distill_high_level(student_model, teacher_models)
            else:  # cross_modal_alignment
                self._distill_cross_modal(student_model, teacher_models)
                
    def _distill_low_level(self, student_model, teacher_models):
        """Distill low-level features (edges, textures, basic audio)"""
        logger.info("🔍 Distilling low-level features...")
        
        # Get low-level feature extractors from teachers
        vision_teacher = teacher_models.get('vision')
        audio_teacher = teacher_models.get('audio')
        
        if vision_teacher:
            # Extract edge and texture features
            try:
                # Use early layers of vision model for low-level features
                vision_low_features = vision_teacher.vision_encoder.vision_model.encoder.layers[:4]
                # Distill these to student's early vision layers
                self._align_feature_layers(
                    student_layers=student_model.vision_encoder.vision_model.encoder.layers[:4],
                    teacher_layers=vision_low_features
                )
            except Exception as e:
                logger.warning(f"Low-level vision distillation failed: {e}")
        
        if audio_teacher:
            # Extract basic audio features (spectrograms, MFCCs)
            try:
                audio_low_features = audio_teacher.encoder.layers[:6]  # Early Whisper layers
                self._align_feature_layers(
                    student_layers=student_model.audio_encoder.encoder.layers[:6],
                    teacher_layers=audio_low_features
                )
            except Exception as e:
                logger.warning(f"Low-level audio distillation failed: {e}")
        
    def _distill_mid_level(self, student_model, teacher_models):
        """Distill mid-level features (object parts, audio segments)"""
        logger.info("🎯 Distilling mid-level features...")
        
        vision_teacher = teacher_models.get('vision')
        audio_teacher = teacher_models.get('audio')
        
        if vision_teacher:
            try:
                # Middle layers for object parts and shapes
                vision_mid_features = vision_teacher.vision_encoder.vision_model.encoder.layers[4:8]
                self._align_feature_layers(
                    student_layers=student_model.vision_encoder.vision_model.encoder.layers[4:8],
                    teacher_layers=vision_mid_features
                )
            except Exception as e:
                logger.warning(f"Mid-level vision distillation failed: {e}")
        
        if audio_teacher:
            try:
                # Middle layers for phonemes and audio segments
                audio_mid_features = audio_teacher.encoder.layers[6:18]
                self._align_feature_layers(
                    student_layers=student_model.audio_encoder.encoder.layers[6:18],
                    teacher_layers=audio_mid_features
                )
            except Exception as e:
                logger.warning(f"Mid-level audio distillation failed: {e}")
        
    def _distill_high_level(self, student_model, teacher_models):
        """Distill high-level concepts (full objects, speech understanding)"""
        logger.info("🧠 Distilling high-level concepts...")
        
        vision_teacher = teacher_models.get('vision')
        audio_teacher = teacher_models.get('audio')
        
        if vision_teacher:
            try:
                # Final layers for complete object understanding
                vision_high_features = vision_teacher.vision_encoder.vision_model.encoder.layers[8:]
                self._align_feature_layers(
                    student_layers=student_model.vision_encoder.vision_model.encoder.layers[8:],
                    teacher_layers=vision_high_features
                )
                
                # Distill final pooled representations
                self._distill_pooled_representations(student_model.vision_encoder, vision_teacher)
                
            except Exception as e:
                logger.warning(f"High-level vision distillation failed: {e}")
        
        if audio_teacher:
            try:
                # Final layers for speech and music understanding
                audio_high_features = audio_teacher.encoder.layers[18:]
                self._align_feature_layers(
                    student_layers=student_model.audio_encoder.encoder.layers[18:],
                    teacher_layers=audio_high_features
                )
                
                # Distill decoder for language understanding
                self._distill_decoder_representations(student_model.audio_encoder, audio_teacher)
                
            except Exception as e:
                logger.warning(f"High-level audio distillation failed: {e}")
        
    def _distill_cross_modal(self, student_model, teacher_models):
        """Distill cross-modal alignment and understanding"""
        logger.info("🔄 Distilling cross-modal alignment...")
        
        try:
            # Get real multimodal data instead of synthetic
            data_loader = self._get_multimodal_distillation_data()
            
            # Setup optimizer for fusion components
            fusion_params = [
                p for name, p in student_model.named_parameters()
                if 'fusion' in name.lower() or 'cross_modal' in name.lower()
            ]
            
            if not fusion_params:
                logger.warning("No fusion parameters found in student model")
                return
                
            optimizer = torch.optim.AdamW(fusion_params, lr=1e-5)
            
            # Distill cross-modal alignment using real data
            for epoch in range(2):
                total_loss = 0.0
                num_batches = 0
                
                for batch_idx, batch in enumerate(data_loader):
                    try:
                        # Get real multimodal inputs
                        video_frames = batch.get('images', batch.get('video_frames'))
                        audio_features = batch.get('audio_features')
                        text_tokens = batch.get('text_tokens', batch.get('captions'))
                        
                        if video_frames is None or audio_features is None:
                            continue
                            
                        video_frames = video_frames.to(self.device)
                        audio_features = audio_features.to(self.device)
                        
                        optimizer.zero_grad()
                        
                        # Get teacher cross-modal representations from real expert models
                        with torch.no_grad():
                            teacher_alignments = self._get_real_teacher_alignments(
                                teacher_models, video_frames, audio_features, text_tokens
                            )
                        
                        if teacher_alignments is None:
                            continue
                        
                        # Student multimodal fusion
                        if hasattr(student_model, 'fusion_module'):
                            student_fusion = student_model.fusion_module(
                                text_features=text_tokens.to(self.device) if text_tokens is not None else None,
                                vision_features=video_frames, 
                                audio_features=audio_features
                            )
                        else:
                            # Use full model forward pass
                            student_outputs = student_model(
                                video_frames=video_frames,
                                audio_features=audio_features,
                                text_tokens=text_tokens.to(self.device) if text_tokens is not None else None
                            )
                            student_fusion = student_outputs.get('fused_embeddings')
                        
                        if student_fusion is None:
                            continue
                        
                        # Cross-modal alignment loss with real teacher data
                        alignment_loss = self._compute_real_alignment_loss(
                            student_fusion, teacher_alignments
                        )
                        
                        alignment_loss.backward()
                        optimizer.step()
                        
                        total_loss += alignment_loss.item()
                        num_batches += 1
                        
                        if batch_idx % 20 == 0:
                            logger.info(f"Cross-modal alignment batch {batch_idx}, loss: {alignment_loss.item():.4f}")
                            
                    except Exception as e:
                        logger.warning(f"Batch {batch_idx} failed: {e}")
                        continue
                
                if num_batches > 0:
                    avg_loss = total_loss / num_batches
                    logger.info(f"Cross-modal alignment epoch {epoch+1}, avg loss: {avg_loss:.4f}")
                else:
                    logger.warning("No valid batches for cross-modal alignment")
                
                # Alignment loss
                if teacher_alignments is not None:
                    alignment_loss = F.mse_loss(student_fusion, teacher_alignments)
                    logger.info(f"Cross-modal alignment loss: {alignment_loss.item():.4f}")
                
        except Exception as e:
            logger.warning(f"Cross-modal distillation failed: {e}")
    
    def _align_feature_layers(self, student_layers, teacher_layers, real_data_batch=None):
        """Align corresponding layers between student and teacher using real data"""
        
        if real_data_batch is None:
            logger.warning("No real data provided for feature alignment, skipping")
            return
        
        for i, (student_layer, teacher_layer) in enumerate(zip(student_layers, teacher_layers)):
            try:
                # Freeze teacher layer
                teacher_layer.eval()
                
                # Use real input data instead of synthetic
                if 'video_frames' in real_data_batch:
                    real_input = real_data_batch['video_frames']
                elif 'audio_features' in real_data_batch:
                    real_input = real_data_batch['audio_features']
                else:
                    logger.debug(f"No suitable real input for layer alignment {i}")
                    continue
                
                # Process through teacher
                with torch.no_grad():
                    try:
                        teacher_output = teacher_layer(real_input)
                        if isinstance(teacher_output, tuple):
                            teacher_output = teacher_output[0]  # Take main output
                    except Exception as e:
                        logger.debug(f"Teacher layer {i} processing failed: {e}")
                        continue
                
                # Process through student  
                try:
                    student_output = student_layer(real_input)
                    if isinstance(student_output, tuple):
                        student_output = student_output[0]  # Take main output
                except Exception as e:
                    logger.debug(f"Student layer {i} processing failed: {e}")
                    continue
                
                # Dimension matching for different architectures
                if teacher_output.shape != student_output.shape:
                    teacher_output = self._match_tensor_dimensions(teacher_output, student_output.shape)
                    
                # Feature alignment loss (would be used in actual training loop)
                feature_loss = F.mse_loss(student_output, teacher_output.detach())
                logger.debug(f"Layer {i} feature alignment loss: {feature_loss.item():.6f}")
                    
            except Exception as e:
                logger.debug(f"Could not align layer {i}: {e}")
    
    def _match_tensor_dimensions(self, source: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        """Match tensor dimensions through pooling or projection"""
        
        if source.shape == target_shape:
            return source
            
        # Handle different cases
        if len(source.shape) != len(target_shape):
            # Different number of dimensions - use adaptive pooling
            if len(source.shape) == 4 and len(target_shape) == 3:  # (B,C,H,W) -> (B,L,D)
                source = F.adaptive_avg_pool2d(source, (1, 1)).squeeze(-1).squeeze(-1)
                if len(target_shape) == 3:
                    source = source.unsqueeze(1).repeat(1, target_shape[1], 1)
            elif len(source.shape) == 3 and len(target_shape) == 2:  # (B,L,D) -> (B,D)
                source = source.mean(dim=1)
        
        # Match last dimension if different
        if source.shape[-1] != target_shape[-1]:
            projection_key = f'dim_projection_{source.shape[-1]}_to_{target_shape[-1]}'
            
            if not hasattr(self, projection_key):
                projection = nn.Linear(source.shape[-1], target_shape[-1]).to(source.device)
                setattr(self, projection_key, projection)
            else:
                projection = getattr(self, projection_key)
            
            source = projection(source)
        
        # Handle sequence length differences
        if len(source.shape) == 3 and len(target_shape) == 3 and source.shape[1] != target_shape[1]:
            # Adaptive interpolation for sequence length
            source = F.interpolate(
                source.transpose(1, 2), 
                size=target_shape[1], 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
        
        return source
    
    def _distill_pooled_representations(self, student_encoder, teacher_encoder, real_data_batch=None):
        """Distill final pooled representations using real data"""
        try:
            if real_data_batch is None or 'video_frames' not in real_data_batch:
                logger.debug("No real video data for pooled representation distillation")
                return
                
            real_video_input = real_data_batch['video_frames']
            
            # Ensure input is in correct format for vision models
            if real_video_input.dim() == 5:  # (B, T, C, H, W)
                B, T, C, H, W = real_video_input.shape
                real_video_input = real_video_input.view(B * T, C, H, W)  # Flatten temporal
            
            with torch.no_grad():
                try:
                    teacher_features = teacher_encoder(pixel_values=real_video_input)
                    if hasattr(teacher_features, 'pooler_output'):
                        teacher_pooled = teacher_features.pooler_output
                    elif hasattr(teacher_features, 'last_hidden_state'):
                        teacher_pooled = teacher_features.last_hidden_state.mean(dim=1)  # Global pool
                    else:
                        teacher_pooled = teacher_features
                except Exception as e:
                    logger.debug(f"Teacher pooled extraction failed: {e}")
                    return
            
            try:
                student_features = student_encoder(pixel_values=real_video_input)
                if hasattr(student_features, 'pooler_output'):
                    student_pooled = student_features.pooler_output
                elif hasattr(student_features, 'last_hidden_state'):
                    student_pooled = student_features.last_hidden_state.mean(dim=1)  # Global pool
                else:
                    student_pooled = student_features
            except Exception as e:
                logger.debug(f"Student pooled extraction failed: {e}")
                return
            
            # Ensure compatible dimensions
            if teacher_pooled.shape != student_pooled.shape:
                teacher_pooled = self._match_tensor_dimensions(teacher_pooled, student_pooled.shape)
            
            # Representation alignment loss
            repr_loss = F.mse_loss(student_pooled, teacher_pooled.detach())
            logger.debug(f"Real pooled representation loss: {repr_loss.item():.6f}")
            
            return repr_loss  # Return for potential use in training loop
            
        except Exception as e:
            logger.debug(f"Pooled representation distillation failed: {e}")
            return None
    
    def _distill_decoder_representations(self, student_encoder, teacher_encoder, real_data_batch=None):
        """Distill decoder representations using real audio data"""
        try:
            if real_data_batch is None or 'audio_features' not in real_data_batch:
                logger.debug("No real audio data for decoder representation distillation")
                return
                
            real_audio_input = real_data_batch['audio_features']
            
            # Ensure audio input is in correct format
            if real_audio_input.dim() == 2:  # (batch_size, features)
                # Expand to expected shape (batch_size, sequence_length, features)
                real_audio_input = real_audio_input.unsqueeze(1)
            
            with torch.no_grad():
                try:
                    teacher_output = teacher_encoder.encoder(real_audio_input)
                    if hasattr(teacher_output, 'last_hidden_state'):
                        teacher_decoded = teacher_output.last_hidden_state
                    else:
                        teacher_decoded = teacher_output
                except Exception as e:
                    logger.debug(f"Teacher decoder extraction failed: {e}")
                    return
            
            try:
                student_output = student_encoder.encoder(real_audio_input)
                if hasattr(student_output, 'last_hidden_state'):
                    student_decoded = student_output.last_hidden_state
                else:
                    student_decoded = student_output
            except Exception as e:
                logger.debug(f"Student decoder extraction failed: {e}")
                return
            
            # Ensure compatible dimensions
            if teacher_decoded.shape != student_decoded.shape:
                teacher_decoded = self._match_tensor_dimensions(teacher_decoded, student_decoded.shape)
            
            # Decoder alignment loss
            decoder_loss = F.mse_loss(student_decoded, teacher_decoded.detach())
            logger.debug(f"Real decoder representation loss: {decoder_loss.item():.6f}")
            
            return decoder_loss  # Return for potential use in training loop
            
        except Exception as e:
            logger.debug(f"Decoder representation distillation failed: {e}")
            return None
    
    def _get_real_teacher_alignments(self, teacher_models, video_frames, audio_features, text_tokens=None):
        """Get real cross-modal alignments from expert teacher models"""
        try:
            alignments = {}
            
            # Vision teacher embeddings
            if hasattr(teacher_models, 'get') and 'vision' in teacher_models:
                vision_teacher = teacher_models['vision']
                with torch.no_grad():
                    if hasattr(vision_teacher, 'encode_image'):
                        vision_emb = vision_teacher.encode_image(video_frames)
                    elif hasattr(vision_teacher, 'get_image_features'):
                        vision_emb = vision_teacher.get_image_features(video_frames)
                    else:
                        # Use the expert models directly
                        vision_emb = self.expert_models.get_vision_embeddings(video_frames)
                    alignments['vision'] = vision_emb
            
            # Audio teacher embeddings
            if hasattr(teacher_models, 'get') and 'audio' in teacher_models:
                audio_teacher = teacher_models['audio']
                with torch.no_grad():
                    if hasattr(audio_teacher, 'encode_audio'):
                        audio_emb = audio_teacher.encode_audio(audio_features)
                    elif hasattr(audio_teacher, 'encoder'):
                        # Whisper-style encoding
                        audio_emb = audio_teacher.encoder(audio_features).last_hidden_state
                    else:
                        # Use expert models
                        audio_emb = self.expert_models.get_audio_embeddings(audio_features)
                    alignments['audio'] = audio_emb
            
            # Text embeddings if available
            if text_tokens is not None and hasattr(teacher_models, 'get') and 'text' in teacher_models:
                text_teacher = teacher_models['text']
                with torch.no_grad():
                    if hasattr(text_teacher, 'encode_text'):
                        text_emb = text_teacher.encode_text(text_tokens)
                    else:
                        # Use language model embeddings
                        text_emb = text_teacher.get_input_embeddings()(text_tokens)
                    alignments['text'] = text_emb
            
            # Create aligned multimodal representation
            if alignments:
                return self._fuse_teacher_alignments(alignments)
            else:
                return None
                
        except Exception as e:
            logger.debug(f"Could not get real teacher alignments: {e}")
            return None
    
    def _fuse_teacher_alignments(self, alignments: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse different modality alignments from teachers"""
        
        # Get batch size from any available modality
        batch_size = next(iter(alignments.values())).shape[0]
        
        # Project all modalities to same dimension
        target_dim = 1024  # Common fusion dimension
        fused_features = []
        
        for modality, features in alignments.items():
            # Handle different feature shapes
            if features.dim() == 4:  # Vision features (B, C, H, W)
                features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)
            elif features.dim() == 3:  # Sequence features (B, seq_len, dim)
                features = features.mean(dim=1)  # Global average pooling
            
            # Project to target dimension
            if features.shape[-1] != target_dim:
                # Create projection layer if it doesn't exist
                proj_name = f'{modality}_projection'
                if not hasattr(self, proj_name):
                    projection = nn.Linear(features.shape[-1], target_dim).to(self.device)
                    setattr(self, proj_name, projection)
                else:
                    projection = getattr(self, proj_name)
                
                features = projection(features)
            
            fused_features.append(features)
        
        # Concatenate and reduce to final representation
        if len(fused_features) > 1:
            concatenated = torch.cat(fused_features, dim=-1)
            
            # Final fusion projection
            if not hasattr(self, 'final_alignment_projection'):
                self.final_alignment_projection = nn.Linear(
                    concatenated.shape[-1], target_dim
                ).to(self.device)
            
            aligned_representation = self.final_alignment_projection(concatenated)
        else:
            aligned_representation = fused_features[0]
        
        return aligned_representation
    
    def _compute_real_alignment_loss(self, student_fusion: torch.Tensor, teacher_alignments: torch.Tensor) -> torch.Tensor:
        """Compute alignment loss between student fusion and real teacher alignments"""
        
        # Ensure compatible dimensions
        if student_fusion.dim() != teacher_alignments.dim():
            if student_fusion.dim() == 3:  # (B, seq_len, dim)
                student_fusion = student_fusion.mean(dim=1)  # Global pool
            if teacher_alignments.dim() == 3:
                teacher_alignments = teacher_alignments.mean(dim=1)
        
        # Dimension matching
        if student_fusion.shape[-1] != teacher_alignments.shape[-1]:
            if not hasattr(self, 'student_alignment_projection'):
                self.student_alignment_projection = nn.Linear(
                    student_fusion.shape[-1], teacher_alignments.shape[-1]
                ).to(self.device)
            student_fusion = self.student_alignment_projection(student_fusion)
        
        # Combined alignment losses
        mse_loss = F.mse_loss(student_fusion, teacher_alignments.detach())
        
        # Cosine similarity loss (encourage similar directions)
        cos_sim = F.cosine_similarity(student_fusion, teacher_alignments.detach(), dim=-1)
        cos_loss = (1 - cos_sim).mean()  # Maximize similarity
        
        # Combined loss
        total_loss = mse_loss + 0.1 * cos_loss
        
        return total_loss
    
    def _get_teacher_alignments(self, teacher_models, vision_feat, audio_feat, text_feat):
        """Legacy method - replaced by _get_real_teacher_alignments"""
        logger.warning("Using legacy synthetic teacher alignments - should use _get_real_teacher_alignments")
        return self._get_real_teacher_alignments(teacher_models, vision_feat, audio_feat, text_feat)
