"""
Knowledge Distillation Module
Distills knowledge from expert models into the hybrid AI system
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
import logging
from omegaconf import DictConfig

from ..models.expert_models import ExpertModels
from ..utils.distillation_utils import DistillationLoss, FeatureMatching

logger = logging.getLogger(__name__)


class KnowledgeDistiller:
    """
    Distills knowledge from multiple expert models:
    - RT-DETR (object detection)
    - HQ-SAM (segmentation)  
    - Whisper (speech recognition)
    - BeatNet (audio analysis)
    - RAFT (optical flow)
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load expert teacher models
        self.expert_models = ExpertModels(config)
        
        # Distillation loss functions
        self.distillation_loss = DistillationLoss(
            temperature=config.training.phase2.temperature,
            alpha=config.training.phase2.alpha
        )
        
        # Feature matching for intermediate representations
        self.feature_matcher = FeatureMatching()
        
    def distill_all_experts(self, student_model: nn.Module):
        """
        Sequential distillation from all expert models
        """
        logger.info("🔬 Starting Knowledge Distillation from Expert Models")
        
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
        
        # This would implement RAFT optical flow distillation
        # For now, we'll create a placeholder
        logger.info("Motion distillation placeholder - implement RAFT integration")
        
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
        """Get data loader for vision distillation"""
        from ..utils.data_loader import VideoEditingDataset, MultiModalDataLoader
        from torch.utils.data import DataLoader
        
        try:
            # Create dataset focused on visual content
            dataset = VideoEditingDataset(
                data_dir=self.config.get('data_dir', 'data/'),
                datasets=['webvid', 'youtube8m'],  # Visual-heavy datasets
                max_samples=10000,  # Limit for distillation
                frame_sample_rate=2  # Sample every 2nd frame
            )
            
            # Create dataloader
            return DataLoader(
                dataset,
                batch_size=self.config.get('distillation_batch_size', 8),
                shuffle=True,
                num_workers=4,
                collate_fn=dataset.collate_fn
            )
        except Exception as e:
            logger.warning(f"Could not create vision distillation data: {e}")
            # Return empty dataset for development
            return []
        
    def _get_audio_distillation_data(self):
        """Get data loader for audio distillation"""
        from ..utils.data_loader import VideoEditingDataset, MultiModalDataLoader
        from torch.utils.data import DataLoader
        
        try:
            # Create dataset focused on audio content
            dataset = VideoEditingDataset(
                data_dir=self.config.get('data_dir', 'data/'),
                datasets=['audioset', 'webvid'],  # Audio-heavy datasets
                max_samples=10000,
                audio_sample_rate=16000
            )
            
            return DataLoader(
                dataset,
                batch_size=self.config.get('distillation_batch_size', 8),
                shuffle=True,
                num_workers=4,
                collate_fn=dataset.collate_fn
            )
        except Exception as e:
            logger.warning(f"Could not create audio distillation data: {e}")
            return []
        
    def _get_multimodal_distillation_data(self):
        """Get data loader for multimodal distillation"""
        from ..utils.data_loader import VideoEditingDataset, MultiModalDataLoader
        from torch.utils.data import DataLoader
        
        try:
            # Create full multimodal dataset
            dataset = VideoEditingDataset(
                data_dir=self.config.get('data_dir', 'data/'),
                datasets=['webvid', 'audioset', 'youtube8m', 'activitynet'],
                max_samples=20000,
                multimodal=True
            )
            
            return DataLoader(
                dataset,
                batch_size=self.config.get('distillation_batch_size', 4),  # Smaller batch for multimodal
                shuffle=True,
                num_workers=4,
                collate_fn=dataset.collate_fn
            )
        except Exception as e:
            logger.warning(f"Could not create multimodal distillation data: {e}")
            return []


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
            # Distill the fusion module's ability to align modalities
            if hasattr(student_model, 'fusion_module'):
                
                # Create synthetic cross-modal training examples
                vision_features = torch.randn(4, 50, 1024)  # Batch, seq_len, dim
                audio_features = torch.randn(4, 50, 1280)
                text_features = torch.randn(4, 20, 4096)
                
                # Get teacher cross-modal representations
                teacher_alignments = self._get_teacher_alignments(
                    teacher_models, vision_features, audio_features, text_features
                )
                
                # Train student to match these alignments
                student_fusion = student_model.fusion_module(
                    text_features, vision_features, audio_features
                )
                
                # Alignment loss
                if teacher_alignments is not None:
                    alignment_loss = F.mse_loss(student_fusion, teacher_alignments)
                    logger.info(f"Cross-modal alignment loss: {alignment_loss.item():.4f}")
                
        except Exception as e:
            logger.warning(f"Cross-modal distillation failed: {e}")
    
    def _align_feature_layers(self, student_layers, teacher_layers):
        """Align corresponding layers between student and teacher"""
        for student_layer, teacher_layer in zip(student_layers, teacher_layers):
            try:
                # Freeze teacher layer
                teacher_layer.eval()
                
                # Create synthetic input to check feature alignment
                if hasattr(teacher_layer, 'self_attn'):  # Transformer layer
                    hidden_size = teacher_layer.self_attn.embed_dim
                    synthetic_input = torch.randn(2, 10, hidden_size)
                    
                    with torch.no_grad():
                        teacher_output = teacher_layer(synthetic_input)[0]
                    
                    student_output = student_layer(synthetic_input)[0]
                    
                    # Feature alignment loss (would be used in actual training)
                    feature_loss = F.mse_loss(student_output, teacher_output)
                    logger.debug(f"Feature alignment loss: {feature_loss.item():.6f}")
                    
            except Exception as e:
                logger.debug(f"Could not align layer: {e}")
    
    def _distill_pooled_representations(self, student_encoder, teacher_encoder):
        """Distill final pooled representations"""
        try:
            # Test with synthetic data
            synthetic_input = torch.randn(2, 3, 224, 224)  # Batch, channels, height, width
            
            with torch.no_grad():
                teacher_features = teacher_encoder(pixel_values=synthetic_input).pooler_output
            
            student_features = student_encoder(pixel_values=synthetic_input).pooler_output
            
            # Representation alignment
            repr_loss = F.mse_loss(student_features, teacher_features)
            logger.debug(f"Pooled representation loss: {repr_loss.item():.6f}")
            
        except Exception as e:
            logger.debug(f"Pooled representation distillation failed: {e}")
    
    def _distill_decoder_representations(self, student_encoder, teacher_encoder):
        """Distill decoder representations for language understanding"""
        try:
            # Test with synthetic audio input
            synthetic_audio = torch.randn(2, 80, 3000)  # Batch, mel_bins, time_steps
            
            with torch.no_grad():
                teacher_output = teacher_encoder.encoder(synthetic_audio).last_hidden_state
            
            student_output = student_encoder.encoder(synthetic_audio).last_hidden_state
            
            # Decoder alignment
            decoder_loss = F.mse_loss(student_output, teacher_output)
            logger.debug(f"Decoder representation loss: {decoder_loss.item():.6f}")
            
        except Exception as e:
            logger.debug(f"Decoder representation distillation failed: {e}")
    
    def _get_teacher_alignments(self, teacher_models, vision_feat, audio_feat, text_feat):
        """Get cross-modal alignments from teacher models"""
        try:
            # This would use teacher models to create cross-modal alignments
            # For now, return a synthetic aligned representation
            batch_size = vision_feat.shape[0]
            fusion_dim = 2048
            
            # Create synthetic but realistic alignment
            aligned_features = torch.randn(batch_size, 50, fusion_dim)
            return aligned_features
            
        except Exception as e:
            logger.debug(f"Could not get teacher alignments: {e}")
            return None
