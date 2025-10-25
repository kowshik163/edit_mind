"""
Multi-Modal Trainer - Handles all training phases for the Autonomous Video Editor
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any
import wandb
from omegaconf import DictConfig
from tqdm import tqdm
import logging

from core.hybrid_ai import HybridVideoAI
# Temporarily commented out due to syntax errors - will be fixed
# from distillation.distiller import KnowledgeDistiller
from learning.enhanced_rlhf_trainer import EnhancedRLHFTrainer as RLHFTrainer
from utils.metrics import VideoEditingMetrics
from utils.data_loader import MultiModalDataLoader

logger = logging.getLogger(__name__)


class MultiModalTrainer:
    """
    Handles all 5 training phases:
    1. Fusion Pretraining
    2. Knowledge Distillation  
    3. Editing Fine-tuning
    4. Self-Improvement (RLHF)
    5. Autonomous Hybrid AI
    """
    
    def __init__(self, config: DictConfig, model: HybridVideoAI):
        self.config = config
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        
        # Initialize specialized trainers
        # Temporarily disabled due to syntax errors
        # self.distiller = KnowledgeDistiller(config)
        self.distiller = None
        
        # Only initialize RLHF trainer if enabled and on GPU (bf16 requirement)
        rlhf_enabled = config.get('rlhf', {}).get('enabled', False) if hasattr(config, 'get') else False
        if rlhf_enabled and torch.cuda.is_available():
            try:
                self.rlhf_trainer = RLHFTrainer(config, model)
            except Exception as e:
                logger.warning(f"Failed to initialize RLHF trainer: {e}")
                self.rlhf_trainer = None
        else:
            self.rlhf_trainer = None
            if rlhf_enabled:
                logger.warning("RLHF training requires GPU - skipping RLHF trainer initialization")
        
        self.metrics = VideoEditingMetrics()
        
        # Data loaders
        self.data_loader = MultiModalDataLoader(config)
        
        # Experiment tracking
        logging_cfg = config.logging if hasattr(config, 'logging') and isinstance(config.logging, (dict, DictConfig)) else {}
        if getattr(logging_cfg, 'get', None):
            use_wandb = logging_cfg.get('use_wandb', True)
            wandb_project = logging_cfg.get('wandb_project', 'auto-editor')
            experiment_name = logging_cfg.get('experiment_name', 'default')
        else:
            use_wandb = True
            wandb_project = 'auto-editor'
            experiment_name = 'default'
        
        # Store wandb flag as instance variable
        self.use_wandb = use_wandb
        
        if use_wandb:
            wandb.init(
                project=wandb_project,
                name=experiment_name,
                config=dict(config),
                reinit=True
            )
        
        self.global_step = 0
        self.current_epoch = 0
        
    def train_all_phases(self):
        """Execute all training phases in sequence"""
        logger.info("🚀 Starting Autonomous Video Editor Training Pipeline")
        
        # Phase 1: Fusion Pretraining
        logger.info("📚 Phase 1: Fusion Pretraining")
        self.phase1_fusion_pretraining()
        
        # Phase 2: Knowledge Distillation
        logger.info("🔬 Phase 2: Knowledge Distillation") 
        self.phase2_distillation()
        
        # Phase 3: Editing Fine-tuning
        logger.info("✂️ Phase 3: Editing Fine-tuning")
        self.phase3_editing_finetuning()
        
        # Phase 4: Self-Improvement (RLHF)
        logger.info("🧠 Phase 4: Self-Improvement (RLHF)")
        self.phase4_self_improvement()
        
        # Phase 5: Final Autonomous Integration
        logger.info("🌟 Phase 5: Autonomous Hybrid AI")
        self.phase5_autonomous_integration()
        
        logger.info("✅ Training pipeline completed!")
        
    def phase1_fusion_pretraining(self):
        """Phase 1: Train multimodal embedding space"""
        
        # Set training phase
        self.model.set_training_phase("pretraining")
        
        # Get pretraining data
        train_loader = self.data_loader.get_train_loader('data')
        val_loader = self.data_loader.get_val_loader('data')
        
        # Setup optimizer with proper type conversion
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config.training.phase1.learning_rate),
            weight_decay=float(self.config.training.phase1.weight_decay)
        )
        
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=int(self.config.training.phase1.warmup_steps)
        )
        
        # Training loop
        for epoch in range(int(self.config.training.phase1.num_epochs)):
            self.current_epoch = epoch
            
            # Training
            self.model.train()
            train_loss = 0.0
            train_steps = 0
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Phase 1 Epoch {epoch+1}")):
                
                try:
                    # Move batch to device
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    # Log batch shapes
                    logger.info(f"Batch {batch_idx} shapes:")
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            logger.info(f"  {k}: {v.shape}")
                    
                    # Forward pass
                    logger.info("Starting forward pass...")
                    outputs = self.model(
                        video_frames=batch.get('video_frames'),
                        audio_features=batch.get('audio_features'),
                        text_input_ids=batch.get('text_input_ids'),
                        text_attention_mask=batch.get('text_attention_mask')
                    )
                    logger.info("Forward pass completed")
                    
                    # Compute contrastive loss for multimodal alignment
                    logger.info("Computing contrastive loss...")
                    loss = self._compute_contrastive_loss(outputs, batch)
                    logger.info(f"Loss computed: {loss.item()}")
                    
                except Exception as e:
                    logger.error(f"ERROR in batch {batch_idx}:")
                    logger.error(f"Error: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    raise
                
                # Backward pass
                loss.backward()
                
                if (batch_idx + 1) % int(self.config.training.phase1.gradient_accumulation_steps) == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                train_loss += loss.item()
                train_steps += 1
                self.global_step += 1
                
                # Logging
                if self.global_step % 100 == 0:
                    if self.use_wandb:
                        wandb.log({
                            "phase1/train_loss": loss.item(),
                            "phase1/learning_rate": scheduler.get_last_lr()[0],
                            "global_step": self.global_step
                        })
                    
                # Validation
                if self.global_step % self.config.logging.eval_every == 0:
                    val_metrics = self._validate(val_loader, phase="phase1")
                    if self.use_wandb:
                        wandb.log(val_metrics)
                    
                # Checkpointing
                if self.global_step % self.config.logging.save_every == 0:
                    self._save_checkpoint(f"phase1_step_{self.global_step}")
                    
            # End of epoch logging
            avg_train_loss = train_loss / train_steps
            logger.info(f"Phase 1 Epoch {epoch+1} - Average Loss: {avg_train_loss:.4f}")
            
        logger.info("✅ Phase 1: Fusion Pretraining Completed")
        return {"status": "completed", "final_loss": avg_train_loss}
        
    def phase2_distillation(self):
        """Phase 2: Distill knowledge from expert models"""
        
        if self.distiller is None:
            logger.warning("Knowledge distiller not initialized - skipping phase 2")
            return {"status": "skipped", "reason": "Distiller not available"}
        
        # Set training phase  
        self.model.set_training_phase("distillation")
        
        # Use distillation trainer
        self.distiller.distill_all_experts(self.model)
        
        logger.info("✅ Phase 2: Knowledge Distillation Completed")
        return {"status": "completed"}
        
    def phase3_editing_finetuning(self):
        """Phase 3: Fine-tune on video editing datasets"""
        
        # Set training phase (enables LoRA)
        self.model.set_training_phase("editing_finetuning")
        
        # Get editing data
        train_loader = self.data_loader.get_train_loader('data')
        val_loader = self.data_loader.get_val_loader('data')
        
        # Setup optimizer for LoRA parameters only
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.training.phase3.learning_rate
        )
        
        # Training loop
        for epoch in range(self.config.training.phase3.num_epochs):
            self.model.train()
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Phase 3 Epoch {epoch+1}")):
                
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass with editing timeline generation
                outputs = self.model(
                    video_frames=batch['video_frames'],
                    audio_features=batch['audio_features'], 
                    text_input_ids=batch['text_input_ids'],
                    text_attention_mask=batch['text_attention_mask'],
                    editing_prompt=batch.get('editing_prompt'),
                    return_timeline=True
                )
                
                # Compute editing-specific loss
                loss = self._compute_editing_loss(outputs, batch)
                
                loss.backward()
                
                if (batch_idx + 1) % 4 == 0:  # Gradient accumulation
                    optimizer.step()
                    optimizer.zero_grad()
                    
                self.global_step += 1
                
                if self.global_step % 50 == 0:
                    if self.use_wandb:
                        wandb.log({
                            "phase3/train_loss": loss.item(),
                            "global_step": self.global_step
                        })
                    
        logger.info("✅ Phase 3: Editing Fine-tuning Completed")
        return {"status": "completed"}
        
    def phase4_self_improvement(self):
        """Phase 4: Self-improvement using RLHF"""
        
        if self.rlhf_trainer is None:
            logger.warning("RLHF trainer not initialized - skipping phase 4")
            return {"status": "skipped", "reason": "RLHF trainer not available"}
        
        self.model.set_training_phase("self_improvement")
        
        # Use RLHF trainer
        self.rlhf_trainer.train_with_human_feedback(self.model)
        
        logger.info("✅ Phase 4: Self-Improvement (RLHF) Completed")
        return {"status": "completed"}
        
    def phase5_autonomous_integration(self):
        """Phase 5: Final integration and testing"""
        
        # Test autonomous editing capabilities
        test_videos = self.data_loader.get_test_videos()
        
        self.model.eval()
        with torch.no_grad():
            for video_path, prompt in test_videos:
                try:
                    output_path = self.model.autonomous_edit(video_path, prompt)
                    
                    # Evaluate output quality
                    quality_score = self.metrics.evaluate_video_quality(output_path)
                    
                    logger.info(f"✅ Successfully edited {video_path} -> {output_path}")
                    logger.info(f"Quality Score: {quality_score:.3f}")
                    
                    if self.use_wandb:
                        wandb.log({
                            "phase5/quality_score": quality_score,
                            "phase5/video_path": video_path
                        })
                    
                except Exception as e:
                    logger.error(f"❌ Failed to edit {video_path}: {str(e)}")
                    
        logger.info("✅ Phase 5: Autonomous Integration Completed")
        return {"status": "completed"}
        
    def _compute_contrastive_loss(self, outputs: Dict, batch: Dict) -> torch.Tensor:
        """Compute contrastive loss for multimodal alignment"""
        
        # Get fused embeddings
        fused_emb = outputs.get('fused_embeddings')
        if fused_emb is None:
            return torch.tensor(0.0, device=self.device)
            
        # Pool fused embeddings to get single vector per sample
        # fused_emb shape: (B, seq_len, fusion_dim) -> (B, fusion_dim)
        pooled_emb = fused_emb.mean(dim=1)  # Average pooling over sequence dimension
        
        # Normalize embeddings for contrastive learning
        pooled_emb = F.normalize(pooled_emb, p=2, dim=1)
        
        # Simple contrastive loss implementation
        # Compute similarity matrix: (B, B)
        batch_size = pooled_emb.size(0)
        temperature = 0.07  # Temperature for contrastive loss
        
        # Compute cosine similarity matrix
        similarity_matrix = torch.matmul(pooled_emb, pooled_emb.t()) / temperature
        
        # Mask for positive pairs (same batch index)
        labels = torch.arange(batch_size, device=self.device)
        
        # Cross-entropy loss on similarity matrix
        # Now similarity_matrix is (B, B) and labels is (B,) - correct shapes!
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss
        
    def _compute_editing_loss(self, outputs: Dict, batch: Dict) -> torch.Tensor:
        """Compute loss for editing timeline prediction with both type and parameter supervision
        
        This enhanced loss function handles:
        1. Cut point prediction (binary classification)
        2. Effect type classification (multi-class)
        3. Effect parameter regression (continuous values)
        4. Transition type classification (multi-class)
        5. Transition parameter regression (continuous values)
        
        Args:
            outputs: Model predictions containing timeline information
            batch: Ground truth batch data with target_timeline
            
        Returns:
            Combined weighted loss tensor
        """
        
        timeline = outputs.get('timeline')
        if timeline is None:
            return torch.tensor(0.0, device=self.device)
            
        # Ground truth editing timeline
        target_timeline = batch.get('target_timeline')
        if target_timeline is None:
            # If no ground truth, use a simple reconstruction loss
            return F.mse_loss(timeline.get('cut_points', torch.zeros(1, device=self.device)), 
                             torch.zeros_like(timeline.get('cut_points', torch.zeros(1, device=self.device))))
        
        # Handle both single dict and list of dicts (batch)
        if isinstance(target_timeline, list):
            # Stack batch items
            target_cut_points = torch.stack([t['cut_points'] for t in target_timeline]).to(self.device)
            target_effect_types = torch.stack([t['effect_types'] for t in target_timeline]).to(self.device)
            target_effect_params = torch.stack([t['effect_params'] for t in target_timeline]).to(self.device)
            target_transition_types = torch.stack([t['transition_types'] for t in target_timeline]).to(self.device)
            target_transition_params = torch.stack([t['transition_params'] for t in target_timeline]).to(self.device)
        else:
            # Single sample - ensure proper shape
            target_cut_points = target_timeline.get('cut_points', torch.zeros(1, device=self.device)).to(self.device)
            target_effect_types = target_timeline.get('effect_types', torch.zeros(1, device=self.device)).to(self.device)
            target_effect_params = target_timeline.get('effect_params', torch.zeros(1, 8, device=self.device)).to(self.device)
            target_transition_types = target_timeline.get('transition_types', torch.zeros(1, device=self.device)).to(self.device)
            target_transition_params = target_timeline.get('transition_params', torch.zeros(1, 8, device=self.device)).to(self.device)
        
        # Initialize total loss
        total_loss = 0.0
        loss_components = {}
        
        # Loss weights (configurable via config in future)
        weights = {
            'cut_points': 1.0,
            'effect_types': 2.0,
            'effect_params': 1.5,
            'transition_types': 2.0,
            'transition_params': 1.5
        }
        
        # 1. Cut points loss (binary classification per frame)
        if 'cut_points' in timeline and target_cut_points is not None:
            pred_cuts = timeline['cut_points']
            
            # Ensure shapes match
            if pred_cuts.shape != target_cut_points.shape:
                # Reshape if needed
                if len(pred_cuts.shape) == 1:
                    pred_cuts = pred_cuts.unsqueeze(0)
                if len(target_cut_points.shape) == 1:
                    target_cut_points = target_cut_points.unsqueeze(0)
            
            # Binary cross entropy for cut detection
            cut_loss = F.binary_cross_entropy_with_logits(
                pred_cuts.float(), 
                target_cut_points.float()
            )
            loss_components['cut_points'] = cut_loss.item()
            total_loss += weights['cut_points'] * cut_loss
        
        # 2. Effect type classification loss
        if 'effect_types' in timeline and target_effect_types is not None:
            pred_effect_types = timeline['effect_types']
            
            # Ensure proper shape [batch_size, max_effects, num_classes]
            if len(pred_effect_types.shape) == 2:
                # [batch_size, max_effects] - need to expand for CrossEntropy
                # Assume model outputs logits over effect type classes
                pass
            
            # Flatten for cross entropy: [batch_size * max_effects, num_classes] and [batch_size * max_effects]
            batch_size = pred_effect_types.shape[0] if len(pred_effect_types.shape) > 1 else 1
            max_effects = target_effect_types.shape[-1] if len(target_effect_types.shape) > 1 else target_effect_types.shape[0]
            
            # Only compute loss on non-zero (non-padding) effect types
            mask = target_effect_types.view(-1) > 0
            
            if mask.sum() > 0 and len(pred_effect_types.shape) > 2:
                pred_flat = pred_effect_types.view(-1, pred_effect_types.shape[-1])
                target_flat = target_effect_types.view(-1).long()
                
                # Apply mask and compute loss
                effect_type_loss = F.cross_entropy(
                    pred_flat[mask],
                    target_flat[mask],
                    reduction='mean'
                )
                loss_components['effect_types'] = effect_type_loss.item()
                total_loss += weights['effect_types'] * effect_type_loss
        
        # 3. Effect parameter regression loss (MSE on continuous parameters)
        if 'effect_params' in timeline and target_effect_params is not None:
            pred_effect_params = timeline['effect_params']
            
            # Only compute loss where effect types are non-zero
            mask = target_effect_types.view(-1) > 0
            
            if mask.sum() > 0:
                # Reshape for loss computation
                pred_params_flat = pred_effect_params.view(-1, pred_effect_params.shape[-1])
                target_params_flat = target_effect_params.view(-1, target_effect_params.shape[-1])
                
                # MSE loss on parameters
                effect_param_loss = F.mse_loss(
                    pred_params_flat[mask],
                    target_params_flat[mask],
                    reduction='mean'
                )
                loss_components['effect_params'] = effect_param_loss.item()
                total_loss += weights['effect_params'] * effect_param_loss
        
        # 4. Transition type classification loss
        if 'transition_types' in timeline and target_transition_types is not None:
            pred_transition_types = timeline['transition_types']
            
            # Only compute loss on non-zero (non-padding) transition types
            mask = target_transition_types.view(-1) > 0
            
            if mask.sum() > 0 and len(pred_transition_types.shape) > 2:
                pred_flat = pred_transition_types.view(-1, pred_transition_types.shape[-1])
                target_flat = target_transition_types.view(-1).long()
                
                transition_type_loss = F.cross_entropy(
                    pred_flat[mask],
                    target_flat[mask],
                    reduction='mean'
                )
                loss_components['transition_types'] = transition_type_loss.item()
                total_loss += weights['transition_types'] * transition_type_loss
        
        # 5. Transition parameter regression loss
        if 'transition_params' in timeline and target_transition_params is not None:
            pred_transition_params = timeline['transition_params']
            
            # Only compute loss where transition types are non-zero
            mask = target_transition_types.view(-1) > 0
            
            if mask.sum() > 0:
                pred_params_flat = pred_transition_params.view(-1, pred_transition_params.shape[-1])
                target_params_flat = target_transition_params.view(-1, target_transition_params.shape[-1])
                
                transition_param_loss = F.mse_loss(
                    pred_params_flat[mask],
                    target_params_flat[mask],
                    reduction='mean'
                )
                loss_components['transition_params'] = transition_param_loss.item()
                total_loss += weights['transition_params'] * transition_param_loss
        
        # Log individual loss components (if wandb is enabled)
        if self.use_wandb and len(loss_components) > 0:
            wandb.log({
                **{f'loss/{k}': v for k, v in loss_components.items()},
                'loss/editing_total': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
            }, step=self.global_step)
        
        return total_loss if isinstance(total_loss, torch.Tensor) else torch.tensor(total_loss, device=self.device)
        
    def _validate(self, val_loader: DataLoader, phase: str) -> Dict[str, float]:
        """Run validation"""
        
        self.model.eval()
        val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(
                    video_frames=batch.get('video_frames'),
                    audio_features=batch.get('audio_features'),
                    text_input_ids=batch.get('text_input_ids'),
                    text_attention_mask=batch.get('text_attention_mask')
                )
                
                if phase == "phase1":
                    loss = self._compute_contrastive_loss(outputs, batch)
                else:
                    loss = self._compute_editing_loss(outputs, batch)
                    
                val_loss += loss.item()
                val_steps += 1
                
        self.model.train()
        
        avg_val_loss = val_loss / val_steps if val_steps > 0 else 0.0
        
        return {f"{phase}/val_loss": avg_val_loss}
        
    def _save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint"""
        
        checkpoint_path = os.path.join(
            self.config.logging.checkpoint_dir,
            f"{checkpoint_name}.pt"
        )
        
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        self.model.save_checkpoint(
            checkpoint_path,
            epoch=self.current_epoch,
        )
        
        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")


def main():
    """Main training entry point"""
    import hydra
    from omegaconf import DictConfig
    
    @hydra.main(config_path="../../configs", config_name="main_config")
    def train(config: DictConfig):
        
        # Initialize model
        model = HybridVideoAI(config)
        
        # Initialize trainer
        trainer = MultiModalTrainer(config, model)
        
        # Start training
        trainer.train_all_phases()
        
    train()


if __name__ == "__main__":
    main()
