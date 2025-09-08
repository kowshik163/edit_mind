"""
Multi-Modal Trainer - Handles all training phases for the Autonomous Video Editor
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any
import wandb
from omegaconf import DictConfig
from tqdm import tqdm
import logging

from ..core.hybrid_ai import HybridVideoAI
from ..distillation.distiller import KnowledgeDistiller
from ..learning.rlhf_trainer import RLHFTrainer
from ..utils.metrics import VideoEditingMetrics
from ..utils.data_loader import MultiModalDataLoader

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
        self.distiller = KnowledgeDistiller(config)
        self.rlhf_trainer = RLHFTrainer(config, model)
        self.metrics = VideoEditingMetrics()
        
        # Data loaders
        self.data_loader = MultiModalDataLoader(config)
        
        # Experiment tracking
        if config.logging.get('use_wandb', True):
            wandb.init(
                project=config.logging.wandb_project,
                name=config.logging.experiment_name,
                config=dict(config)
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
        train_loader = self.data_loader.get_pretraining_loader()
        val_loader = self.data_loader.get_validation_loader()
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.phase1.learning_rate,
            weight_decay=self.config.training.phase1.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=self.config.training.phase1.warmup_steps
        )
        
        # Training loop
        for epoch in range(self.config.training.phase1.num_epochs):
            self.current_epoch = epoch
            
            # Training
            self.model.train()
            train_loss = 0.0
            train_steps = 0
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Phase 1 Epoch {epoch+1}")):
                
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    video_frames=batch.get('video_frames'),
                    audio_features=batch.get('audio_features'),
                    text_input_ids=batch.get('text_input_ids'),
                    text_attention_mask=batch.get('text_attention_mask')
                )
                
                # Compute contrastive loss for multimodal alignment
                loss = self._compute_contrastive_loss(outputs, batch)
                
                # Backward pass
                loss.backward()
                
                if (batch_idx + 1) % self.config.training.phase1.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                train_loss += loss.item()
                train_steps += 1
                self.global_step += 1
                
                # Logging
                if self.global_step % 100 == 0:
                    wandb.log({
                        "phase1/train_loss": loss.item(),
                        "phase1/learning_rate": scheduler.get_last_lr()[0],
                        "global_step": self.global_step
                    })
                    
                # Validation
                if self.global_step % self.config.logging.eval_every == 0:
                    val_metrics = self._validate(val_loader, phase="phase1")
                    wandb.log(val_metrics)
                    
                # Checkpointing
                if self.global_step % self.config.logging.save_every == 0:
                    self._save_checkpoint(f"phase1_step_{self.global_step}")
                    
            # End of epoch logging
            avg_train_loss = train_loss / train_steps
            logger.info(f"Phase 1 Epoch {epoch+1} - Average Loss: {avg_train_loss:.4f}")
            
        logger.info("✅ Phase 1: Fusion Pretraining Completed")
        
    def phase2_distillation(self):
        """Phase 2: Distill knowledge from expert models"""
        
        # Set training phase  
        self.model.set_training_phase("distillation")
        
        # Use distillation trainer
        self.distiller.distill_all_experts(self.model)
        
        logger.info("✅ Phase 2: Knowledge Distillation Completed")
        
    def phase3_editing_finetuning(self):
        """Phase 3: Fine-tune on video editing datasets"""
        
        # Set training phase (enables LoRA)
        self.model.set_training_phase("editing_finetuning")
        
        # Get editing data
        train_loader = self.data_loader.get_editing_loader()
        val_loader = self.data_loader.get_validation_loader()
        
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
                    wandb.log({
                        "phase3/train_loss": loss.item(),
                        "global_step": self.global_step
                    })
                    
        logger.info("✅ Phase 3: Editing Fine-tuning Completed")
        
    def phase4_self_improvement(self):
        """Phase 4: Self-improvement using RLHF"""
        
        self.model.set_training_phase("self_improvement")
        
        # Use RLHF trainer
        self.rlhf_trainer.train_with_human_feedback(self.model)
        
        logger.info("✅ Phase 4: Self-Improvement (RLHF) Completed")
        
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
                    
                    wandb.log({
                        "phase5/quality_score": quality_score,
                        "phase5/video_path": video_path
                    })
                    
                except Exception as e:
                    logger.error(f"❌ Failed to edit {video_path}: {str(e)}")
                    
        logger.info("✅ Phase 5: Autonomous Integration Completed")
        
    def _compute_contrastive_loss(self, outputs: Dict, batch: Dict) -> torch.Tensor:
        """Compute contrastive loss for multimodal alignment"""
        
        # Get fused embeddings
        fused_emb = outputs.get('fused_embeddings')
        if fused_emb is None:
            return torch.tensor(0.0, device=self.device)
            
        # Simple contrastive loss implementation
        # In practice, you'd use more sophisticated losses like InfoNCE
        batch_size = fused_emb.size(0)
        
        # Create positive and negative pairs
        similarity_matrix = torch.matmul(fused_emb, fused_emb.transpose(-2, -1))
        
        # Mask for positive pairs (same batch index)
        labels = torch.arange(batch_size, device=self.device)
        
        # Cross-entropy loss on similarity matrix
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss
        
    def _compute_editing_loss(self, outputs: Dict, batch: Dict) -> torch.Tensor:
        """Compute loss for editing timeline prediction"""
        
        timeline = outputs.get('timeline')
        if timeline is None:
            return torch.tensor(0.0, device=self.device)
            
        # Ground truth editing timeline
        target_timeline = batch.get('target_timeline')
        if target_timeline is None:
            # If no ground truth, use a simple reconstruction loss
            return F.mse_loss(timeline.get('cut_points', torch.zeros(1)), 
                             torch.zeros_like(timeline.get('cut_points', torch.zeros(1))))
        
        # Timeline-specific loss computation
        loss = 0.0
        
        if 'cut_points' in timeline and 'cut_points' in target_timeline:
            loss += F.mse_loss(timeline['cut_points'], target_timeline['cut_points'])
            
        if 'transitions' in timeline and 'transitions' in target_timeline:
            loss += F.cross_entropy(timeline['transitions'], target_timeline['transitions'])
            
        return loss
        
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
