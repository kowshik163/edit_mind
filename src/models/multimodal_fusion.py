"""
Multimodal Fusion Module - Fuses text, vision, and audio embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class MultiModalFusionModule(nn.Module):
    """
    Fusion module that combines text, vision, and audio embeddings
    using cross-attention and learned fusion weights
    """
    
    def __init__(self, 
                 text_dim: int = 4096,
                 vision_dim: int = 1024, 
                 audio_dim: int = 1280,
                 fusion_dim: int = 2048,
                 num_heads: int = 16,
                 dropout: float = 0.1):
        super().__init__()
        
        self.text_dim = text_dim
        self.vision_dim = vision_dim
        self.audio_dim = audio_dim
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        
        # Projection layers to common dimension
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.vision_proj = nn.Linear(vision_dim, fusion_dim)
        self.audio_proj = nn.Linear(audio_dim, fusion_dim)
        
        # Cross-attention layers
        self.text_vision_attention = nn.MultiheadAttention(
            fusion_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.text_audio_attention = nn.MultiheadAttention(
            fusion_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.vision_audio_attention = nn.MultiheadAttention(
            fusion_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Self-attention for final fusion
        self.fusion_attention = nn.MultiheadAttention(
            fusion_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Fusion gates - learned weights for combining modalities
        self.text_gate = nn.Parameter(torch.ones(1))
        self.vision_gate = nn.Parameter(torch.ones(1))
        self.audio_gate = nn.Parameter(torch.ones(1))
        
        # Layer normalization
        self.text_norm = nn.LayerNorm(fusion_dim)
        self.vision_norm = nn.LayerNorm(fusion_dim)
        self.audio_norm = nn.LayerNorm(fusion_dim)
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        
        # Store dropout for later use
        self.dropout = nn.Dropout(dropout)
        
        # Feed-forward networks
        self.text_ffn = self._make_ffn(fusion_dim, dropout)
        self.vision_ffn = self._make_ffn(fusion_dim, dropout)
        self.audio_ffn = self._make_ffn(fusion_dim, dropout)
        self.fusion_ffn = self._make_ffn(fusion_dim, dropout)
        
    def _make_ffn(self, dim: int, dropout: float) -> nn.Module:
        """Create feed-forward network"""
        return nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self,
                text_emb: Optional[torch.Tensor] = None,
                vision_emb: Optional[torch.Tensor] = None,
                audio_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Fuse multimodal embeddings
        
        Args:
            text_emb: (B, L_text, text_dim) 
            vision_emb: (B, L_vision, vision_dim)
            audio_emb: (B, L_audio, audio_dim)
            
        Returns:
            fused_emb: (B, L, fusion_dim) fused embeddings
        """
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info("="*60)
        logger.info("FUSION MODULE - Starting")
        
        modalities = []
        
        # Project each modality to common dimension and align sequence lengths
        target_seq_len = None
        
        # First pass: determine target sequence length (use the shortest non-None modality)
        seq_lengths = []
        if text_emb is not None:
            seq_lengths.append(text_emb.shape[1])
            logger.info(f"  text_emb input shape: {text_emb.shape}")
        if vision_emb is not None:
            seq_lengths.append(vision_emb.shape[1])
            logger.info(f"  vision_emb input shape: {vision_emb.shape}")
        if audio_emb is not None:
            seq_lengths.append(audio_emb.shape[1])
            logger.info(f"  audio_emb input shape: {audio_emb.shape}")
        
        if seq_lengths:
            target_seq_len = min(seq_lengths)  # Use minimum to avoid padding
            logger.info(f"  target_seq_len: {target_seq_len}")
        else:
            raise ValueError("At least one modality must be provided")
        
        # Project and align each modality
        if text_emb is not None:
            logger.info(f"  Projecting text_emb...")
            text_proj = self.text_proj(text_emb)
            text_proj = self.text_norm(text_proj)
            logger.info(f"  text_proj shape after projection: {text_proj.shape}")
            # Align sequence length
            if text_proj.shape[1] > target_seq_len:
                text_proj = text_proj[:, :target_seq_len, :]
            elif text_proj.shape[1] < target_seq_len:
                # Pad with zeros
                padding = torch.zeros(text_proj.shape[0], target_seq_len - text_proj.shape[1], 
                                     text_proj.shape[2], device=text_proj.device, dtype=text_proj.dtype)
                text_proj = torch.cat([text_proj, padding], dim=1)
            logger.info(f"  text_proj shape after alignment: {text_proj.shape}")
            modalities.append(('text', text_proj))
            
        if vision_emb is not None:
            vision_proj = self.vision_proj(vision_emb)
            vision_proj = self.vision_norm(vision_proj)
            # Align sequence length
            if vision_proj.shape[1] > target_seq_len:
                vision_proj = vision_proj[:, :target_seq_len, :]
            elif vision_proj.shape[1] < target_seq_len:
                padding = torch.zeros(vision_proj.shape[0], target_seq_len - vision_proj.shape[1], 
                                     vision_proj.shape[2], device=vision_proj.device, dtype=vision_proj.dtype)
                vision_proj = torch.cat([vision_proj, padding], dim=1)
            modalities.append(('vision', vision_proj))
            
        if audio_emb is not None:
            audio_proj = self.audio_proj(audio_emb)
            audio_proj = self.audio_norm(audio_proj)
            # Align sequence length
            if audio_proj.shape[1] > target_seq_len:
                audio_proj = audio_proj[:, :target_seq_len, :]
            elif audio_proj.shape[1] < target_seq_len:
                padding = torch.zeros(audio_proj.shape[0], target_seq_len - audio_proj.shape[1], 
                                     audio_proj.shape[2], device=audio_proj.device, dtype=audio_proj.dtype)
                audio_proj = torch.cat([audio_proj, padding], dim=1)
            modalities.append(('audio', audio_proj))
            
        if len(modalities) == 0:
            raise ValueError("At least one modality must be provided")
        
        # Single modality - just return processed
        if len(modalities) == 1:
            name, emb = modalities[0]
            if name == 'text':
                return self.text_ffn(emb) + emb
            elif name == 'vision':
                return self.vision_ffn(emb) + emb
            else:  # audio
                return self.audio_ffn(emb) + emb
                
        # Multi-modal fusion
        fused_embeddings = []
        logger.info(f"Processing {len(modalities)} modalities for fusion")
        
        for i, (name_i, emb_i) in enumerate(modalities):
            logger.info(f"  Modality {i}: {name_i}, shape: {emb_i.shape}")
            enhanced_emb = emb_i
            
            # Cross-attention with other modalities
            for j, (name_j, emb_j) in enumerate(modalities):
                if i != j:
                    logger.info(f"    Cross-attending {name_i} -> {name_j}")
                    logger.info(f"      query ({name_i}): {emb_i.shape}")
                    logger.info(f"      key/value ({name_j}): {emb_j.shape}")
                    
                    try:
                        # Cross-attend from modality i to modality j
                        attended_emb, _ = self._cross_attend(emb_i, emb_j, name_i, name_j)
                        logger.info(f"      attended output: {attended_emb.shape}")
                        enhanced_emb = enhanced_emb + attended_emb
                    except Exception as e:
                        logger.error(f"      ERROR in cross-attention: {e}")
                        raise
                    
            # Apply modality-specific processing
            logger.info(f"  Applying {name_i}-specific FFN to shape: {enhanced_emb.shape}")
            try:
                if name_i == 'text':
                    enhanced_emb = self.text_ffn(enhanced_emb) + enhanced_emb
                    enhanced_emb = enhanced_emb * torch.sigmoid(self.text_gate)
                elif name_i == 'vision':
                    enhanced_emb = self.vision_ffn(enhanced_emb) + enhanced_emb  
                    enhanced_emb = enhanced_emb * torch.sigmoid(self.vision_gate)
                else:  # audio
                    enhanced_emb = self.audio_ffn(enhanced_emb) + enhanced_emb
                    enhanced_emb = enhanced_emb * torch.sigmoid(self.audio_gate)
                logger.info(f"  After FFN: {enhanced_emb.shape}")
            except Exception as e:
                logger.error(f"  ERROR in FFN: {e}")
                raise
                
            fused_embeddings.append(enhanced_emb)
            
        # Instead of concatenating (which creates variable lengths),
        # average the modality embeddings to create a fixed-size output
        if len(fused_embeddings) > 1:
            # Stack and average across modalities
            fused = torch.stack(fused_embeddings, dim=0).mean(dim=0)
        else:
            fused = fused_embeddings[0]
            
        # Final self-attention fusion
        fused_attended, _ = self.fusion_attention(fused, fused, fused)
        fused_attended = self.fusion_norm(fused_attended + fused)
        
        # Final FFN
        output = self.fusion_ffn(fused_attended) + fused_attended
        
        return output
        
    def _cross_attend(self, query_emb: torch.Tensor, key_value_emb: torch.Tensor,
                     query_type: str, kv_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform cross-attention between two modalities"""
        
        if query_type == 'text' and kv_type == 'vision':
            return self.text_vision_attention(query_emb, key_value_emb, key_value_emb)
        elif query_type == 'vision' and kv_type == 'text':
            return self.text_vision_attention(key_value_emb, query_emb, query_emb)
        elif query_type == 'text' and kv_type == 'audio':
            return self.text_audio_attention(query_emb, key_value_emb, key_value_emb)
        elif query_type == 'audio' and kv_type == 'text':
            return self.text_audio_attention(key_value_emb, query_emb, query_emb)
        elif query_type == 'vision' and kv_type == 'audio':
            return self.vision_audio_attention(query_emb, key_value_emb, key_value_emb)
        elif query_type == 'audio' and kv_type == 'vision':
            return self.vision_audio_attention(key_value_emb, query_emb, query_emb)
        else:
            # Fallback to text-vision attention
            return self.text_vision_attention(query_emb, key_value_emb, key_value_emb)


class AdaptiveFusionModule(nn.Module):
    """
    Advanced fusion module with adaptive attention weights
    based on input content and context
    """
    
    def __init__(self, fusion_dim: int = 2048, num_heads: int = 16):
        super().__init__()
        
        self.fusion_dim = fusion_dim
        
        # Adaptive attention weight networks
        self.text_weight_net = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 4),
            nn.ReLU(),
            nn.Linear(fusion_dim // 4, 1),
            nn.Sigmoid()
        )
        
        self.vision_weight_net = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 4),
            nn.ReLU(), 
            nn.Linear(fusion_dim // 4, 1),
            nn.Sigmoid()
        )
        
        self.audio_weight_net = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 4),
            nn.ReLU(),
            nn.Linear(fusion_dim // 4, 1), 
            nn.Sigmoid()
        )
        
        # Temporal attention for video sequences
        self.temporal_attention = nn.MultiheadAttention(
            fusion_dim, num_heads, batch_first=True
        )
        
    def forward(self, fused_embeddings: torch.Tensor, 
                text_emb: Optional[torch.Tensor] = None,
                vision_emb: Optional[torch.Tensor] = None,
                audio_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply adaptive fusion weights and temporal modeling
        """
        B, L, D = fused_embeddings.shape
        
        # Compute adaptive weights based on content
        weights = []
        if text_emb is not None:
            text_weight = self.text_weight_net(text_emb.mean(dim=1, keepdim=True))
            weights.append(text_weight)
        if vision_emb is not None:
            vision_weight = self.vision_weight_net(vision_emb.mean(dim=1, keepdim=True))
            weights.append(vision_weight)
        if audio_emb is not None:
            audio_weight = self.audio_weight_net(audio_emb.mean(dim=1, keepdim=True))
            weights.append(audio_weight)
            
        if weights:
            # Normalize weights to sum to 1
            weights = torch.cat(weights, dim=-1)
            weights = F.softmax(weights, dim=-1)
            
            # Apply weights to fused embeddings (simplified)
            weighted_emb = fused_embeddings * weights.mean(dim=-1, keepdim=True)
        else:
            weighted_emb = fused_embeddings
            
        # Temporal attention for video understanding
        temporal_emb, _ = self.temporal_attention(
            weighted_emb, weighted_emb, weighted_emb
        )
        
        return temporal_emb + weighted_emb  # Residual connection


# Alias for backward compatibility
MultiModalFusion = MultiModalFusionModule
