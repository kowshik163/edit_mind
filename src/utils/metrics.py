"""
Video editing metrics and evaluation for autonomous video editor
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


class VideoEditingMetrics:
    """Comprehensive metrics for evaluating video editing quality"""
    
    def __init__(self):
        self.metrics_history = []
    
    def evaluate_video_quality(self, video_path: str) -> Dict[str, float]:
        """Evaluate comprehensive video quality metrics"""
        try:
            metrics = {}
            
            # Load video
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            cap.release()
            
            if not frames:
                return {"overall_quality": 0.0}
            
            # Calculate various quality metrics
            metrics.update(self._calculate_visual_quality(frames))
            metrics.update(self._calculate_temporal_consistency(frames))
            metrics.update(self._calculate_aesthetic_score(frames))
            
            # Overall quality score (weighted average)
            weights = {
                'sharpness': 0.25,
                'brightness': 0.15,
                'contrast': 0.15,
                'temporal_consistency': 0.25,
                'aesthetic_score': 0.20
            }
            
            overall_quality = sum(metrics.get(k, 0.5) * w for k, w in weights.items())
            metrics['overall_quality'] = overall_quality
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating video quality: {e}")
            return {"overall_quality": 0.0}
    
    def _calculate_visual_quality(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """Calculate visual quality metrics"""
        if not frames:
            return {}
        
        # Sample frames for analysis
        sample_frames = frames[::max(1, len(frames)//10)]  # Sample 10 frames
        
        sharpness_scores = []
        brightness_scores = []
        contrast_scores = []
        
        for frame in sample_frames:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Sharpness (variance of Laplacian)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var() / 10000  # Normalize
            sharpness_scores.append(min(sharpness, 1.0))
            
            # Brightness (mean pixel value)
            brightness = gray.mean() / 255.0
            brightness_scores.append(brightness)
            
            # Contrast (standard deviation)
            contrast = gray.std() / 128.0  # Normalize
            contrast_scores.append(min(contrast, 1.0))
        
        return {
            'sharpness': np.mean(sharpness_scores),
            'brightness': np.mean(brightness_scores),
            'contrast': np.mean(contrast_scores)
        }
    
    def _calculate_temporal_consistency(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """Calculate temporal consistency metrics"""
        if len(frames) < 2:
            return {'temporal_consistency': 1.0}
        
        # Calculate frame differences
        differences = []
        for i in range(1, len(frames)):
            prev_frame = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
            curr_frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            
            # Calculate absolute difference
            diff = cv2.absdiff(prev_frame, curr_frame)
            avg_diff = diff.mean() / 255.0
            differences.append(avg_diff)
        
        # Temporal consistency is inverse of variance in differences
        diff_variance = np.var(differences)
        temporal_consistency = max(0.0, 1.0 - diff_variance * 10)  # Scale factor
        
        return {'temporal_consistency': temporal_consistency}
    
    def _calculate_aesthetic_score(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """Calculate aesthetic quality score (simplified)"""
        if not frames:
            return {'aesthetic_score': 0.5}
        
        # Sample middle frame for aesthetic analysis
        mid_frame = frames[len(frames)//2]
        gray = cv2.cvtColor(mid_frame, cv2.COLOR_BGR2GRAY)
        
        # Rule of thirds score (simplified)
        h, w = gray.shape
        third_h, third_w = h//3, w//3
        
        # Check if interesting content is near rule of thirds intersections
        intersections = [
            (third_h, third_w), (third_h, 2*third_w),
            (2*third_h, third_w), (2*third_h, 2*third_w)
        ]
        
        aesthetic_score = 0.5  # Base score
        
        for y, x in intersections:
            # Check local contrast around intersection
            roi = gray[max(0, y-20):min(h, y+20), max(0, x-20):min(w, x+20)]
            if roi.size > 0:
                local_contrast = roi.std() / 128.0
                aesthetic_score += local_contrast * 0.1
        
        aesthetic_score = min(aesthetic_score, 1.0)
        
        return {'aesthetic_score': aesthetic_score}
    
    def evaluate_editing_accuracy(self, 
                                 predicted_cuts: List[float], 
                                 ground_truth_cuts: List[float],
                                 tolerance: float = 0.5) -> Dict[str, float]:
        """Evaluate accuracy of cut point detection"""
        
        if not ground_truth_cuts:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Convert to binary arrays for each time step (assuming 30fps, 1-second resolution)
        max_time = max(max(predicted_cuts, default=0), max(ground_truth_cuts, default=0)) + 1
        time_steps = np.arange(0, max_time, 0.1)  # 100ms resolution
        
        pred_binary = np.zeros(len(time_steps))
        gt_binary = np.zeros(len(time_steps))
        
        # Mark predicted cuts
        for cut_time in predicted_cuts:
            idx = int(cut_time * 10)  # Convert to 100ms resolution
            if idx < len(pred_binary):
                pred_binary[idx] = 1
        
        # Mark ground truth cuts with tolerance
        for cut_time in ground_truth_cuts:
            start_idx = max(0, int((cut_time - tolerance) * 10))
            end_idx = min(len(gt_binary), int((cut_time + tolerance) * 10))
            gt_binary[start_idx:end_idx] = 1
        
        # Calculate metrics
        precision = precision_score(gt_binary, pred_binary, zero_division=0)
        recall = recall_score(gt_binary, pred_binary, zero_division=0)
        f1 = f1_score(gt_binary, pred_binary, zero_division=0)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'num_predicted': len(predicted_cuts),
            'num_ground_truth': len(ground_truth_cuts)
        }
    
    def evaluate_timeline_generation(self, 
                                   predicted_timeline: Dict,
                                   ground_truth_timeline: Dict) -> Dict[str, float]:
        """Evaluate generated timeline against ground truth"""
        
        metrics = {}
        
        # Evaluate cuts
        pred_cuts = predicted_timeline.get('cuts', [])
        gt_cuts = ground_truth_timeline.get('cuts', [])
        
        cut_metrics = self.evaluate_editing_accuracy(pred_cuts, gt_cuts)
        metrics.update({f'cut_{k}': v for k, v in cut_metrics.items()})
        
        # Evaluate transitions (simplified - just count matches)
        pred_transitions = predicted_timeline.get('transitions', [])
        gt_transitions = ground_truth_timeline.get('transitions', [])
        
        if gt_transitions:
            transition_accuracy = len(set(pred_transitions) & set(gt_transitions)) / len(gt_transitions)
        else:
            transition_accuracy = 0.0
        
        metrics['transition_accuracy'] = transition_accuracy
        
        # Overall timeline score
        timeline_score = (cut_metrics.get('f1', 0) + transition_accuracy) / 2
        metrics['timeline_score'] = timeline_score
        
        return metrics
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics for tracking"""
        self.metrics_history.append({
            'step': step,
            'metrics': metrics.copy()
        })
        
        logger.info(f"Step {step} metrics: {metrics}")
    
    def save_metrics(self, save_path: str):
        """Save metrics history to file"""
        with open(save_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def get_summary_stats(self) -> Dict[str, float]:
        """Get summary statistics of all metrics"""
        if not self.metrics_history:
            return {}
        
        all_metrics = {}
        for entry in self.metrics_history:
            for key, value in entry['metrics'].items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)
        
        summary = {}
        for key, values in all_metrics.items():
            summary[f'{key}_mean'] = np.mean(values)
            summary[f'{key}_std'] = np.std(values)
            summary[f'{key}_min'] = np.min(values)
            summary[f'{key}_max'] = np.max(values)
        
        return summary


class DistillationMetrics:
    """Metrics for knowledge distillation evaluation"""
    
    def __init__(self):
        self.distillation_losses = []
    
    def calculate_feature_alignment(self, 
                                  student_features: torch.Tensor,
                                  teacher_features: torch.Tensor) -> float:
        """Calculate alignment between student and teacher features"""
        # Cosine similarity
        student_norm = F.normalize(student_features, p=2, dim=-1)
        teacher_norm = F.normalize(teacher_features, p=2, dim=-1)
        
        similarity = F.cosine_similarity(student_norm, teacher_norm, dim=-1)
        return similarity.mean().item()
    
    def calculate_attention_transfer(self,
                                   student_attention: torch.Tensor,
                                   teacher_attention: torch.Tensor) -> float:
        """Calculate attention transfer quality"""
        # KL divergence between attention distributions
        student_attn = F.softmax(student_attention, dim=-1)
        teacher_attn = F.softmax(teacher_attention, dim=-1)
        
        kl_div = F.kl_div(student_attn.log(), teacher_attn, reduction='batchmean')
        return kl_div.item()
    
    def evaluate_distillation_loss(self,
                                 student_logits: torch.Tensor,
                                 teacher_logits: torch.Tensor,
                                 temperature: float = 4.0) -> Dict[str, float]:
        """Evaluate knowledge distillation loss"""
        
        # Soft targets from teacher
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        
        # KL divergence loss
        distillation_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        distillation_loss *= (temperature ** 2)
        
        # Hard targets accuracy (if available)
        teacher_preds = teacher_logits.argmax(dim=-1)
        student_preds = student_logits.argmax(dim=-1)
        agreement = (teacher_preds == student_preds).float().mean()
        
        return {
            'distillation_loss': distillation_loss.item(),
            'teacher_student_agreement': agreement.item()
        }


class RLHFMetrics:
    """Metrics for Reinforcement Learning from Human Feedback evaluation"""
    
    def __init__(self):
        self.reward_history = []
    
    def calculate_reward_score(self, 
                             generated_video: str,
                             reference_metrics: Dict[str, float]) -> float:
        """Calculate reward score for generated video"""
        
        # This would integrate with actual reward models
        # For now, use a simplified heuristic based on quality metrics
        
        base_reward = reference_metrics.get('overall_quality', 0.5)
        
        # Bonus for good editing practices
        editing_bonus = 0.0
        if reference_metrics.get('temporal_consistency', 0) > 0.7:
            editing_bonus += 0.1
        if reference_metrics.get('cut_f1', 0) > 0.5:
            editing_bonus += 0.1
        
        total_reward = base_reward + editing_bonus
        
        self.reward_history.append(total_reward)
        return total_reward
    
    def get_reward_trend(self) -> Dict[str, float]:
        """Get reward trend statistics"""
        if len(self.reward_history) < 2:
            return {'trend': 0.0, 'improvement': 0.0}
        
        recent_rewards = self.reward_history[-10:]  # Last 10 rewards
        early_rewards = self.reward_history[:10]    # First 10 rewards
        
        recent_mean = np.mean(recent_rewards)
        early_mean = np.mean(early_rewards) if len(early_rewards) > 0 else recent_mean
        
        improvement = recent_mean - early_mean
        
        return {
            'recent_reward': recent_mean,
            'early_reward': early_mean,
            'improvement': improvement,
            'total_episodes': len(self.reward_history)
        }
