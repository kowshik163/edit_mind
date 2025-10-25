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
import librosa  # Import librosa at the top level if used in fallbacks
from scipy import signal # Import signal at the top level if used in fallbacks
import torchaudio # Import torchaudio at the top level if used in fallbacks


try:
    # Assuming models and utils are in the parent directory or properly installed
    from ..models.expert_models import ExpertModels
    from ..utils.distillation_utils import DistillationLoss, FeatureMatching
except ImportError:
    # Fallback for direct execution or different structure
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1])) # Go up one level from 'distillation' to 'src'
    from models.expert_models import ExpertModels
    from utils.distillation_utils import DistillationLoss, FeatureMatching


logger = logging.getLogger(__name__)

# --- Fallback Classes (Moved to Module Level) ---

class AdvancedBeatNetFallback:
    """
    Advanced music analysis fallback using librosa and enhanced algorithms.
    Implements BeatNet-inspired techniques with librosa backend.
    """
    def __init__(self):
        self.sr = 44100
        self.hop_length = 512
        self.frame_length = 2048

    def process_audio(self, audio_data):
        """Advanced tempo, beat, and structure detection"""
        try:
            # Advanced tempo detection with multiple algorithms
            tempo, beats_indices = librosa.beat.beat_track(
                y=audio_data,
                sr=self.sr,
                hop_length=self.hop_length,
                start_bpm=120,
                tightness=100
            )
            beats = librosa.frames_to_time(beats_indices, sr=self.sr, hop_length=self.hop_length)


            # Onset detection with multiple methods
            onset_frames = librosa.onset.onset_detect(
                y=audio_data,
                sr=self.sr,
                hop_length=self.hop_length,
                backtrack=True
            )
            onsets = librosa.frames_to_time(onset_frames, sr=self.sr, hop_length=self.hop_length)


            # Spectral features for rhythm analysis
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio_data, sr=self.sr, hop_length=self.hop_length
            )[0]

            # Rhythm pattern analysis
            rhythm_pattern = self._analyze_advanced_rhythm(beats_indices, spectral_centroids)

            # Musical structure analysis
            structure = self._analyze_structure(audio_data)

            return {
                'tempo': float(tempo),
                'beats': beats.tolist(), # Use time-based beats
                'onsets': onsets.tolist(), # Use time-based onsets
                'rhythm_pattern': rhythm_pattern,
                'spectral_centroids': spectral_centroids.tolist(),
                'structure': structure,
                'beat_strength': self._calculate_beat_strength(audio_data, beats_indices),
                'rhythmic_complexity': self._calculate_rhythmic_complexity(beats_indices)
            }

        except Exception as e:
            logger.warning(f"Advanced BeatNet analysis failed: {e}")
            return self._basic_fallback(audio_data)

    def _analyze_advanced_rhythm(self, beats_indices, spectral_centroids):
        """Advanced rhythm pattern analysis"""
        if len(beats_indices) < 8:
            return 'insufficient_data'

        # Calculate beat intervals and their variation
        intervals = np.diff(beats_indices) # Use frame indices for interval calculation
        avg_interval = np.mean(intervals)
        interval_std = np.std(intervals)

        # Analyze spectral content at beat locations
        beat_spectral_means = []
        for beat_idx in beats_indices:
            # Ensure beat_idx is within the bounds of spectral_centroids
            if beat_idx < len(spectral_centroids):
                 beat_spectral_means.append(spectral_centroids[beat_idx])
            else:
                 # Handle cases where beat index might exceed centroid length (e.g., end of audio)
                 if spectral_centroids.size > 0:
                    beat_spectral_means.append(spectral_centroids[-1]) # Use last available value

        # Check if beat_spectral_means is empty before calculating mean/std
        if not beat_spectral_means:
             spectral_mean_at_beats = 0
             spectral_std_at_beats = 0
        else:
             spectral_mean_at_beats = np.mean(beat_spectral_means)
             spectral_std_at_beats = np.std(beat_spectral_means)


        # Classify rhythm based on multiple factors
        interval_variation = interval_std / avg_interval if avg_interval > 0 else 0

        if interval_variation < 0.1:  # Very regular
            # Convert avg_interval (frames) to seconds then BPM
            avg_interval_sec = avg_interval * self.hop_length / self.sr
            bpm = 60 / avg_interval_sec if avg_interval_sec > 0 else 0
            if bpm > 140:
                return 'fast_regular'
            elif bpm > 90:
                return 'medium_regular'
            else:
                return 'slow_regular'
        elif interval_variation < 0.3: # Moderately varied
             return 'moderately_syncopated'
        else:  # Highly varied
             return 'complex_rhythm'


    def _analyze_structure(self, audio_data):
        """Analyze musical structure (intro, verse, chorus, etc.)"""
        try:
            # Use chroma features for harmonic analysis
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=self.sr, hop_length=self.hop_length)

            # Use MFCC for timbral analysis
            mfcc = librosa.feature.mfcc(y=audio_data, sr=self.sr, n_mfcc=13, hop_length=self.hop_length)

            # Simple structure detection based on feature similarity
            segments = self._detect_segments(chroma, mfcc)

            return {
                'segments': segments,
                'harmonic_progression': chroma.mean(axis=1).tolist(),
                'timbral_features': mfcc.mean(axis=1).tolist()
            }
        except Exception as e:
            logger.warning(f"Structure analysis failed: {e}")
            return {'segments': [], 'harmonic_progression': [], 'timbral_features': []}

    def _detect_segments(self, chroma, mfcc):
        """Detect musical segments based on feature changes"""
        # Ensure features have the same time dimension
        min_len = min(chroma.shape[1], mfcc.shape[1])
        chroma = chroma[:, :min_len]
        mfcc = mfcc[:, :min_len]

        # Combine features and compute self-similarity matrix
        features = np.vstack((chroma, mfcc))
        similarity = librosa.segment.recurrence_matrix(features, mode='affinity', metric='cosine')

        # Find segment boundaries using novelty function
        try:
            # Use a smaller kernel size for potentially finer segmentation
            lag = 2 # Reduced lag for potentially shorter segments
            max_size = 5 # Reduced max size

            # Check if similarity matrix dimensions allow for the lag
            if similarity.shape[0] > lag and similarity.shape[1] > lag:
                 bounds = librosa.segment.agglomerative(similarity, k=lag) # Simplified boundary detection
                 segment_boundaries_frames = librosa.frames_to_time(bounds, sr=self.sr, hop_length=self.hop_length)

                 segments = []
                 start_time = 0.0
                 for boundary_time in segment_boundaries_frames:
                     segments.append({
                         'start_time': start_time,
                         'end_time': boundary_time,
                         'type': 'segment' # Basic type for now
                     })
                     start_time = boundary_time

                 # Add the last segment
                 if start_time < (min_len * self.hop_length / self.sr):
                     segments.append({
                         'start_time': start_time,
                         'end_time': min_len * self.hop_length / self.sr,
                         'type': 'segment'
                     })

                 return segments
            else:
                 logger.warning("Similarity matrix too small for segmentation lag.")
                 return [{'start_time': 0.0, 'end_time': min_len * self.hop_length / self.sr, 'type': 'segment'}]


        except Exception as e:
            logger.warning(f"Librosa segmentation failed: {e}. Falling back to fixed segmentation.")
            # Fallback to fixed-length segments if librosa fails
            segments = []
            segment_duration_sec = 5.0 # 5-second segments
            total_duration_sec = min_len * self.hop_length / self.sr
            num_segments = int(np.ceil(total_duration_sec / segment_duration_sec))

            for i in range(num_segments):
                 start_time = i * segment_duration_sec
                 end_time = min((i + 1) * segment_duration_sec, total_duration_sec)
                 segments.append({
                     'start_time': start_time,
                     'end_time': end_time,
                     'type': 'fixed_segment'
                 })
            return segments


    def _calculate_beat_strength(self, audio_data, beats_indices):
        """Calculate beat strength/salience"""
        try:
            onset_strength = librosa.onset.onset_strength(
                y=audio_data, sr=self.sr, hop_length=self.hop_length
            )

            # Get strength at beat locations
            beat_strengths = []
            for beat_idx in beats_indices:
                # Ensure beat index is within the bounds of onset_strength
                if beat_idx < len(onset_strength):
                    beat_strengths.append(float(onset_strength[beat_idx]))
                elif onset_strength.size > 0: # Handle edge case at the very end
                    beat_strengths.append(float(onset_strength[-1]))


            return np.mean(beat_strengths) if beat_strengths else 0.0
        except Exception as e:
            logger.warning(f"Beat strength calculation failed: {e}")
            return 0.5

    def _calculate_rhythmic_complexity(self, beats_indices):
        """Calculate rhythmic complexity score"""
        if len(beats_indices) < 4:
            return 0.0

        intervals = np.diff(beats_indices)
        mean_interval = np.mean(intervals)
        # Higher std deviation relative to mean interval = more complex rhythm
        complexity = np.std(intervals) / mean_interval if mean_interval > 0 else 0
        return float(np.clip(complexity, 0, 1))

    def _basic_fallback(self, audio_data):
        """Basic fallback analysis"""
        duration = len(audio_data) / self.sr
        return {
            'tempo': 120.0,
            'beats': list(np.arange(0, duration, 0.5)), # Assume 120 BPM
            'onsets': list(np.arange(0, duration, 0.5)),
            'rhythm_pattern': 'unknown',
            'spectral_centroids': [],
            'structure': {'segments': [{'start_time': 0, 'end_time': duration, 'type': 'segment'}], 'harmonic_progression': [], 'timbral_features': []},
            'beat_strength': 0.5,
            'rhythmic_complexity': 0.0
        }

class AdvancedDemucsFallback:
    """
    Advanced audio source separation fallback using sophisticated algorithms.
    Implements Demucs-inspired techniques without requiring the full model.
    """
    def __init__(self):
        self.sr = 44100
        self.n_fft = 2048
        self.hop_length = 512
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.separator_available = True # Assume librosa/scipy are available

    def separate_sources(self, audio_data):
        """Advanced source separation using multiple techniques"""
        try:
            # Convert to torch tensor if needed
            if isinstance(audio_data, np.ndarray):
                # Ensure it's float32 for STFT
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                # Ensure it's mono for separation
                if audio_data.ndim > 1:
                     audio_data = librosa.to_mono(audio_data.T) # librosa expects (channels, samples)
                audio_tensor = torch.from_numpy(audio_data).to(self.device)
            else:
                audio_tensor = audio_data.float().to(self.device)
                if audio_tensor.ndim > 1: # Ensure mono
                    audio_tensor = torch.mean(audio_tensor, dim=0)


            # Advanced spectral analysis
            stft = torch.stft(
                audio_tensor,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=torch.hann_window(self.n_fft).to(self.device), # Added window
                return_complex=True
            )

            magnitude = torch.abs(stft)
            phase = torch.angle(stft)

            # Advanced source separation using multiple methods
            separated = self._advanced_spectral_separation(magnitude, phase, audio_data) # Pass original numpy data if needed

            # Apply post-processing
            separated = self._post_process_separation(separated)

            return separated

        except Exception as e:
            logger.warning(f"Advanced separation failed: {e}")
            return self._fallback_separation(audio_data) # Pass numpy data

    def _advanced_spectral_separation(self, magnitude, phase, original_audio_np): # Added original_audio_np
        """Advanced spectral separation using multiple techniques"""

        # Technique 1: Harmonic-Percussive Separation using Librosa (more robust)
        try:
             # Ensure magnitude is on CPU and numpy for librosa
             magnitude_np = magnitude.cpu().numpy()
             harmonic_np, percussive_np = librosa.decompose.hpss(S=magnitude_np)
             harmonic = torch.from_numpy(harmonic_np).to(self.device)
             percussive = torch.from_numpy(percussive_np).to(self.device)
        except Exception as e:
            logger.warning(f"Librosa HPSS failed: {e}. Using basic median filtering.")
             # Fallback to basic median filtering if librosa fails
            harmonic, percussive = self._harmonic_percussive_separation_basic(magnitude)

        # Technique 2: Intelligent Masking for Vocals/Instrumental
        vocal_mask, instrumental_mask = self._intelligent_masking(magnitude)

        # Reconstruct base components
        vocals_mag = magnitude * vocal_mask
        instrumental_mag = magnitude * instrumental_mask
        harmonic_mag = harmonic # Already has magnitude
        percussive_mag = percussive # Already has magnitude


        # Further refine: Extract drums from percussive, bass from harmonic
        drums_mag = self._extract_drums_mask(percussive_mag)
        bass_mag = self._extract_bass_mask(harmonic_mag)

        # Ensure masks don't overlap excessively or create silence
        # (Simplified energy conservation - real NMF/deep methods are better)
        total_mask_energy = vocal_mask + instrumental_mask + drums_mag + bass_mag + 1e-8
        vocal_mask /= total_mask_energy
        instrumental_mask /= total_mask_energy
        drums_mag /= total_mask_energy # Use drums_mag directly as mask here
        bass_mag /= total_mask_energy # Use bass_mag directly as mask here


        # Reconstruct final sources
        vocals = self._reconstruct_audio(magnitude * vocal_mask, phase)
        # Combine remaining instrumental parts for 'other'
        other_mask = instrumental_mask - drums_mag - bass_mag
        other_mask = torch.clamp(other_mask, min=0.0) # Ensure non-negative
        other = self._reconstruct_audio(magnitude * other_mask, phase)
        drums = self._reconstruct_audio(magnitude * drums_mag, phase)
        bass = self._reconstruct_audio(magnitude * bass_mag, phase)


        return {
            'vocals': vocals,
            'other': other, # Changed from 'instrumental'
            'drums': drums,
            'bass': bass,
            # Optionally include intermediate harmonic/percussive
            # 'harmonic': self._reconstruct_audio(harmonic_mag, phase),
            # 'percussive': self._reconstruct_audio(percussive_mag, phase)
        }

    def _harmonic_percussive_separation_basic(self, magnitude):
         """Basic median filtering for HPSS as fallback"""
         magnitude_np = magnitude.cpu().numpy()
         # Use scipy signal for median filter if librosa failed
         harmonic = torch.from_numpy(signal.medfilt2d(magnitude_np, kernel_size=(1, 17))).to(self.device)
         percussive = torch.from_numpy(signal.medfilt2d(magnitude_np, kernel_size=(17, 1))).to(self.device)
         return harmonic, percussive


    def _intelligent_masking(self, magnitude):
        """Create intelligent masks for vocal/instrumental separation"""
        try:
             # Use librosa's vocal separation approach (based on median filtering)
             magnitude_np = magnitude.cpu().numpy()
             S_filter = librosa.decompose.nn_filter(magnitude_np,
                                                   aggregate=np.median,
                                                   metric='cosine',
                                                   width=int(librosa.time_to_frames(2, sr=self.sr, hop_length=self.hop_length)))
             S_filter = np.minimum(magnitude_np, S_filter) # Ensure filter doesn't exceed original

             # Estimate masks based on filtering
             margin_i, margin_v = 2, 10 # Margins for masking
             power = 2 # Masking power

             mask_i = librosa.util.softmask(S_filter,
                                            margin_i * (magnitude_np - S_filter),
                                            power=power)

             mask_v = librosa.util.softmask(magnitude_np - S_filter,
                                            margin_v * S_filter,
                                            power=power)

             # Convert back to torch tensors
             vocal_mask = torch.from_numpy(mask_v).to(self.device)
             instrumental_mask = torch.from_numpy(mask_i).to(self.device)

             return vocal_mask, instrumental_mask

        except Exception as e:
            logger.warning(f"Intelligent masking failed: {e}. Using basic frequency split.")
             # Basic frequency split as fallback
            freq_bins = magnitude.shape[0]
            vocal_range = slice(int(freq_bins * 0.1), int(freq_bins * 0.6)) # Broader vocal range

            vocal_mask = torch.zeros_like(magnitude)
            vocal_mask[vocal_range, :] = 1.0
            instrumental_mask = 1.0 - vocal_mask
            return vocal_mask, instrumental_mask


    def _extract_drums_mask(self, percussive_mag):
        """Extract drums mask from percussive component"""
        # Drums are often broadband and transient in the percussive layer
        # Focus on energy concentration in time
        temporal_energy = torch.mean(percussive_mag, dim=0) # Energy across frequency bins for each time frame
        threshold = torch.quantile(temporal_energy, 0.85) # Keep top 15% energy frames
        drum_mask_time = (temporal_energy > threshold).float()

        # Apply time mask across all frequencies
        drum_mask = drum_mask_time.unsqueeze(0).repeat(percussive_mag.shape[0], 1)

        # Refine mask based on typical drum frequencies (low-mid)
        freq_bins = percussive_mag.shape[0]
        drum_freq_mask = torch.zeros(freq_bins, device=self.device)
        drum_freq_mask[int(freq_bins*0.02):int(freq_bins*0.4)] = 1.0 # 2% to 40% frequency range

        final_drum_mask = drum_mask * drum_freq_mask.unsqueeze(1)
        return self._smooth_mask(final_drum_mask)


    def _extract_bass_mask(self, harmonic_mag):
        """Extract bass mask from harmonic component"""
        # Bass is typically low-frequency and sustained in the harmonic layer
        bass_mask = torch.zeros_like(harmonic_mag)
        freq_bins = harmonic_mag.shape[0]
        # Bass frequency range (e.g., up to 250 Hz)
        max_bass_bin = int((250 / (self.sr / 2)) * freq_bins)
        bass_range = slice(0, max_bass_bin)

        # Simple energy threshold in the bass range
        bass_energy = harmonic_mag[bass_range, :]
        threshold = torch.quantile(bass_energy, 0.75)
        bass_mask[bass_range, :] = (bass_energy > threshold).float()

        return self._smooth_mask(bass_mask)


    def _smooth_mask(self, mask):
        """Smooth mask to reduce artifacts using a simple Gaussian blur"""
        try:
             # Use torchaudio for smoothing if available, otherwise fallback
             mask_unsqueezed = mask.unsqueeze(0).unsqueeze(0) # Add batch and channel dims
             # Small Gaussian kernel for gentle smoothing
             gaussian_blur = torchaudio.transforms.GaussianBlur(kernel_size=5, sigma=1.0).to(self.device)
             smoothed_mask = gaussian_blur(mask_unsqueezed).squeeze(0).squeeze(0)
             return smoothed_mask
        except Exception as e:
             logger.debug(f"Mask smoothing failed: {e}. Returning original mask.")
             return mask # Return original if smoothing fails


    def _reconstruct_audio(self, magnitude, phase):
        """Reconstruct audio from magnitude and phase"""
        try:
             # Ensure magnitude and phase are on the same device
             magnitude = magnitude.to(self.device)
             phase = phase.to(self.device)

             complex_spec = magnitude * torch.exp(1j * phase)
             # Use hann_window consistent with STFT, ensure correct length
             window = torch.hann_window(self.n_fft).to(self.device)

             audio = torch.istft(
                 complex_spec,
                 n_fft=self.n_fft,
                 hop_length=self.hop_length,
                 window=window,
                 return_complex=False # istft returns real tensor
             )
             # Return as numpy array on CPU
             return audio.cpu().numpy()
        except Exception as e:
            logger.error(f"Audio reconstruction failed: {e}")
            # Return silence on failure
            num_samples = phase.shape[1] * self.hop_length # Estimate length
            return np.zeros(num_samples, dtype=np.float32)


    def _post_process_separation(self, separated):
        """Apply post-processing to improve separation quality"""
        try:
             # Apply gentle filtering to reduce artifacts
             for source_name, source_audio in separated.items():
                 if source_audio is not None and len(source_audio) > 0:
                     # Simple low-pass filter for very high frequencies using scipy
                     nyquist = 0.5 * self.sr
                     cutoff = 0.95 * nyquist # Cutoff at 95% of Nyquist
                     b, a = signal.butter(5, cutoff / nyquist, btype='low')
                     # Ensure audio is float64 for filtfilt if necessary
                     if source_audio.dtype != np.float64:
                          source_audio = source_audio.astype(np.float64)
                     filtered_audio = signal.filtfilt(b, a, source_audio)
                     separated[source_name] = filtered_audio.astype(np.float32) # Convert back to float32
        except Exception as e:
            logger.warning(f"Post-processing failed: {e}")

        return separated

    def _fallback_separation(self, audio_data_np): # Takes numpy array
        """Fallback separation using basic spectral methods (librosa)"""
        try:
            # Basic spectral separation using librosa's HPSS
            harmonic, percussive = librosa.decompose.hpss(
                S=librosa.stft(audio_data_np, n_fft=self.n_fft, hop_length=self.hop_length),
                margin=1.0
            )

            # Reconstruct harmonic and percussive components
            harmonic_audio = librosa.istft(harmonic, hop_length=self.hop_length)
            percussive_audio = librosa.istft(percussive, hop_length=self.hop_length)

            # Very rough estimation of other stems
            vocals = harmonic_audio * 0.4 # Assume vocals are mostly harmonic
            other = harmonic_audio * 0.6
            drums = percussive_audio * 0.7 # Assume drums are mostly percussive
            bass = harmonic_audio * 0.2 # Assume bass is low-freq harmonic

            return {
                'vocals': vocals,
                'other': other,
                'drums': drums,
                'bass': bass,
            }

        except Exception as e:
            logger.error(f"Basic fallback separation failed: {e}")
            # Ultra-basic fallback: return portions of original audio
            length = len(audio_data_np)
            return {
                'vocals': audio_data_np * 0.4,
                'other': audio_data_np * 0.6,
                'drums': audio_data_np[:length//2] * 0.1, # Fake separation
                'bass': audio_data_np[length//2:] * 0.2,
            }


class BeatNetFallback:
    """Minimal music analysis fallback"""
    def __init__(self):
        self.sr = 44100

    def process_audio(self, audio_data):
        """Basic rhythm detection"""
        duration = len(audio_data) / self.sr
        return {
            'tempo': 120.0,
            'beats': list(np.arange(0, duration, 0.5)), # Assume 120 BPM
            'onsets': list(np.arange(0, duration, 0.5)),
            'rhythm_pattern': 'unknown',
            'spectral_centroids': [],
            'structure': {'segments': [{'start_time': 0, 'end_time': duration, 'type': 'segment'}], 'harmonic_progression': [], 'timbral_features': []},
            'beat_strength': 0.5,
            'rhythmic_complexity': 0.0
        }

class DemucsFallback:
    """Basic audio source separation fallback"""
    def __init__(self):
        self.sr = 44100

    def separate_sources(self, audio_data):
        """Basic source separation"""
        length = len(audio_data)
        return {
            'vocals': audio_data * 0.4,
            'other': audio_data * 0.6, # Renamed from instrumental
            'drums': audio_data[:length//2] * 0.1, # Fake separation for drums
            'bass': audio_data[length//2:] * 0.2, # Fake separation for bass
        }

# --- KnowledgeDistiller Class ---

class KnowledgeDistiller:
    """
    Enhanced distillation from state-of-the-art teacher models:
    - RT-DETR (Real-time Detection Transformer)
    - HQ-SAM (High-Quality Segment Anything Model)
    - Whisper (speech recognition)
    - BeatNet (music/rhythm analysis)
    - Demucs (audio source separation)
    - RAFT (optical flow) - Placeholder added
    """

    def __init__(self, config: DictConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load expert teacher models (assuming ExpertModels handles loading correctly now)
        self.expert_models = ExpertModels(config)

        # Initialize advanced teacher models (will load or use fallbacks)
        self._initialize_advanced_teachers()

        # Distillation loss functions
        # Ensure training.phase2 exists or provide defaults
        phase2_config = config.get('training', {}).get('phase2', {'temperature': 3.0, 'alpha': 0.7})
        self.distillation_loss = DistillationLoss(
            temperature=phase2_config.get('temperature', 3.0),
            alpha=phase2_config.get('alpha', 0.7)
        )

        # Feature matching for intermediate representations
        self.feature_matcher = FeatureMatching()

        # Placeholders/Attributes for advanced teacher models
        self.rt_detr = None
        self.hq_sam = None
        self.beatnet = None
        self.demucs = None
        self.raft = None # Added placeholder for RAFT

    def _initialize_advanced_teachers(self):
        """Initialize state-of-the-art teacher models or fallbacks"""
        logger.info("🔬 Initializing advanced teacher models...")

        # Initialize RT-DETR for object detection
        self.rt_detr = self._load_rt_detr()
        self._validate_model_authenticity('RT-DETR', self.rt_detr)

        # Initialize HQ-SAM for segmentation
        self.hq_sam = self._load_hq_sam()
        self._validate_model_authenticity('HQ-SAM', self.hq_sam)

        # Initialize BeatNet for music analysis
        self.beatnet = self._load_beatnet()
        self._validate_model_authenticity('BeatNet', self.beatnet)

        # Initialize Demucs for audio separation
        self.demucs = self._load_demucs()
        self._validate_model_authenticity('Demucs', self.demucs)

        # Initialize RAFT for optical flow (Placeholder)
        self.raft = self._load_raft()
        self._validate_model_authenticity('RAFT', self.raft)


        # Log final model status
        self._log_model_status()

    # --- Model Loading Methods (_load_rt_detr, _load_hq_sam, etc.) ---
    # These methods attempt to load real models and fall back gracefully.
    # Keep the structure similar to the original file but ensure correct class placement.

    def _load_rt_detr(self):
        """Loads RT-DETR or fallback."""
        # --- (Keep the complex loading logic from the original file) ---
        # --- (Ensure it returns a dict or None) ---
        # Example structure:
        try:
             # Try loading real RT-DETR (using transformers or original repo)
             # ... loading logic ...
             logger.info("✅ Successfully loaded real RT-DETR")
             return { 'model': model, 'processor': processor, 'type': 'real_rtdetr', 'model_name': '...', 'capabilities': {...} }
        except Exception as e:
             logger.warning(f"Real RT-DETR failed: {e}. Falling back.")
             try:
                 # Try DETR
                 # ... loading logic ...
                 logger.info("⚠️ Using DETR fallback for RT-DETR")
                 return { 'model': model, 'processor': processor, 'type': 'detr_fallback', 'model_name': '...', 'capabilities': {...} }
             except Exception as e2:
                 logger.error(f"❌ No suitable object detection model found: {e2}")
                 return None # Return None on complete failure


    def _load_hq_sam(self):
        """Loads HQ-SAM or fallback."""
        # --- (Keep the complex loading logic from the original file) ---
        # --- (Ensure it returns a dict or None) ---
        try:
            # Try loading real SAM (transformers)
            # ... loading logic ...
            logger.info("✅ Successfully loaded real SAM")
            return { 'model': model, 'processor': processor, 'type': 'real_sam', 'model_name': '...', 'capabilities': {...} }
        except Exception as e:
            logger.warning(f"Real SAM failed: {e}. Falling back.")
            try:
                # Try DeepLabV3
                # ... loading logic ...
                logger.info("⚠️ Using DeepLabV3 fallback for SAM")
                return { 'model': model, 'transform': transform, 'type': 'deeplab_fallback', 'model_name': '...', 'capabilities': {...} }
            except Exception as e2:
                logger.error(f"❌ No suitable segmentation model found: {e2}")
                return None


    def _load_beatnet(self):
        """Loads real BeatNet or the AdvancedBeatNetFallback."""
        try:
            # Try loading real BeatNet package
            import BeatNet
            from BeatNet.BeatNet import BeatNet as RealBeatNet
            beatnet_model = RealBeatNet(model='1', mode='offline', inference_model='DBN', plot=[], thread=False) # Example init
            logger.info("✅ Successfully loaded real BeatNet model")
            return {
                'model': beatnet_model,
                'type': 'real_beatnet',
                'model_name': 'BeatNet-1',
                'capabilities': {...}, # Add capabilities
                'processor': self._create_beatnet_processor(beatnet_model)
            }
        except ImportError:
            logger.warning("BeatNet package not installed. Using librosa fallback.")
            try:
                # Ensure librosa is imported if using fallback
                import librosa
                return {
                     'model': AdvancedBeatNetFallback(), # Use the class defined at module level
                     'type': 'beatnet_fallback_advanced',
                     'model_name': 'librosa_beat_analysis',
                     'capabilities': {...}, # Define fallback capabilities
                     'processor': lambda audio_data: AdvancedBeatNetFallback().process_audio(audio_data) # Simple wrapper
                }
            except ImportError:
                 logger.error("❌ Librosa not found. Cannot provide BeatNet fallback.")
                 return None
        except Exception as e:
            logger.error(f"❌ Failed to initialize BeatNet or fallback: {e}")
            return None


    def _load_demucs(self):
        """Loads real Demucs or the AdvancedDemucsFallback."""
        try:
            # Try loading real Demucs package
            import demucs.api
            from demucs import pretrained
            model_name = 'htdemucs' # Or get from config
            model = pretrained.get_model(model_name)
            model.eval()
            model.to(self.device)
            logger.info(f"✅ Successfully loaded real Demucs model: {model_name}")
            return {
                'model': model,
                'type': 'real_demucs',
                'model_name': model_name,
                'capabilities': {...}, # Add capabilities
                'processor': self._create_demucs_processor(model, model_name)
            }
        except ImportError:
            logger.warning("Demucs package not installed. Using advanced fallback.")
            try:
                 # Ensure necessary libs are imported
                 import librosa
                 import torchaudio
                 from scipy import signal
                 return {
                      'model': AdvancedDemucsFallback(), # Use the class defined at module level
                      'type': 'demucs_fallback_advanced',
                      'model_name': 'spectral_separation',
                      'capabilities': {...}, # Define fallback capabilities
                      'processor': lambda audio_data: AdvancedDemucsFallback().separate_sources(audio_data) # Simple wrapper
                 }
            except ImportError as e:
                 logger.error(f"❌ Librosa/Scipy/Torchaudio not found. Cannot provide Demucs fallback. {e}")
                 return None
        except Exception as e:
            logger.error(f"❌ Failed to initialize Demucs or fallback: {e}")
            return None

    def _load_raft(self):
        """Loads RAFT model for optical flow (Placeholder)."""
        logger.warning("RAFT model loading is currently a placeholder.")
        # In a real implementation:
        # try:
        #    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        #    weights = Raft_Large_Weights.DEFAULT
        #    model = raft_large(weights=weights)
        #    model.eval().to(self.device)
        #    transform = weights.transforms()
        #    logger.info("✅ Successfully loaded RAFT model")
        #    return {'model': model, 'transform': transform, 'type': 'real_raft', 'model_name': 'raft_large'}
        # except ImportError:
        #    logger.warning("Torchvision optical flow models not available.")
        # except Exception as e:
        #    logger.warning(f"Failed to load RAFT model: {e}")
        return {'model': None, 'type': 'raft_placeholder', 'model_name': 'none'} # Placeholder


    # --- Validation and Logging Methods ---
    def _validate_model_authenticity(self, model_name: str, model_info: Optional[Dict]):
        # --- (Keep logic from original file) ---
        if model_info is None:
            logger.error(f"❌ {model_name}: Failed to load")
            return

        model_type = model_info.get('type', 'unknown')
        model_path = model_info.get('model_name', 'unknown')

        # Check if using real pre-trained models vs fallbacks
        if 'fallback' in model_type.lower():
            logger.warning(f"⚠️  {model_name}: Using FALLBACK implementation ({model_path})")
            if 'beatnet' in model_name.lower() or 'demucs' in model_name.lower():
                 logger.warning(f"   Consider installing the official '{model_name.lower()}' package for full functionality.")
        elif 'real' in model_type.lower() or 'teacher' in model_type.lower() or 'configured' in model_type.lower():
            logger.info(f"✅ {model_name}: Using AUTHENTIC pre-trained model ({model_path})")
        elif any(indicator in model_path.lower() for indicator in ['facebook', 'microsoft', 'openai', 'lyuwenyu', 'pekingU', 'google']):
            logger.info(f"✅ {model_name}: Using official pre-trained model ({model_path})")
        elif model_type != 'raft_placeholder': # Don't warn for placeholder
            logger.warning(f"🔍 {model_name}: Using alternative implementation ({model_path})")


    def _log_model_status(self):
        # --- (Keep logic from original file, add RAFT) ---
        models_status = {
            'RT-DETR (Object Detection)': self.rt_detr,
            'HQ-SAM (Segmentation)': self.hq_sam,
            'BeatNet (Music Analysis)': self.beatnet,
            'Demucs (Audio Separation)': self.demucs,
            'RAFT (Optical Flow)': self.raft # Added RAFT
        }

        authentic_count = 0
        fallback_count = 0
        failed_count = 0
        placeholder_count = 0


        logger.info("🏗️  Advanced Teacher Models Status:")

        for model_name, model_info in models_status.items():
            if model_info is None:
                logger.info(f"   ❌ {model_name}: NOT LOADED")
                failed_count += 1
            else:
                model_type = model_info.get('type', '')
                if 'placeholder' in model_type.lower():
                     logger.info(f"   ⚪ {model_name}: PLACEHOLDER")
                     placeholder_count += 1
                elif 'fallback' in model_type.lower():
                    logger.info(f"   ⚠️  {model_name}: FALLBACK")
                    fallback_count += 1
                else:
                    logger.info(f"   ✅ {model_name}: AUTHENTIC/OFFICIAL")
                    authentic_count += 1

        total_models = len(models_status)
        # Exclude placeholders from success rate calculation if desired
        # success_rate = (authentic_count + fallback_count) / (total_models - placeholder_count) * 100 if (total_models - placeholder_count) > 0 else 0
        success_rate = (authentic_count + fallback_count) / total_models * 100 # Include placeholders in total


        logger.info(f"📊 Model Loading Summary:")
        logger.info(f"   Authentic/Official: {authentic_count}/{total_models}")
        logger.info(f"   Fallback Models: {fallback_count}/{total_models}")
        logger.info(f"   Placeholders: {placeholder_count}/{total_models}") # Added placeholders
        logger.info(f"   Failed Models: {failed_count}/{total_models}")
        logger.info(f"   Success Rate (incl. fallbacks): {success_rate:.1f}%")

        if authentic_count == total_models - placeholder_count: # Adjusted for placeholder
            logger.info("🎯 Perfect! All essential authentic teacher models loaded successfully.")
        elif authentic_count > fallback_count:
            logger.info("👍 Good! Majority of authentic/official models loaded.")
        elif fallback_count > 0:
            logger.warning("⚠️  Warning! Using fallback implementations - consider installing proper packages for best results.")
        elif failed_count > 0: # Check failed count explicitly
            logger.error("❌ Critical! Some essential teacher models failed to load.")

        # Provide installation guidance if needed
        if fallback_count > 0 or failed_count > 0:
            self._provide_installation_guidance()


    def _provide_installation_guidance(self):
         # --- (Keep logic, maybe update package names if needed) ---
        logger.info("📋 Installation Guidance for Authentic Models:")

        if self.rt_detr is None or 'fallback' in self.rt_detr.get('type', '').lower():
            logger.info("   RT-DETR/DETR:")
            logger.info("     pip install transformers torch torchvision")
            logger.info("     # Models like facebook/detr-resnet-50 or PekingU/* will be downloaded from Hugging Face.")

        if self.hq_sam is None or 'fallback' in self.hq_sam.get('type', '').lower():
            logger.info("   HQ-SAM/SAM:")
            logger.info("     pip install transformers torch torchvision")
            logger.info("     # Models like facebook/sam-vit-* will be downloaded from Hugging Face Hub.")

        if self.beatnet is None or 'fallback' in self.beatnet.get('type', '').lower():
            logger.info("   BeatNet:")
            logger.info("     pip install BeatNet librosa") # Added librosa
            logger.info("     # Official BeatNet package for advanced music analysis.")

        if self.demucs is None or 'fallback' in self.demucs.get('type', '').lower():
            logger.info("   Demucs:")
            logger.info("     pip install demucs torch torchaudio") # Added torch/torchaudio
            logger.info("     # Official Demucs for professional audio separation.")

        if self.raft is not None and 'placeholder' in self.raft.get('type', '').lower():
             logger.info("   RAFT (Optical Flow):")
             logger.info("     Ensure you have a recent version of torchvision:")
             logger.info("     pip install --upgrade torchvision")


    # --- Processor Creation Methods ---
    def _create_beatnet_processor(self, beatnet_model):
        """Creates a callable processor for the real BeatNet model."""
        def process_audio_with_beatnet(audio_data_np): # Takes numpy array
            try:
                # BeatNet expects numpy array, mono
                if audio_data_np.ndim > 1:
                    audio_data_np = librosa.to_mono(audio_data_np.T) # Ensure mono

                # Process with BeatNet
                # The output format might vary slightly based on BeatNet version
                output = beatnet_model.process(audio_data_np) # output is likely a structured array or dict

                # Convert structured array to dict if necessary
                if isinstance(output, np.ndarray) and output.dtype.names:
                     processed_output = {name: output[name].tolist() for name in output.dtype.names}
                elif isinstance(output, dict):
                     processed_output = output # Assume it's already a dict
                else:
                     logger.warning("Unexpected BeatNet output format.")
                     processed_output = {} # Handle unexpected format

                # Ensure consistent keys (match fallback keys)
                final_output = {
                    'beats': processed_output.get('beats', []),
                    'downbeats': processed_output.get('downbeats', []),
                    'tempo': processed_output.get('tempo', [120.0])[0], # Often returned as list
                    'time_signature': processed_output.get('time_signature', [4, 4]),
                    'beat_activation': processed_output.get('beat_activation', []),
                    'downbeat_activation': processed_output.get('downbeat_activation', [])
                }
                return final_output

            except Exception as e:
                logger.warning(f"Real BeatNet processing failed: {e}")
                # Fallback within the processor if real model fails at runtime
                return AdvancedBeatNetFallback().process_audio(audio_data_np)


        return process_audio_with_beatnet


    def _create_demucs_processor(self, model, model_name):
        """Creates a callable processor for the real Demucs model."""
        def separate_with_demucs(audio_data_np): # Takes numpy array
            try:
                from demucs.apply import apply_model
                from demucs.audio import convert_audio

                # Ensure audio is float32 numpy array
                if audio_data_np.dtype != np.float32:
                     audio_data_np = audio_data_np.astype(np.float32)

                # Convert numpy to torch tensor, ensure correct shape [channels, samples]
                audio_tensor = torch.from_numpy(audio_data_np)
                if audio_tensor.ndim == 1:
                    audio_tensor = audio_tensor.unsqueeze(0) # Add channel dim for mono
                elif audio_tensor.ndim == 2 and audio_tensor.shape[0] > audio_tensor.shape[1]:
                     audio_tensor = audio_tensor.T # Transpose if (samples, channels)

                # Convert to model's expected sample rate and format
                audio_tensor = convert_audio(audio_tensor, self.sr, model.samplerate, model.audio_channels) # Use self.sr
                audio_tensor = audio_tensor.to(self.device)

                # Apply Demucs separation (add batch dimension)
                with torch.no_grad():
                    # apply_model expects batch dimension
                    separated_tensor = apply_model(model, audio_tensor.unsqueeze(0), device=self.device)[0] # Remove batch dim from output

                # Extract stems (Demucs returns tensor [stems, channels, samples])
                stems_np = {}
                stem_names = model.sources if hasattr(model, 'sources') else ['drums', 'bass', 'other', 'vocals']
                for i, source in enumerate(stem_names):
                    if i < separated_tensor.shape[0]:
                         # Convert back to original sample rate and mono/stereo based on input
                         stem_audio_tensor = convert_audio(separated_tensor[i], model.samplerate, self.sr, audio_data_np.ndim)
                         stems_np[source] = stem_audio_tensor.cpu().numpy()
                         # If original was mono, ensure output is mono
                         if audio_data_np.ndim == 1 and stems_np[source].ndim > 1:
                             stems_np[source] = stems_np[source].mean(axis=0)


                return stems_np

            except Exception as e:
                logger.warning(f"Real Demucs separation failed: {e}")
                # Fallback within the processor
                return AdvancedDemucsFallback().separate_sources(audio_data_np)

        return separate_with_demucs


    # --- Distillation Methods ---
    # Keep the structure, ensure they handle None models gracefully
    # and use the correct processor/model attributes.

    def distill_rt_detr_knowledge(self, student_model: nn.Module, video_frames: torch.Tensor) -> Dict[str, Any]:
        """Distill object detection knowledge from RT-DETR/fallback."""
        if self.rt_detr is None or self.rt_detr.get('model') is None:
             logger.warning("No object detection model available for distillation.")
             return {}

        logger.info("🎯 Distilling object detection knowledge...")
        model_info = self.rt_detr
        model = model_info['model']
        processor = model_info.get('processor') # May be None for fallback
        transform = model_info.get('transform') # For torchvision fallback

        distilled_data = {
            'spatial_understanding_target': [],
            'object_locations_target': [],
            'object_confidences_target': [],
            'object_classes_target': []
        }

        try:
            with torch.no_grad():
                 B, T, C, H, W = video_frames.shape
                 frames_flat = video_frames.view(B * T, C, H, W).to(self.device)

                 # Process frames in smaller batches if necessary
                 batch_size_det = 16 # Adjust based on GPU memory
                 for i in range(0, frames_flat.size(0), batch_size_det):
                      batch_frames = frames_flat[i : i + batch_size_det]

                      if model_info['type'] == 'retinanet_fallback':
                           # --- RetinaNet Fallback Logic ---
                           # Requires preprocessing with transform
                           batch_processed = torch.stack([transform(frame.cpu()) for frame in batch_frames]).to(self.device)
                           detections_batch = model(batch_processed) # List of dicts

                           for detection in detections_batch:
                                boxes = detection['boxes']
                                scores = detection['scores']
                                labels = detection['labels']
                                # Keep only high-confidence detections
                                keep = scores > 0.5
                                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

                                distilled_data['object_locations_target'].append(boxes.cpu())
                                distilled_data['object_confidences_target'].append(scores.cpu())
                                distilled_data['object_classes_target'].append(labels.cpu())
                                # Simple spatial feature: average box center and size
                                if boxes.numel() > 0:
                                     centers = (boxes[:, :2] + boxes[:, 2:]) / 2
                                     sizes = boxes[:, 2:] - boxes[:, :2]
                                     spatial_feat = torch.cat([centers.mean(dim=0), sizes.mean(dim=0)], dim=0)
                                else:
                                     spatial_feat = torch.zeros(4, device=self.device)
                                distilled_data['spatial_understanding_target'].append(spatial_feat.cpu())

                      elif processor: # DETR/RT-DETR
                           # --- DETR/RT-DETR Logic ---
                           from PIL import Image
                           pil_images = [Image.fromarray((f.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)) for f in batch_frames]

                           inputs = processor(images=pil_images, return_tensors="pt").to(self.device)
                           outputs = model(**inputs)

                           # Post-process results
                           target_sizes = torch.tensor([img.size[::-1] for img in pil_images], device=self.device) # height, width
                           results_batch = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)

                           for results in results_batch:
                                boxes = results['boxes']
                                scores = results['scores']
                                labels = results['labels']

                                distilled_data['object_locations_target'].append(boxes.cpu())
                                distilled_data['object_confidences_target'].append(scores.cpu())
                                distilled_data['object_classes_target'].append(labels.cpu())

                                # Use last hidden state mean as spatial feature
                                if hasattr(outputs, 'last_hidden_state'):
                                     # Need to handle batch dimension correctly here
                                     # Assuming outputs.last_hidden_state is (batch_size_det, num_queries, dim)
                                     # We need features per image
                                     spatial_feat = outputs.last_hidden_state[len(distilled_data['spatial_understanding_target']) % batch_size_det].mean(dim=0) # Mean over queries
                                else:
                                     spatial_feat = torch.zeros(256, device=self.device) # Fallback dim
                                distilled_data['spatial_understanding_target'].append(spatial_feat.cpu())
                      else:
                           logger.warning("No processor/transform found for object detection model.")


            # Aggregate features across frames (simple mean for now)
            if distilled_data['spatial_understanding_target']:
                 spatial_knowledge = torch.stack(distilled_data['spatial_understanding_target']).mean(dim=0).unsqueeze(0)
            else:
                 spatial_knowledge = torch.zeros((1, 4 if model_info['type'] == 'retinanet_fallback' else 256), device=self.device)


            # Reshape list of tensors per frame into tensors per batch item
            # This requires knowing the original T dimension
            def aggregate_per_video(data_list, original_T):
                 if not data_list: return []
                 return [torch.cat(data_list[j*original_T:(j+1)*original_T], dim=0) for j in range(B)]

            locations_agg = aggregate_per_video(distilled_data['object_locations_target'], T)
            confidences_agg = aggregate_per_video(distilled_data['object_confidences_target'], T)
            classes_agg = aggregate_per_video(distilled_data['object_classes_target'], T)


            return {
                'spatial_understanding_target': spatial_knowledge, # Target for student's spatial features
                'object_locations_target': locations_agg, # List[Tensor(num_objs, 4)] per video
                'object_confidences_target': confidences_agg,# List[Tensor(num_objs)] per video
                'object_classes_target': classes_agg # List[Tensor(num_objs)] per video
            }

        except Exception as e:
            logger.error(f"Object detection distillation failed: {e}", exc_info=True)
            return {}

    def distill_hq_sam_knowledge(self, student_model: nn.Module, video_frames: torch.Tensor) -> Dict[str, Any]:
        """Distill segmentation knowledge from HQ-SAM/fallback."""
        if self.hq_sam is None or self.hq_sam.get('model') is None:
             logger.warning("No segmentation model available for distillation.")
             return {}

        logger.info("🎨 Distilling segmentation knowledge...")
        model_info = self.hq_sam
        model = model_info['model']
        processor = model_info.get('processor')
        transform = model_info.get('transform')

        distilled_data = {
            'segmentation_map_target': [],
            'boundary_features_target': []
        }

        try:
            with torch.no_grad():
                B, T, C, H, W = video_frames.shape
                frames_flat = video_frames.view(B * T, C, H, W).to(self.device)

                batch_size_seg = 4 # Adjust based on GPU memory
                for i in range(0, frames_flat.size(0), batch_size_seg):
                    batch_frames = frames_flat[i : i + batch_size_seg]

                    if processor: # SAM
                        # --- SAM Logic ---
                        from PIL import Image
                        pil_images = [Image.fromarray((f.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)) for f in batch_frames]

                        # SAM needs input points or boxes, let's use the whole image for a general mask
                        input_points = [[[H // 2, W // 2]]] * len(pil_images) # Center point

                        inputs = processor(pil_images, input_points=input_points, return_tensors="pt").to(self.device)
                        outputs = model(**inputs)
                        # High-resolution masks (upsampled)
                        masks = processor.image_processor.post_process_masks(
                            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
                        )[0] # Assuming post_process_masks returns list per batch item

                        # Take the highest scoring mask for simplicity as target
                        # masks shape: (batch_size_seg, num_masks, H, W)
                        for single_image_masks in masks:
                             if single_image_masks.numel() > 0:
                                 # Find best mask (e.g., largest area or highest predicted IoU if available)
                                 # For simplicity, take the first one or average? Let's take first.
                                 best_mask = single_image_masks[0, 0] # Taking first mask of first query point
                                 distilled_data['segmentation_map_target'].append(best_mask.cpu())
                                 # Extract boundary features
                                 boundary_feat = self._extract_boundary_features_torch(best_mask.unsqueeze(0).unsqueeze(0))
                                 distilled_data['boundary_features_target'].append(boundary_feat.cpu())

                             else: # No mask found
                                 distilled_data['segmentation_map_target'].append(torch.zeros((H, W)))
                                 distilled_data['boundary_features_target'].append(torch.zeros((H*W))) # Placeholder size


                    elif transform: # DeepLabV3
                         # --- DeepLabV3 Logic ---
                         batch_processed = torch.stack([transform(frame.cpu()) for frame in batch_frames]).to(self.device)
                         output = model(batch_processed)['out'] # (batch, num_classes, H', W')
                         # Get semantic segmentation map (argmax over classes)
                         seg_map = torch.argmax(output, dim=1) # (batch, H', W')

                         for single_seg_map in seg_map:
                              # Resize back to original frame size? Or keep processed size? Keep processed for now.
                              distilled_data['segmentation_map_target'].append(single_seg_map.float().cpu())
                              boundary_feat = self._extract_boundary_features_torch(single_seg_map.unsqueeze(0).unsqueeze(0).float())
                              distilled_data['boundary_features_target'].append(boundary_feat.cpu())
                    else:
                         logger.warning("No processor/transform found for segmentation model.")


            # Aggregate features (simple mean)
            if distilled_data['segmentation_map_target']:
                 # Resize all maps to a common size (e.g., target H, W) before stacking/averaging
                 target_H, target_W = H, W # Use original H, W
                 resized_maps = [F.interpolate(m.unsqueeze(0).unsqueeze(0), size=(target_H, target_W), mode='nearest').squeeze()
                                 for m in distilled_data['segmentation_map_target']]
                 segmentation_knowledge = torch.stack(resized_maps).mean(dim=0).unsqueeze(0)

                 # Boundary features need careful aggregation, maybe mean or max pool
                 # For simplicity, let's just take the features of the first frame's boundaries
                 boundary_knowledge = distilled_data['boundary_features_target'][0].unsqueeze(0) if distilled_data['boundary_features_target'] else torch.zeros((1, H*W))

            else:
                 segmentation_knowledge = torch.zeros((1, H, W))
                 boundary_knowledge = torch.zeros((1, H*W))


            return {
                'segmentation_understanding_target': segmentation_knowledge, # Target for student's segmentation features
                'object_boundaries_target': boundary_knowledge, # Target for student's boundary features
                # Optionally return raw masks per frame if needed for loss calculation
                # 'segmentation_maps_per_frame': distilled_data['segmentation_map_target']
            }

        except Exception as e:
            logger.error(f"Segmentation distillation failed: {e}", exc_info=True)
            return {}

    def distill_beatnet_knowledge(self, student_model: nn.Module, audio_data: torch.Tensor) -> Dict[str, Any]:
        """Distill music structure knowledge from BeatNet/fallback."""
        if self.beatnet is None or self.beatnet.get('model') is None:
            logger.warning("No music analysis model available for distillation.")
            return {}

        logger.info("🎵 Distilling music analysis knowledge...")
        model_info = self.beatnet
        processor = model_info['processor']

        distilled_data = {
            'tempo_target': [],
            'beats_target': [],
            'rhythm_complexity_target': [],
            'structure_target': [] # E.g., segment boundaries
        }

        try:
             # Process audio batch
             # Processor expects numpy, handle batching
             B = audio_data.shape[0]
             for i in range(B):
                  audio_np_mono = audio_data[i].mean(dim=0).cpu().numpy() # Ensure mono numpy
                  music_analysis = processor(audio_np_mono)

                  if 'error' not in music_analysis:
                      distilled_data['tempo_target'].append(music_analysis.get('tempo', 120.0))
                      distilled_data['beats_target'].append(torch.tensor(music_analysis.get('beats', []), dtype=torch.float32))
                      distilled_data['rhythm_complexity_target'].append(music_analysis.get('rhythmic_complexity', 0.0))
                      # Represent structure simply, e.g., number of segments
                      distilled_data['structure_target'].append(len(music_analysis.get('structure', {}).get('segments', [])))
                  else:
                      # Append default values on error
                      distilled_data['tempo_target'].append(120.0)
                      distilled_data['beats_target'].append(torch.tensor([]))
                      distilled_data['rhythm_complexity_target'].append(0.0)
                      distilled_data['structure_target'].append(1)


             # Aggregate rhythm knowledge (e.g., average tempo, complexity)
             avg_tempo = np.mean(distilled_data['tempo_target']) if distilled_data['tempo_target'] else 120.0
             avg_complexity = np.mean(distilled_data['rhythm_complexity_target']) if distilled_data['rhythm_complexity_target'] else 0.0
             avg_structure = np.mean(distilled_data['structure_target']) if distilled_data['structure_target'] else 1.0


             rhythm_knowledge = torch.tensor([avg_tempo, avg_complexity, avg_structure], device=self.device)


             # Beats need careful handling for distillation loss (e.g., align sequences)
             # For now, just provide the list of tensors per batch item
             beats_per_item = distilled_data['beats_target']


             return {
                'music_understanding_target': rhythm_knowledge.unsqueeze(0), # Add batch dim
                'tempo_target': avg_tempo, # Single value for the batch (average)
                'beat_locations_target': beats_per_item, # List[Tensor] for beats
                # 'onset_times': onsets # If needed
             }

        except Exception as e:
            logger.error(f"Music analysis distillation failed: {e}", exc_info=True)
            return {}

    def distill_demucs_knowledge(self, student_model: nn.Module, audio_data: torch.Tensor) -> Dict[str, Any]:
        """Distill audio source separation knowledge from Demucs/fallback."""
        if self.demucs is None or self.demucs.get('model') is None:
             logger.warning("No audio separation model available for distillation.")
             return {}

        logger.info("🎶 Distilling audio separation knowledge...")
        model_info = self.demucs
        processor = model_info['processor']

        distilled_data = {
            'separated_sources_target': [], # List of dicts {'vocals': tensor, ...}
            'source_balance_target': [] # List of dicts {'vocals': float, ...}
        }

        try:
             # Process audio batch
             B = audio_data.shape[0]
             for i in range(B):
                  audio_np_stereo = audio_data[i].cpu().numpy() # Processor expects numpy
                  # Ensure stereo if model expects it, otherwise mono handled inside processor
                  if audio_np_stereo.ndim == 1:
                       audio_np_stereo = np.stack([audio_np_stereo, audio_np_stereo], axis=0) # Convert mono to stereo if needed by processor

                  separated_sources_np = processor(audio_np_stereo)

                  if 'error' not in separated_sources_np:
                       # Convert numpy arrays back to tensors
                       separated_sources_tensors = {k: torch.from_numpy(v).to(self.device) for k, v in separated_sources_np.items()}
                       distilled_data['separated_sources_target'].append(separated_sources_tensors)

                       # Calculate balance
                       balance = self._analyze_audio_balance_simple(separated_sources_np)
                       distilled_data['source_balance_target'].append(balance)
                  else:
                       # Handle error case - append empty dicts or defaults
                       distilled_data['separated_sources_target'].append({})
                       distilled_data['source_balance_target'].append({})


             # Aggregate knowledge (e.g., average balance)
             avg_balance = {}
             if distilled_data['source_balance_target']:
                  keys = distilled_data['source_balance_target'][0].keys()
                  for key in keys:
                       avg_balance[key] = np.mean([d.get(key, 0) for d in distilled_data['source_balance_target']])

             audio_knowledge = torch.tensor(list(avg_balance.values()), device=self.device) if avg_balance else torch.zeros(4, device=self.device)


             # Separated sources are the main target for distillation loss
             # This list contains dicts of tensors per batch item
             separated_targets = distilled_data['separated_sources_target']


             return {
                'audio_understanding_target': audio_knowledge.unsqueeze(0), # Target for student's audio understanding features
                'separated_sources_target': separated_targets, # List[Dict[str, Tensor]] per batch item
                # 'source_masks': masks # If masks are generated and needed
             }

        except Exception as e:
            logger.error(f"Audio separation distillation failed: {e}", exc_info=True)
            return {}

    def distill_motion_expert(self, student_model: nn.Module, video_frames: Optional[torch.Tensor] = None):
        """Distill optical flow knowledge from RAFT (Placeholder)."""
        if self.raft is None or self.raft.get('model') is None:
             logger.warning("No optical flow model available for distillation.")
             return {}

        if video_frames is None or video_frames.shape[1] < 2:
             logger.warning("Need at least 2 frames for motion distillation.")
             return {}

        logger.info("🎬 Distilling motion knowledge (RAFT)...")
        model_info = self.raft
        model = model_info['model']
        transform = model_info['transform']

        try:
             # Setup optimizer for student motion/temporal components
             motion_params = [
                 p for name, p in student_model.named_parameters()
                 if ('motion' in name.lower() or 'temporal' in name.lower() or 'vision' in name.lower()) and p.requires_grad
             ]
             if not motion_params:
                  logger.warning("No trainable parameters found for motion distillation.")
                  return {}
             optimizer = torch.optim.AdamW(motion_params, lr=1e-5) # Use optimizer from main trainer?

             # Get consecutive frame pairs
             B, T, C, H, W = video_frames.shape
             # Create pairs [f0, f1], [f1, f2], ...
             frame1 = video_frames[:, :-1, :, :, :].reshape(-1, C, H, W)
             frame2 = video_frames[:, 1:, :, :, :].reshape(-1, C, H, W)

             # Teacher predictions - RAFT optical flow
             with torch.no_grad():
                  # Preprocess frames for RAFT
                  # Note: RAFT transform might expect PIL Images or specific tensor format
                  # This needs careful implementation based on the transform
                  # Placeholder: Assume transform takes (C, H, W) tensor
                  frame1_processed = torch.stack([transform(f.cpu()) for f in frame1]).to(self.device)
                  frame2_processed = torch.stack([transform(f.cpu()) for f in frame2]).to(self.device)

                  # Get flow predictions (RAFT output might be a list of flows)
                  flow_predictions = model(frame1_processed, frame2_processed)
                  teacher_flow = flow_predictions[-1] # Final prediction (B*(T-1), 2, H, W)


             # Student predictions
             # Get student's internal representation related to motion/temporal aspects
             # This depends heavily on the student model architecture
             # Example: Use difference between features of consecutive frames
             student_outputs = student_model(video_frames=video_frames) # Process the original sequence
             student_vision_emb = student_outputs.get('temporal_features') # Or 'vision_embeddings', etc.

             if student_vision_emb is None or student_vision_emb.shape[1] < 2:
                  logger.warning("Student model did not produce enough temporal features for motion distillation.")
                  return {}

             # Extract motion features (e.g., difference between consecutive frame features)
             # student_vision_emb might be (B, T, Dim)
             student_motion_proxy = student_vision_emb[:, 1:, :] - student_vision_emb[:, :-1, :] # (B, T-1, Dim)
             student_motion_proxy = student_motion_proxy.reshape(-1, student_motion_proxy.shape[-1]) # (B*(T-1), Dim)


             # Motion distillation loss (align student proxy with teacher flow)
             motion_loss = self._compute_motion_distillation_loss_raft(
                 student_motion_proxy, teacher_flow
             )

             # --- Training Step (should happen in the main trainer loop) ---
             # optimizer.zero_grad()
             # motion_loss.backward()
             # optimizer.step()
             # --- End Training Step ---

             logger.info(f"Calculated motion distillation loss: {motion_loss.item():.4f}")

             # Return the target flow for the main trainer to use
             return {
                 'motion_flow_target': teacher_flow.cpu() # Target for student's motion understanding
             }

        except Exception as e:
            logger.error(f"Motion (RAFT) distillation failed: {e}", exc_info=True)
            return {}


    # --- Helper methods for distillation loss, feature extraction etc. ---
    # Keep these methods, ensure they handle potential None inputs.

    def _extract_spatial_features(self, boxes: torch.Tensor) -> torch.Tensor:
         """Extracts simple spatial features from bounding boxes."""
         if boxes is None or boxes.numel() == 0:
             # Return a zero tensor of expected size, e.g., 4 for center_x, center_y, w, h
             return torch.zeros(4, device=self.device)
         centers = (boxes[:, :2] + boxes[:, 2:]) / 2
         sizes = boxes[:, 2:] - boxes[:, :2]
         # Aggregate features, e.g., mean center and mean size
         spatial_feat = torch.cat([centers.mean(dim=0), sizes.mean(dim=0)], dim=0)
         return spatial_feat

    def _extract_boundary_features_torch(self, segmentation_map: torch.Tensor) -> torch.Tensor:
        """Extract boundary features using torch operations (Sobel)."""
        # segmentation_map shape: (B, C, H, W), assume C=1 for mask
        if segmentation_map.dim() != 4:
            raise ValueError("Input segmentation_map must be 4D (B, C, H, W)")
        if segmentation_map.size(1) != 1:
            segmentation_map = segmentation_map[:, 0:1, :, :] # Take first channel if multiple

        # Sobel kernels
        sobel_x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=segmentation_map.dtype, device=self.device).unsqueeze(0).unsqueeze(0)
        sobel_y_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=segmentation_map.dtype, device=self.device).unsqueeze(0).unsqueeze(0)

        # Apply convolution
        edges_x = F.conv2d(segmentation_map, sobel_x_kernel, padding=1)
        edges_y = F.conv2d(segmentation_map, sobel_y_kernel, padding=1)

        # Calculate gradient magnitude
        boundaries = torch.sqrt(edges_x**2 + edges_y**2)
        # Return flattened boundary map per batch item
        return boundaries.view(boundaries.size(0), -1) # (B, H*W)


    def _process_spatial_detections(self, detections: List[Dict]) -> torch.Tensor:
        """Process spatial detections into knowledge representation (target)."""
        # This method creates the distillation *target* based on teacher output
        spatial_features_list = []
        feature_dim = 0
        for det_per_frame in detections:
             boxes = det_per_frame.get('boxes')
             if boxes is not None and boxes.numel() > 0:
                  feat = self._extract_spatial_features(boxes)
                  spatial_features_list.append(feat)
                  if feature_dim == 0: feature_dim = feat.shape[0] # Get dim from first valid feature
             # else: # Optionally append zeros if no boxes
             #      if feature_dim > 0:
             #           spatial_features_list.append(torch.zeros(feature_dim, device=self.device))


        if spatial_features_list:
            # Aggregate features across frames (e.g., mean)
            aggregated_features = torch.stack(spatial_features_list).mean(dim=0)
            return aggregated_features.unsqueeze(0) # Add batch dimension
        else:
            # Return zero tensor if no detections found in any frame
             return torch.zeros((1, feature_dim if feature_dim > 0 else 4), device=self.device)


    def _process_segmentation_maps(self, segmentations: List[Dict]) -> torch.Tensor:
        """Process segmentation maps into knowledge representation (target)."""
        # This creates the distillation *target*
        boundary_features_list = []
        feature_dim = 0

        for seg_data in segmentations:
             map_tensor = seg_data.get('segmentation_map') # This might be raw output or processed map
             if map_tensor is not None:
                  # Ensure map_tensor is suitable for boundary extraction (e.g., single channel mask)
                  if map_tensor.dim() == 4 and map_tensor.size(1) > 1: # Output logits/probs
                       map_tensor = torch.argmax(map_tensor, dim=1, keepdim=True).float() # Convert to mask
                  elif map_tensor.dim() == 3: # (B, H, W)
                       map_tensor = map_tensor.unsqueeze(1).float() # Add channel dim
                  elif map_tensor.dim() == 2: # (H, W)
                       map_tensor = map_tensor.unsqueeze(0).unsqueeze(0).float() # Add batch and channel

                  boundaries = self._extract_boundary_features_torch(map_tensor) # Expects (B, 1, H, W)
                  boundary_features_list.append(boundaries.squeeze(0)) # Remove batch dim
                  if feature_dim == 0: feature_dim = boundaries.shape[1]

        if boundary_features_list:
            aggregated_boundaries = torch.stack(boundary_features_list).mean(dim=0)
            return aggregated_boundaries.unsqueeze(0) # Add batch dim
        else:
             # Estimate size H*W if possible, otherwise use a placeholder
             # This depends on knowing the expected output size
             placeholder_dim = 224*224 # Example
             return torch.zeros((1, feature_dim if feature_dim > 0 else placeholder_dim), device=self.device)


    # --- (Keep _analyze_tempo_stability, _encode_beat_pattern, etc.) ---
    def _analyze_tempo_stability(self, tempo: float) -> float:
         """Analyzes tempo stability (simple version)."""
         # More complex analysis could look at tempo variations over time
         return 1.0 if 60 <= tempo <= 180 else 0.5 # Simple stability score

    def _encode_beat_pattern(self, beats: List[float]) -> torch.Tensor:
         """Encodes beat pattern into a feature vector (e.g., interval histogram)."""
         if len(beats) < 2:
             return torch.zeros(64, device=self.device)
         intervals = np.diff(beats)
         if len(intervals) == 0:
             return torch.zeros(64, device=self.device)
         # Create histogram of intervals (simple pattern representation)
         hist, _ = np.histogram(intervals, bins=64, range=(0, 2.0)) # Intervals up to 2s
         return torch.tensor(hist, dtype=torch.float32, device=self.device)

    def _calculate_rhythmic_complexity(self, music_analysis: Dict) -> float:
        """Calculates rhythmic complexity."""
        beats = music_analysis.get('beats', [])
        if len(beats) < 4: return 0.0
        intervals = np.diff(beats)
        mean_interval = np.mean(intervals)
        complexity = np.std(intervals) / mean_interval if mean_interval > 0 else 0
        return float(np.clip(complexity, 0, 1))

    def _identify_musical_structure(self, music_analysis: Dict) -> Dict[str, Any]:
        """Identifies basic musical structure (placeholder)."""
        # Placeholder - real implementation needs more advanced analysis
        num_segments = len(music_analysis.get('structure', {}).get('segments', []))
        return {
            'structure_type': 'simple' if num_segments < 5 else 'complex',
            'num_segments': num_segments
        }

    # --- (Keep _evaluate_separation_quality, _analyze_vocal_content, etc.) ---
    def _evaluate_separation_quality(self, sources: Dict[str, np.ndarray]) -> float:
        """Evaluates source separation quality (simple energy entropy)."""
        try:
            source_arrays = [s for s in sources.values() if s is not None and s.size > 0]
            if not source_arrays: return 0.0
            total_energy = sum(np.sum(source**2) for source in source_arrays)
            if total_energy < 1e-10: return 0.0

            energies = [(np.sum(source**2) / total_energy) for source in source_arrays]
            # Entropy calculation
            entropy = -sum(e * np.log(e + 1e-10) for e in energies if e > 0)
            # Normalize entropy (max entropy is log(num_sources))
            max_entropy = np.log(len(source_arrays)) if len(source_arrays) > 0 else 1
            quality = entropy / max_entropy if max_entropy > 0 else 0
            return float(np.clip(quality, 0, 1))
        except Exception as e:
            logger.warning(f"Separation quality evaluation failed: {e}")
            return 0.5 # Default quality

    def _analyze_vocal_content(self, vocals: Optional[np.ndarray]) -> Dict[str, float]:
        """Analyzes vocal content."""
        if vocals is None or vocals.size == 0:
             return {'vocal_energy_ratio': 0.0, 'presence_confidence': 0.0}
        try:
            energy = np.sum(vocals**2)
            # Simple confidence based on relative energy (needs total energy context)
            # Placeholder confidence
            confidence = 1.0 if energy > 1e-3 else 0.0 # Adjust threshold based on normalization
            return {
                'vocal_energy_ratio': float(energy), # Needs normalization later
                'presence_confidence': float(confidence)
            }
        except Exception as e:
            logger.warning(f"Vocal analysis failed: {e}")
            return {'vocal_energy_ratio': 0.0, 'presence_confidence': 0.0}

    def _analyze_instrumental_content(self, instrumental: Optional[np.ndarray]) -> Dict[str, float]:
         """Analyzes instrumental content."""
         # This needs other sources to calculate ratio correctly
         return {'instrumental_energy_ratio': 0.0, 'complexity_score': 0.0} # Placeholder

    def _analyze_audio_balance_simple(self, sources: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Analyzes balance between audio sources."""
        try:
            energies = {name: np.sum(source**2) for name, source in sources.items() if source is not None and source.size > 0}
            total_energy = sum(energies.values())
            if total_energy < 1e-10:
                return {name: 0.0 for name in sources.keys()}

            balance = {name: energy / total_energy for name, energy in energies.items()}
            # Add zeros for sources that were None or empty
            for name in sources.keys():
                 if name not in balance:
                      balance[name] = 0.0
            return balance
        except Exception as e:
            logger.warning(f"Audio balance analysis failed: {e}")
            num_sources = len(sources)
            return {name: 1.0/num_sources if num_sources > 0 else 0.0 for name in sources.keys()} # Default equal balance


    def _create_source_masks(self, sources: Dict) -> Dict[str, np.ndarray]:
         """Creates source separation masks (simple energy-based)."""
         # This requires all source arrays to have the same length
         try:
              valid_sources = {k: v for k, v in sources.items() if v is not None and v.size > 0}
              if not valid_sources: return {}

              # Ensure all arrays have the same length (pad or truncate if necessary)
              max_len = max(s.shape[-1] for s in valid_sources.values())
              aligned_sources = {}
              for name, source in valid_sources.items():
                   if source.shape[-1] < max_len:
                        pad_width = max_len - source.shape[-1]
                        # Pad based on dimensions
                        if source.ndim == 1:
                             padding = (0, pad_width)
                        elif source.ndim == 2:
                             padding = ((0, 0), (0, pad_width))
                        else: continue # Skip if unexpected dimensions
                        aligned_sources[name] = np.pad(source, padding, mode='constant')
                   else:
                        aligned_sources[name] = source[..., :max_len] # Truncate if longer

              # Calculate total absolute magnitude at each time step/channel
              total_abs = sum(np.abs(source) for source in aligned_sources.values())
              total_abs = np.maximum(total_abs, 1e-10) # Avoid division by zero

              masks = {name: np.abs(source) / total_abs for name, source in aligned_sources.items()}
              return masks
         except Exception as e:
              logger.warning(f"Source mask creation failed: {e}")
              # Return uniform masks as fallback
              num_sources = len(sources)
              first_source = next((s for s in sources.values() if s is not None), None)
              if first_source is not None and num_sources > 0:
                   mask_shape = first_source.shape
                   return {name: np.full(mask_shape, 1.0/num_sources) for name in sources.keys()}
              else:
                   return {}


    # --- Main Distillation Entry Point ---
    def distill_all_experts(self, student_model: nn.Module, video_frames: Optional[torch.Tensor] = None, audio_data: Optional[torch.Tensor] = None):
        """
        Enhanced sequential distillation from all expert models including advanced teachers.
        Returns a dictionary containing the *targets* for distillation loss calculation.
        """
        logger.info("🔬 Starting Enhanced Knowledge Distillation (Generating Targets)")

        distillation_targets = {}

        # Distill from advanced teacher models if data is provided
        if video_frames is not None:
            # RT-DETR object detection knowledge targets
            rt_detr_targets = self.distill_rt_detr_knowledge(student_model, video_frames)
            distillation_targets.update(rt_detr_targets)

            # HQ-SAM segmentation knowledge targets
            hq_sam_targets = self.distill_hq_sam_knowledge(student_model, video_frames)
            distillation_targets.update(hq_sam_targets)

            # RAFT motion knowledge targets (if implemented)
            raft_targets = self.distill_motion_expert(student_model, video_frames)
            distillation_targets.update(raft_targets)


        if audio_data is not None:
            # BeatNet music analysis knowledge targets
            beatnet_targets = self.distill_beatnet_knowledge(student_model, audio_data)
            distillation_targets.update(beatnet_targets)

            # Demucs audio separation knowledge targets
            demucs_targets = self.distill_demucs_knowledge(student_model, audio_data)
            distillation_targets.update(demucs_targets)

        # Distill from base ExpertModels (CLIP, Whisper features, etc.)
        # This part depends on how ExpertModels provides features
        try:
             logger.info("Extracting features from base ExpertModels...")
             with torch.no_grad():
                  base_expert_features = self.expert_models.get_all_expert_features(video_frames, audio_data)
                  # Add relevant base features as targets
                  if 'vision_clip_vision_target' in base_expert_features: # Example key
                       distillation_targets['clip_vision_target'] = base_expert_features['vision_clip_vision_target']
                  if 'audio_whisper_features_target' in base_expert_features: # Example key
                       distillation_targets['whisper_features_target'] = base_expert_features['audio_whisper_features_target']
                  # Add other base expert features as needed...

        except Exception as e:
             logger.warning(f"Failed to get features from base ExpertModels: {e}")


        logger.info(f"✅ Generated distillation targets with {len(distillation_targets)} knowledge components")
        # The main trainer will use these targets with student outputs to calculate loss.
        return distillation_targets


    # --- (Other distillation phases like distill_vision_experts, etc. might need adjustment ---
    # --- depending on whether they generate targets or perform training steps) ---
    # --- For now, keep them as placeholders or integrate their logic into distill_all_experts ---

    def _compute_motion_distillation_loss_raft(self, student_motion_proxy, teacher_flow):
         """Computes loss between student motion proxy and RAFT flow."""
         # Teacher flow: (B', 2, H, W) where B' = B*(T-1)
         # Student proxy: (B', Dim)

         B_prime, C_flow, H_flow, W_flow = teacher_flow.shape
         B_prime_student, Dim_student = student_motion_proxy.shape

         if B_prime != B_prime_student:
              logger.warning(f"Batch size mismatch in motion distillation: Teacher {B_prime}, Student {B_prime_student}")
              # Attempt to align if possible (e.g., if one has an extra frame)
              min_B = min(B_prime, B_prime_student)
              teacher_flow = teacher_flow[:min_B]
              student_motion_proxy = student_motion_proxy[:min_B]


         # Convert teacher flow to a feature vector representation
         # Option 1: Global Average Pooling of flow magnitude/angle
         flow_magnitude = torch.sqrt(teacher_flow[:, 0]**2 + teacher_flow[:, 1]**2)
         flow_angle = torch.atan2(teacher_flow[:, 1], teacher_flow[:, 0])
         teacher_motion_vec = torch.stack([
              flow_magnitude.mean(dim=[1, 2]),
              flow_angle.mean(dim=[1, 2]),
              flow_magnitude.std(dim=[1, 2]),
              flow_angle.std(dim=[1, 2])
         ], dim=1) # (B', 4)

         # Option 2: Adaptive Pooling + Linear Layer
         # teacher_pooled = F.adaptive_avg_pool2d(teacher_flow, (1, 1)).view(B_prime, C_flow) # (B', 2)
         # If needed, project teacher_pooled to match student_motion_proxy dimension

         # Let's use Option 1 for simplicity here
         target_dim = teacher_motion_vec.shape[1] # Dimension is 4

         # Project student features to match teacher feature dimension
         if not hasattr(self, 'raft_student_projection'):
              self.raft_student_projection = nn.Linear(Dim_student, target_dim).to(self.device)
         student_motion_projected = self.raft_student_projection(student_motion_proxy)

         # Calculate loss (e.g., MSE)
         motion_loss = F.mse_loss(student_motion_projected, teacher_motion_vec.detach())

         return motion_loss


    # --- (Keep data loader placeholders/implementations) ---
    # These should return iterables yielding batches compatible with distillation methods.
    def _get_vision_distillation_data(self):
         logger.warning("Using synthetic vision data loader for distillation.")
         return self._create_synthetic_vision_data()

    def _get_audio_distillation_data(self):
         logger.warning("Using synthetic audio data loader for distillation.")
         return self._create_synthetic_audio_data()

    def _get_motion_distillation_data(self):
         logger.warning("Using synthetic motion data loader for distillation.")
         return self._create_synthetic_motion_data()

    def _get_multimodal_distillation_data(self):
         logger.warning("Using synthetic multimodal data loader for distillation.")
         return self._create_synthetic_multimodal_data()

    # --- (Keep synthetic data creation methods) ---
    def _create_synthetic_vision_data(self):
        # ... (implementation from original file) ...
        class SyntheticVisionData:
            def __init__(self, num_samples=50, batch_size=4, T=8):
                self.num_samples = num_samples
                self.batch_size = batch_size
                self.T = T # Sequence length

            def __len__(self):
                 return self.num_samples // self.batch_size

            def __iter__(self):
                for i in range(len(self)):
                    frames = torch.randn(self.batch_size, self.T, 3, 224, 224)
                    yield {
                        'images': frames, # Keep 'images' key if used elsewhere
                        'video_frames': frames, # Add 'video_frames' key if expected by student
                        'video_id': [f'synth_vis_{i*self.batch_size + j}' for j in range(self.batch_size)]
                    }
        return SyntheticVisionData()


    def _create_synthetic_audio_data(self):
        # ... (implementation from original file, ensure correct keys) ...
         class SyntheticAudioData:
            def __init__(self, num_samples=50, batch_size=4):
                 self.num_samples = num_samples
                 self.batch_size = batch_size

            def __len__(self):
                 return self.num_samples // self.batch_size

            def __iter__(self):
                 for i in range(len(self)):
                      # Raw audio waveform (B, Channels, Samples) - Use 2 channels for Demucs test
                      # Adjust length as needed (e.g., 5 seconds at 16kHz)
                      audio_len = 16000 * 5
                      audio_raw = torch.randn(self.batch_size, 2, audio_len)
                      yield {
                          'audio_data': audio_raw, # Key for raw audio
                          'audio_id': [f'synth_aud_{i*self.batch_size + j}' for j in range(self.batch_size)]
                          # Add audio_features if your student expects precomputed features
                          # 'audio_features': torch.randn(self.batch_size, 80, 300) # Example mel features
                      }
         return SyntheticAudioData()


    def _create_synthetic_motion_data(self):
        # ... (implementation from original file) ...
         class SyntheticMotionData:
             def __init__(self, num_samples=50, batch_size=2, T=5): # T>=2
                 self.num_samples = num_samples
                 self.batch_size = batch_size
                 self.T = T

             def __len__(self):
                  return self.num_samples // self.batch_size

             def __iter__(self):
                 for i in range(len(self)):
                      # Create video sequence for motion analysis
                      video_frames = torch.randn(self.batch_size, self.T, 3, 224, 224)
                      # Introduce slight motion between frames
                      for t in range(1, self.T):
                           video_frames[:, t] += video_frames[:, t-1] * 0.1 + torch.randn_like(video_frames[:, t]) * 0.05
                      video_frames = torch.clamp(video_frames, -1, 1) # Keep values reasonable

                      yield {
                          'video_frames': video_frames, # Full sequence for student
                          'video_id': [f'synth_mot_{i*self.batch_size + j}' for j in range(self.batch_size)]
                      }
         return SyntheticMotionData()


    def _create_synthetic_multimodal_data(self):
        # ... (implementation from original file) ...
         class SyntheticMultiModalData:
             def __init__(self, num_samples=50, batch_size=2, T=8):
                 self.num_samples = num_samples
                 self.batch_size = batch_size
                 self.T = T

             def __len__(self):
                  return self.num_samples // self.batch_size

             def __iter__(self):
                 for i in range(len(self)):
                      video_frames = torch.randn(self.batch_size, self.T, 3, 224, 224)
                      audio_len = 16000 * 5 # ~5 seconds to match ~8 frames at 1.6fps
                      audio_raw = torch.randn(self.batch_size, 1, audio_len) # Mono audio
                      text_tokens = torch.randint(0, 1000, (self.batch_size, 20)) # B, seq_len

                      yield {
                          'video_frames': video_frames,
                          'audio_data': audio_raw,
                          'text_tokens': text_tokens,
                          'sample_id': [f'synth_mm_{i*self.batch_size + j}' for j in range(self.batch_size)]
                      }
         return SyntheticMultiModalData()


# --- ProgressiveDistillation Class ---
# Keep this class as it was, seems structurally okay.
class ProgressiveDistillation:
    """
    Progressive distillation strategy that gradually transfers knowledge
    """

    def __init__(self, config: DictConfig):
        # Ensure training config structure exists or use defaults
        training_config = config.get('training', {})
        progressive_config = training_config.get('progressive_distillation', {})

        self.config = config
        self.num_stages = progressive_config.get('num_stages', 4)
        self.epochs_per_stage = progressive_config.get('epochs_per_stage', 5) # Changed from warmup_epochs
        self.current_stage = 0

        # Define weights per stage - more granular control
        self.stage_weights = [
            # Stage 0: Focus on basic features
            {'feature_matching': 1.0, 'attention_matching': 0.1, 'output_matching': 0.0, 'cross_modal_matching': 0.0, 'advanced_teachers': 0.1},
            # Stage 1: Add attention and more advanced features
            {'feature_matching': 0.8, 'attention_matching': 0.5, 'output_matching': 0.1, 'cross_modal_matching': 0.1, 'advanced_teachers': 0.3},
            # Stage 2: Increase output matching and cross-modal
            {'feature_matching': 0.5, 'attention_matching': 0.3, 'output_matching': 0.7, 'cross_modal_matching': 0.5, 'advanced_teachers': 0.6},
            # Stage 3: Focus on outputs, cross-modal, and advanced teachers
            {'feature_matching': 0.2, 'attention_matching': 0.1, 'output_matching': 1.0, 'cross_modal_matching': 1.0, 'advanced_teachers': 1.0},
        ]
        # Ensure stage_weights covers num_stages
        if len(self.stage_weights) < self.num_stages:
             self.stage_weights.extend([self.stage_weights[-1]] * (self.num_stages - len(self.stage_weights)))


    def get_distillation_weights(self, epoch: int) -> Dict[str, float]:
        """Get distillation weights for current training stage"""

        # Determine current stage based on epoch
        self.current_stage = min(epoch // self.epochs_per_stage, self.num_stages - 1)

        weights = self.stage_weights[self.current_stage]

        logger.debug(f"Epoch {epoch}, Stage {self.current_stage} distillation weights: {weights}")
        return weights

    def should_update_stage(self, epoch: int, loss_history: List[float]) -> bool:
        """Determine if we should move to next distillation stage"""

        # Simple epoch-based stage update
        new_stage = min(epoch // self.epochs_per_stage, self.num_stages - 1)
        if new_stage > self.current_stage:
             self.current_stage = new_stage
             logger.info(f"📈 Progressing to distillation stage {self.current_stage} at epoch {epoch}")
             return True

        # Optional: Add loss-based plateau detection
        # if len(loss_history) >= 10: # Need enough history
        #     recent_losses = loss_history[-5:]
        #     past_losses = loss_history[-10:-5]
        #     if np.mean(recent_losses) > np.mean(past_losses) * 0.98: # Loss not improving much
        #         if self.current_stage < self.num_stages - 1:
        #             self.current_stage += 1
        #             logger.info(f"📈 Loss plateaued, progressing to stage {self.current_stage} at epoch {epoch}")
        #             return True

        return False