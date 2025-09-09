"""
Audio Processor - Advanced audio understanding with Whisper and librosa
"""

import os
import torch
import numpy as np
import librosa
import soundfile as sf
import logging
import ffmpeg
from typing import Dict, List, Any, Optional, Tuple
from omegaconf import DictConfig
from transformers import WhisperProcessor, WhisperModel

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Advanced audio processor with Whisper integration and feature extraction"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Initialize Whisper model
        self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        self.whisper_model = WhisperModel.from_pretrained("openai/whisper-base")
        self.whisper_model.to(self.device)
        
        # Audio processing settings
        self.sample_rate = config.get('audio_sample_rate', 16000)
        self.n_fft = config.get('n_fft', 2048)
        self.hop_length = config.get('hop_length', 512)
        self.n_mels = config.get('n_mels', 128)
        
        logger.info(f"AudioProcessor initialized on {self.device}")
    
    def load_audio(self, video_path: str) -> Dict[str, Any]:
        """Load and preprocess audio from video file - FIXED to return actual data"""
        try:
            # Check if file exists
            if not os.path.exists(video_path):
                logger.error(f"Video file not found: {video_path}")
                return self._get_empty_audio_data()
            
            # Extract audio using ffmpeg
            audio_path = self._extract_audio(video_path)
            
            # Load audio with librosa
            try:
                audio, sr = librosa.load(audio_path, sr=self.sample_rate)
                duration = len(audio) / sr
            except Exception as e:
                logger.error(f"Failed to load audio with librosa: {e}")
                # Create silence as fallback
                audio = np.zeros(self.sample_rate)  # 1 second silence
                sr = self.sample_rate
                duration = 1.0
            
            # Extract various audio features
            features = self._extract_audio_features(audio, sr)
            
            # Get transcription
            transcription = self.transcribe_audio(audio)
            
            # Audio content analysis
            content_analysis = self.analyze_audio_content(features)
            
            # Event detection
            events = self.detect_audio_events(features)
            
            return {
                'audio': torch.from_numpy(audio).float(),  # ACTUAL AUDIO DATA
                'features': features,  # ACTUAL FEATURE DICT
                'sample_rate': sr,
                'duration': duration,
                'transcription': transcription,
                'content_analysis': content_analysis,
                'events': events,
                'audio_path': audio_path
            }
            
        except Exception as e:
            logger.error(f"Error loading audio from {video_path}: {e}")
            return self._get_empty_audio_data()
    
    def _get_empty_audio_data(self) -> Dict[str, Any]:
        """Return empty audio data structure"""
        return {
            'audio': torch.zeros(self.sample_rate),  # 1 second silence
            'features': self._get_empty_features(),
            'sample_rate': self.sample_rate,
            'duration': 0.0,
            'transcription': {'text': '', 'confidence': 0.0, 'language': 'unknown'},
            'content_analysis': {},
            'events': [],
            'audio_path': None
        }
    
    def _extract_audio(self, video_path: str) -> str:
        """Extract audio from video using ffmpeg"""
        audio_path = video_path.replace('.mp4', '_audio.wav').replace('.avi', '_audio.wav')
        
        try:
            (
                ffmpeg
                .input(video_path)
                .output(audio_path, acodec='pcm_s16le', ar=self.sample_rate)
                .overwrite_output()
                .run(quiet=True)
            )
        except Exception as e:
            logger.warning(f"FFmpeg audio extraction failed: {e}")
            # Fallback: create silent audio
            silence = np.zeros(self.sample_rate)  # 1 second of silence
            sf.write(audio_path, silence, self.sample_rate)
        
        return audio_path
    
    def _extract_audio_features(self, audio: np.ndarray, sr: int) -> Dict[str, torch.Tensor]:
        """Extract comprehensive audio features"""
        features = {}
        
        try:
            # Mel-frequency cepstral coefficients (MFCCs)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            features['mfccs'] = torch.from_numpy(mfccs).float()
            
            # Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_fft=self.n_fft, 
                hop_length=self.hop_length, n_mels=self.n_mels
            )
            features['mel_spectrogram'] = torch.from_numpy(mel_spec).float()
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            features['chroma'] = torch.from_numpy(chroma).float()
            
            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            features['spectral_centroid'] = torch.from_numpy(spectral_centroids).float()
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features['zero_crossing_rate'] = torch.from_numpy(zcr).float()
            
            # RMS energy
            rms = librosa.feature.rms(y=audio)[0]
            features['rms_energy'] = torch.from_numpy(rms).float()
            
            # Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            features['spectral_rolloff'] = torch.from_numpy(rolloff).float()
            
            # Tempo and beat tracking
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
            features['tempo'] = torch.tensor(tempo).float()
            features['beats'] = torch.from_numpy(beats).float()
            
        except Exception as e:
            logger.warning(f"Error extracting audio features: {e}")
            features = self._get_empty_features()
        
        return features
    
    def _get_empty_features(self) -> Dict[str, torch.Tensor]:
        """Return empty feature dict for error cases"""
        return {
            'mfccs': torch.zeros(13, 1),
            'mel_spectrogram': torch.zeros(self.n_mels, 1),
            'chroma': torch.zeros(12, 1),
            'spectral_centroid': torch.zeros(1),
            'zero_crossing_rate': torch.zeros(1),
            'rms_energy': torch.zeros(1),
            'spectral_rolloff': torch.zeros(1),
            'tempo': torch.tensor(120.0),
            'beats': torch.zeros(1)
        }
    
    def transcribe_audio(self, audio: np.ndarray) -> Dict[str, Any]:
        """Transcribe audio using Whisper"""
        try:
            # Prepare input for Whisper
            inputs = self.whisper_processor(audio, return_tensors="pt", sampling_rate=self.sample_rate)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate transcription
            # Generate transcription with confidence scores
            with torch.no_grad():
                predicted_ids = self.whisper_model.generate(
                    **inputs,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                transcription = self.whisper_processor.batch_decode(
                    predicted_ids.sequences, skip_special_tokens=True
                )[0]
                
                # Calculate confidence from generation scores
                if hasattr(predicted_ids, 'scores') and predicted_ids.scores:
                    # Average confidence across tokens
                    scores = torch.stack(predicted_ids.scores)
                    probs = torch.softmax(scores, dim=-1)
                    max_probs = torch.max(probs, dim=-1)[0]
                    confidence = torch.mean(max_probs).item()
                else:
                    confidence = 0.8  # Default confidence
                
                # Detect language from Whisper model
                try:
                    # Use Whisper's built-in language detection
                    language_tokens = predicted_ids.sequences[0][:4]  # First few tokens often contain language info
                    language = self.whisper_processor.tokenizer.decode(language_tokens)
                    
                    # Parse common language tokens
                    if '<|en|>' in language or 'english' in transcription.lower()[:50]:
                        detected_language = 'en'
                    elif '<|es|>' in language or any(word in transcription.lower() for word in ['el', 'la', 'de', 'con']):
                        detected_language = 'es'
                    elif '<|fr|>' in language or any(word in transcription.lower() for word in ['le', 'la', 'de', 'avec']):
                        detected_language = 'fr'
                    else:
                        detected_language = 'en'  # Default to English
                        
                except Exception as e:
                    logger.debug(f"Language detection failed: {e}")
                    detected_language = 'en'
            
            return {
                'text': transcription,
                'confidence': confidence,
                'language': detected_language
            }
            
        except Exception as e:
            logger.warning(f"Error transcribing audio: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'language': 'unknown'
            }
    
    def analyze_audio_content(self, audio_features: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Analyze audio content for editing insights"""
        try:
            analysis = {}
            
            # Music vs speech classification (simple heuristic)
            if 'mfccs' in audio_features:
                mfcc_var = audio_features['mfccs'].var(dim=1).mean()
                analysis['is_music'] = mfcc_var > 0.1  # Simple threshold
            
            # Energy analysis
            if 'rms_energy' in audio_features:
                rms = audio_features['rms_energy']
                analysis['avg_energy'] = rms.mean().item()
                analysis['energy_variation'] = rms.std().item()
                
                # Detect silence/quiet sections
                silence_threshold = rms.mean() * 0.1
                analysis['silence_ratio'] = (rms < silence_threshold).float().mean().item()
            
            # Rhythm analysis
            if 'tempo' in audio_features:
                analysis['tempo'] = audio_features['tempo'].item()
                analysis['rhythm_strength'] = self._calculate_rhythm_strength(audio_features)
            
            # Spectral analysis
            if 'spectral_centroid' in audio_features:
                centroid = audio_features['spectral_centroid']
                analysis['brightness'] = centroid.mean().item()
                analysis['spectral_variation'] = centroid.std().item()
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing audio content: {e}")
            return {}
    
    def _calculate_rhythm_strength(self, features: Dict[str, torch.Tensor]) -> float:
        """Calculate rhythm strength from audio features"""
        try:
            if 'beats' in features and len(features['beats']) > 1:
                # Calculate beat consistency
                beat_intervals = torch.diff(features['beats'])
                rhythm_strength = 1.0 / (1.0 + beat_intervals.std().item())
                return min(rhythm_strength, 1.0)
            return 0.0
        except:
            return 0.0
    
    def detect_audio_events(self, audio_features: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
        """Detect audio events like music starts, speech, applause, etc."""
        events = []
        
        try:
            if 'rms_energy' in audio_features:
                rms = audio_features['rms_energy']
                
                # Simple event detection based on energy changes
                threshold = rms.mean() + rms.std()
                high_energy_frames = torch.where(rms > threshold)[0]
                
                if len(high_energy_frames) > 0:
                    events.append({
                        'type': 'high_energy',
                        'start_frame': high_energy_frames[0].item(),
                        'end_frame': high_energy_frames[-1].item(),
                        'confidence': 0.7
                    })
                
                # Detect silence
                silence_threshold = rms.mean() * 0.1
                silence_frames = torch.where(rms < silence_threshold)[0]
                
                if len(silence_frames) > 10:  # More than 10 frames of silence
                    events.append({
                        'type': 'silence',
                        'start_frame': silence_frames[0].item(),
                        'end_frame': silence_frames[-1].item(),
                        'confidence': 0.9
                    })
            
        except Exception as e:
            logger.warning(f"Error detecting audio events: {e}")
        
        return events
