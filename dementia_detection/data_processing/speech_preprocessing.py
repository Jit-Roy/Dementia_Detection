"""
Speech Preprocessing Module
Contains functions for audio preprocessing including noise reduction, 
silence trimming, normalization, and speaker diarization.
"""

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt
from scipy.ndimage import uniform_filter1d
from pydub import AudioSegment
from pydub.silence import split_on_silence
from typing import Tuple, List, Optional
from pathlib import Path
from ..config.settings import AUDIO_CONFIG


class AudioPreprocessor:
    """Main class for audio preprocessing operations."""
    
    def __init__(self, sample_rate: int = AUDIO_CONFIG["sample_rate"]):
        self.sample_rate = sample_rate
        
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            return audio, sr
        except Exception as e:
            raise
    
    def normalize_audio(self, audio: np.ndarray, method: str = "rms") -> np.ndarray:
        if len(audio) == 0:
            return audio
            
        if method == "peak":
            # Peak normalization
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak
        elif method == "rms":
            # RMS normalization
            rms = np.sqrt(np.mean(audio**2))
            if rms > 0:
                audio = audio / rms * 0.1  # Scale to reasonable level
        elif method == "lufs":
            # Simple LUFS-like normalization
            audio = librosa.util.normalize(audio)
            
        return np.clip(audio, -1.0, 1.0)
    
    def reduce_noise(self, audio: np.ndarray, sr: int, 
                    strength: float = AUDIO_CONFIG["noise_reduction_strength"]) -> np.ndarray:
        # Compute STFT
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise spectrum from first 0.5 seconds
        noise_frames = int(0.5 * sr / 512)  # 512 is default hop_length
        noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
        
        # Spectral subtraction
        alpha = strength * 2.0  # Oversubtraction factor
        beta = 0.001  # Spectral floor
        
        subtracted = magnitude - alpha * noise_spectrum
        subtracted = np.maximum(subtracted, beta * magnitude)
        
        # Reconstruct audio
        enhanced_stft = subtracted * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft)
        
        return enhanced_audio
    
    def apply_bandpass_filter(self, audio: np.ndarray, sr: int, 
                             lowcut: float = 80, highcut: float = 8000, 
                             order: int = 4) -> np.ndarray:
        nyquist = sr / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        
        b, a = butter(order, [low, high], btype='band')
        filtered_audio = filtfilt(b, a, audio)
        
        return filtered_audio
    
    def detect_voice_activity(self, audio: np.ndarray, sr: int, 
                             frame_duration: float = 0.03) -> List[bool]:
        try:
            # Use RMS energy for voice activity detection
            hop_length = int(sr * frame_duration)
            
            # Compute RMS energy
            rms = librosa.feature.rms(
                y=audio, 
                hop_length=hop_length
            )[0]
            
            # Compute threshold based on audio statistics
            rms_mean = np.mean(rms)
            rms_std = np.std(rms)
            threshold = rms_mean + 0.3 * rms_std  # Adjustable threshold
            
            # Voice activity decision
            vad_results = (rms > threshold).tolist()
            
            return vad_results
            
        except Exception as e:
            # Fallback: assume all frames are speech
            num_frames = int(np.ceil(len(audio) / (sr * frame_duration)))
            return [True] * num_frames
    
    def remove_silence(self, audio: np.ndarray, sr: int, 
                      threshold: float = AUDIO_CONFIG["silence_threshold"]) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        # Convert to AudioSegment for silence detection
        audio_int16 = (audio * 32767).astype(np.int16)
        audio_segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=1
        )
        
        # Split on silence
        silence_thresh = -40  # dB
        min_silence_len = 500  # ms
        
        speech_chunks = split_on_silence(
            audio_segment,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=100  # Keep 100ms of silence at edges
        )
        
        if not speech_chunks:
            return audio, [(0.0, len(audio) / sr)]
        
        # Reconstruct audio and timestamps
        trimmed_audio = []
        timestamps = []
        current_time = 0.0
        
        for chunk in speech_chunks:
            chunk_array = np.array(chunk.get_array_of_samples(), dtype=np.float32) / 32767.0
            chunk_duration = len(chunk_array) / sr
            
            trimmed_audio.append(chunk_array)
            timestamps.append((current_time, current_time + chunk_duration))
            current_time += chunk_duration
        
        if trimmed_audio:
            final_audio = np.concatenate(trimmed_audio)
        else:
            final_audio = audio
            timestamps = [(0.0, len(audio) / sr)]
        
        return final_audio, timestamps
    
    def segment_speakers(self, audio: np.ndarray, sr: int) -> List[Tuple[float, float, int]]:
        # Compute energy
        hop_length = 512
        frame_length = 2048
        
        energy = librosa.feature.rms(
            y=audio, 
            frame_length=frame_length, 
            hop_length=hop_length
        )[0]
        
        # Convert frame indices to time
        times = librosa.frames_to_time(
            np.arange(len(energy)), 
            sr=sr, 
            hop_length=hop_length
        )
        
        # Simple segmentation based on energy changes
        # This is a basic implementation - for production use pyannote.audio
        
        # Smooth energy signal
        energy_smooth = uniform_filter1d(energy, size=10)
        
        # Detect significant energy changes
        energy_diff = np.diff(energy_smooth)
        threshold = np.std(energy_diff) * 1.5
        
        change_points = np.where(np.abs(energy_diff) > threshold)[0]
        
        # Create segments
        segments = []
        start_idx = 0
        speaker_id = 0
        
        for change_idx in change_points:
            if change_idx > start_idx:
                start_time = times[start_idx]
                end_time = times[change_idx]
                segments.append((start_time, end_time, speaker_id))
                
                start_idx = change_idx
                speaker_id = 1 - speaker_id  # Alternate between speakers
        
        # Add final segment
        if start_idx < len(times):
            start_time = times[start_idx]
            end_time = times[-1] + (times[1] - times[0])  # Add frame duration
            segments.append((start_time, end_time, speaker_id))
        
        return segments
    
    def preprocess_audio(self, file_path: str, 
                        apply_noise_reduction: bool = True,
                        apply_normalization: bool = True,
                        remove_silence: bool = True) -> dict:
        try:
            # Load audio
            audio, sr = self.load_audio(file_path)
            original_duration = len(audio) / sr
            
            # Apply bandpass filter
            audio = self.apply_bandpass_filter(audio, sr)
            
            # Noise reduction
            if apply_noise_reduction:
                audio = self.reduce_noise(audio, sr)
            
            # Normalization
            if apply_normalization:
                audio = self.normalize_audio(audio)
            
            # Voice activity detection
            vad_results = self.detect_voice_activity(audio, sr)
            speech_ratio = sum(vad_results) / len(vad_results) if vad_results else 0
            
            # Remove silence
            speech_segments = []
            if remove_silence:
                audio, speech_segments = self.remove_silence(audio, sr)
            
            # Speaker segmentation
            speaker_segments = self.segment_speakers(audio, sr)
            
            processed_duration = len(audio) / sr
            
            result = {
                "audio": audio,
                "sample_rate": sr,
                "original_duration": original_duration,
                "processed_duration": processed_duration,
                "speech_ratio": speech_ratio,
                "speech_segments": speech_segments,
                "speaker_segments": speaker_segments,
                "vad_results": vad_results
            }
            return result
            
        except Exception as e:
            raise


def save_processed_audio(audio: np.ndarray, sr: int, output_path: str) -> None:
    """
    Save processed audio to file.
    
    Args:
        audio: Processed audio array
        sr: Sample rate
        output_path: Output file path
    """
    try:
        sf.write(output_path, audio, sr)
    except Exception as e:
        raise
