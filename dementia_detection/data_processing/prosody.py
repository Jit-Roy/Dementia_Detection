"""
Advanced Prosody Analysis Module
Extracts detailed prosodic features including rhythm, intonation patterns,
phonation ratios, and speech dynamics for dementia detection.
"""

import librosa
import numpy as np
import scipy.stats
from scipy.signal import find_peaks, butter, filtfilt
from scipy.ndimage import uniform_filter1d
from typing import Dict, List, Tuple, Optional





class AdvancedProsodyAnalyzer:
    """Extract comprehensive prosodic features for dementia detection."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
    def extract_phonation_features(self, audio: np.ndarray, 
                                 voiced_segments: Optional[List[Tuple[float, float]]] = None) -> Dict[str, float]:
        """
        Extract phonation ratio and voicing characteristics.
        
        Args:
            audio: Input audio array
            voiced_segments: List of (start, end) time pairs for voiced segments
            
        Returns:
            Dictionary of phonation features
        """
        try:
            features = {}
            
            if voiced_segments is None:
                voiced_segments = self._detect_voiced_segments(audio)
            
            total_duration = len(audio) / self.sample_rate
            total_voiced_time = sum(end - start for start, end in voiced_segments)
            total_unvoiced_time = total_duration - total_voiced_time
            
            # Phonation ratio - key dementia indicator
            features["phonation_ratio"] = total_voiced_time / total_duration if total_duration > 0 else 0
            features["voicing_ratio"] = features["phonation_ratio"]  # Alternative name
            
            # Voicing characteristics
            features["total_voiced_time"] = total_voiced_time
            features["total_unvoiced_time"] = total_unvoiced_time
            features["voiced_segments_count"] = len(voiced_segments)
            features["avg_voiced_segment_duration"] = total_voiced_time / len(voiced_segments) if len(voiced_segments) > 0 else 0
            
            # Voiced segment statistics
            if len(voiced_segments) > 0:
                segment_durations = [end - start for start, end in voiced_segments]
                features["voiced_segment_duration_std"] = np.std(segment_durations)
                features["voiced_segment_duration_median"] = np.median(segment_durations)
                features["voiced_segment_duration_range"] = np.max(segment_durations) - np.min(segment_durations)
                
                # Voicing continuity - longer segments indicate better control
                features["voicing_continuity"] = np.mean(segment_durations)
                features["voicing_stability"] = 1.0 / (1.0 + np.std(segment_durations))
            else:
                features.update({
                    "voiced_segment_duration_std": 0, "voiced_segment_duration_median": 0,
                    "voiced_segment_duration_range": 0, "voicing_continuity": 0, "voicing_stability": 0
                })
                
            return features
            
        except Exception as e:
            
            return self._empty_phonation_features()
    
    def extract_pause_distribution_features(self, audio: np.ndarray, 
                                          speech_segments: Optional[List[Tuple[float, float]]] = None) -> Dict[str, float]:
        """
        Extract detailed pause distribution and timing features.
        
        Args:
            audio: Input audio array
            speech_segments: List of (start, end) time pairs for speech segments
            
        Returns:
            Dictionary of pause distribution features
        """
        try:
            features = {}
            
            if speech_segments is None:
                speech_segments = self._detect_speech_segments(audio)
            
            # Calculate pause durations
            pause_durations = []
            if len(speech_segments) > 1:
                for i in range(len(speech_segments) - 1):
                    pause_start = speech_segments[i][1]
                    pause_end = speech_segments[i + 1][0]
                    if pause_end > pause_start:
                        pause_durations.append(pause_end - pause_start)
            
            # Basic pause statistics
            features["num_pauses"] = len(pause_durations)
            features["total_pause_time"] = sum(pause_durations)
            features["pause_rate"] = len(pause_durations) / (len(audio) / self.sample_rate) if len(audio) > 0 else 0
            
            if len(pause_durations) > 0:
                # Central tendency measures
                features["mean_pause_duration"] = np.mean(pause_durations)
                features["median_pause_duration"] = np.median(pause_durations)
                features["pause_duration_std"] = np.std(pause_durations)
                
                # Distribution characteristics
                features["pause_duration_variance"] = np.var(pause_durations)
                features["pause_duration_skewness"] = scipy.stats.skew(pause_durations)
                features["pause_duration_kurtosis"] = scipy.stats.kurtosis(pause_durations)
                features["pause_duration_range"] = np.max(pause_durations) - np.min(pause_durations)
                features["pause_duration_iqr"] = np.percentile(pause_durations, 75) - np.percentile(pause_durations, 25)
                
                # Percentile measures
                features["pause_duration_q25"] = np.percentile(pause_durations, 25)
                features["pause_duration_q75"] = np.percentile(pause_durations, 75)
                features["pause_duration_q90"] = np.percentile(pause_durations, 90)
                
                # Clinical indicators
                features["long_pause_ratio"] = sum(1 for p in pause_durations if p > 1.0) / len(pause_durations)  # >1sec pauses
                features["short_pause_ratio"] = sum(1 for p in pause_durations if p < 0.2) / len(pause_durations)  # <0.2sec pauses
                features["pause_irregularity"] = np.std(pause_durations) / np.mean(pause_durations)  # CV
                
            else:
                # No pauses detected
                features.update({
                    "mean_pause_duration": 0, "median_pause_duration": 0, "pause_duration_std": 0,
                    "pause_duration_variance": 0, "pause_duration_skewness": 0, "pause_duration_kurtosis": 0,
                    "pause_duration_range": 0, "pause_duration_iqr": 0, "pause_duration_q25": 0,
                    "pause_duration_q75": 0, "pause_duration_q90": 0, "long_pause_ratio": 0,
                    "short_pause_ratio": 0, "pause_irregularity": 0
                })
            
            return features
            
        except Exception as e:
            
            return self._empty_pause_features()
    
    def extract_speech_rhythm_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract speech rhythm and temporal regularity features.
        
        Args:
            audio: Input audio array
            
        Returns:
            Dictionary of rhythm features
        """
        try:
            features = {}
            
            # Detect syllabic nuclei (peaks in energy envelope)
            syllable_nuclei = self._detect_syllable_nuclei(audio)
            
            if len(syllable_nuclei) > 1:
                # Inter-syllabic intervals
                syllable_intervals = np.diff(syllable_nuclei) / self.sample_rate
                
                # Rhythm regularity measures
                features["syllable_rate"] = len(syllable_nuclei) / (len(audio) / self.sample_rate)
                features["mean_syllable_interval"] = np.mean(syllable_intervals)
                features["syllable_interval_std"] = np.std(syllable_intervals)
                features["syllable_interval_cv"] = np.std(syllable_intervals) / np.mean(syllable_intervals)
                
                # Rhythm variability (key dementia indicator)
                features["rhythm_variability"] = np.std(syllable_intervals)
                features["rhythm_irregularity"] = features["syllable_interval_cv"]
                
                # Temporal dynamics
                features["syllable_interval_range"] = np.max(syllable_intervals) - np.min(syllable_intervals)
                features["rhythm_entropy"] = scipy.stats.entropy(np.histogram(syllable_intervals, bins=10)[0] + 1)
                
                # Rhythmic consistency measures
                normalized_intervals = syllable_intervals / np.mean(syllable_intervals)
                features["rhythm_consistency"] = 1.0 / (1.0 + np.std(normalized_intervals))
                features["isochrony_measure"] = np.exp(-np.std(normalized_intervals))  # Closer to 1 = more regular
                
            else:
                features.update({
                    "syllable_rate": 0, "mean_syllable_interval": 0, "syllable_interval_std": 0,
                    "syllable_interval_cv": 0, "rhythm_variability": 0, "rhythm_irregularity": 0,
                    "syllable_interval_range": 0, "rhythm_entropy": 0, "rhythm_consistency": 0,
                    "isochrony_measure": 0
                })
            
            # Word-level rhythm (if detectable)
            word_boundaries = self._estimate_word_boundaries(audio)
            if len(word_boundaries) > 1:
                word_intervals = np.diff(word_boundaries) / self.sample_rate
                features["word_rate"] = len(word_boundaries) / (len(audio) / self.sample_rate)
                features["word_interval_variability"] = np.std(word_intervals) / np.mean(word_intervals) if np.mean(word_intervals) > 0 else 0
            else:
                features["word_rate"] = 0
                features["word_interval_variability"] = 0
                
            return features
            
        except Exception as e:
            
            return self._empty_rhythm_features()
    
    def extract_energy_dynamics_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract energy dynamics and loudness variation features.
        
        Args:
            audio: Input audio array
            
        Returns:
            Dictionary of energy dynamics features
        """
        try:
            features = {}
            
            # RMS energy envelope
            hop_length = 512
            rms_energy = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
            
            # Basic energy statistics
            features["rms_mean"] = np.mean(rms_energy)
            features["rms_std"] = np.std(rms_energy)
            features["rms_max"] = np.max(rms_energy)
            features["rms_min"] = np.min(rms_energy)
            features["rms_range"] = features["rms_max"] - features["rms_min"]
            
            # Energy dynamics (key for dementia detection)
            features["energy_variance"] = np.var(rms_energy)
            features["energy_dynamic_range"] = 20 * np.log10(features["rms_max"] / (features["rms_min"] + 1e-10))
            features["energy_variability"] = np.std(rms_energy) / np.mean(rms_energy) if np.mean(rms_energy) > 0 else 0
            
            # Energy contour analysis
            energy_diff = np.diff(rms_energy)
            features["energy_contour_slope"] = np.mean(energy_diff)
            features["energy_contour_variability"] = np.std(energy_diff)
            
            # Energy distribution characteristics
            features["energy_skewness"] = scipy.stats.skew(rms_energy)
            features["energy_kurtosis"] = scipy.stats.kurtosis(rms_energy)
            
            # Percentile measures
            features["energy_q25"] = np.percentile(rms_energy, 25)
            features["energy_q75"] = np.percentile(rms_energy, 75)
            features["energy_iqr"] = features["energy_q75"] - features["energy_q25"]
            
            # Loudness stability (clinical indicator)
            features["loudness_stability"] = 1.0 / (1.0 + features["energy_variability"])
            features["energy_consistency"] = np.exp(-features["energy_variability"])
            
            # Spectral energy distribution
            stft = librosa.stft(audio, hop_length=hop_length)
            magnitude = np.abs(stft)
            spectral_energy = np.sum(magnitude, axis=0)
            
            features["spectral_energy_mean"] = np.mean(spectral_energy)
            features["spectral_energy_std"] = np.std(spectral_energy)
            features["spectral_energy_variability"] = np.std(spectral_energy) / np.mean(spectral_energy) if np.mean(spectral_energy) > 0 else 0
            
            return features
            
        except Exception as e:
            
            return self._empty_energy_features()
    
    def _detect_voiced_segments(self, audio: np.ndarray) -> List[Tuple[float, float]]:
        """Detect voiced segments using energy and spectral features."""
        try:
            # Use RMS energy and spectral centroid for voicing detection
            hop_length = 512
            rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate, hop_length=hop_length)[0]
            
            # Voicing criteria: sufficient energy + appropriate spectral characteristics
            energy_threshold = np.mean(rms) + 0.1 * np.std(rms)
            centroid_threshold = np.mean(spectral_centroid) - 0.2 * np.std(spectral_centroid)
            
            voiced_frames = (rms > energy_threshold) & (spectral_centroid > centroid_threshold)
            
            # Convert to time segments
            times = librosa.frames_to_time(np.arange(len(voiced_frames)), sr=self.sample_rate, hop_length=hop_length)
            segments = []
            
            in_segment = False
            start_time = 0
            
            for i, (time, is_voiced) in enumerate(zip(times, voiced_frames)):
                if is_voiced and not in_segment:
                    start_time = time
                    in_segment = True
                elif not is_voiced and in_segment:
                    segments.append((start_time, time))
                    in_segment = False
            
            if in_segment:
                segments.append((start_time, times[-1]))
            
            # Filter short segments
            min_segment_duration = 0.1  # 100ms minimum
            segments = [(start, end) for start, end in segments if end - start >= min_segment_duration]
            
            return segments
            
        except Exception as e:
            
            return [(0.0, len(audio) / self.sample_rate)]
    
    def _detect_speech_segments(self, audio: np.ndarray) -> List[Tuple[float, float]]:
        """Detect speech segments using energy-based VAD."""
        try:
            hop_length = 512
            rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
            
            # Speech detection threshold
            threshold = np.mean(rms) + 0.2 * np.std(rms)
            speech_frames = rms > threshold
            
            # Convert to time segments
            times = librosa.frames_to_time(np.arange(len(speech_frames)), sr=self.sample_rate, hop_length=hop_length)
            segments = []
            
            in_segment = False
            start_time = 0
            
            for i, (time, is_speech) in enumerate(zip(times, speech_frames)):
                if is_speech and not in_segment:
                    start_time = time
                    in_segment = True
                elif not is_speech and in_segment:
                    segments.append((start_time, time))
                    in_segment = False
            
            if in_segment:
                segments.append((start_time, times[-1]))
            
            return segments
            
        except Exception as e:
            
            return [(0.0, len(audio) / self.sample_rate)]
    
    def _detect_syllable_nuclei(self, audio: np.ndarray) -> np.ndarray:
        """Detect syllabic nuclei based on energy peaks."""
        try:
            # Smooth energy envelope
            rms = librosa.feature.rms(y=audio, hop_length=256)[0]
            smoothed_rms = uniform_filter1d(rms, size=5)
            
            # Find peaks in energy
            peak_indices, _ = find_peaks(smoothed_rms, 
                                       height=np.mean(smoothed_rms),
                                       distance=int(0.1 * self.sample_rate / 256))  # Min 100ms apart
            
            # Convert to sample indices
            syllable_nuclei = librosa.frames_to_samples(peak_indices, hop_length=256)
            return syllable_nuclei
            
        except Exception as e:
            
            return np.array([])
    
    def _estimate_word_boundaries(self, audio: np.ndarray) -> np.ndarray:
        """Estimate word boundaries using energy and spectral change detection."""
        try:
            # Use spectral centroid changes and energy dips
            hop_length = 512
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate, hop_length=hop_length)[0]
            rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
            
            # Detect significant changes
            centroid_diff = np.abs(np.diff(spectral_centroid))
            rms_diff = np.abs(np.diff(rms))
            
            # Combined change measure
            change_measure = centroid_diff + rms_diff
            
            # Find boundaries at significant changes
            boundary_threshold = np.mean(change_measure) + np.std(change_measure)
            boundary_frames = np.where(change_measure > boundary_threshold)[0]
            
            # Convert to sample indices
            word_boundaries = librosa.frames_to_samples(boundary_frames, hop_length=hop_length)
            return word_boundaries
            
        except Exception as e:
            
            return np.array([])
    
    def _empty_phonation_features(self) -> Dict[str, float]:
        """Return empty phonation features dictionary."""
        return {
            "phonation_ratio": 0, "voicing_ratio": 0, "total_voiced_time": 0,
            "total_unvoiced_time": 0, "voiced_segments_count": 0, "avg_voiced_segment_duration": 0,
            "voiced_segment_duration_std": 0, "voiced_segment_duration_median": 0,
            "voiced_segment_duration_range": 0, "voicing_continuity": 0, "voicing_stability": 0
        }
    
    def _empty_pause_features(self) -> Dict[str, float]:
        """Return empty pause features dictionary."""
        return {
            "num_pauses": 0, "total_pause_time": 0, "pause_rate": 0, "mean_pause_duration": 0,
            "median_pause_duration": 0, "pause_duration_std": 0, "pause_duration_variance": 0,
            "pause_duration_skewness": 0, "pause_duration_kurtosis": 0, "pause_duration_range": 0,
            "pause_duration_iqr": 0, "pause_duration_q25": 0, "pause_duration_q75": 0,
            "pause_duration_q90": 0, "long_pause_ratio": 0, "short_pause_ratio": 0,
            "pause_irregularity": 0
        }
    
    def _empty_rhythm_features(self) -> Dict[str, float]:
        """Return empty rhythm features dictionary."""
        return {
            "syllable_rate": 0, "mean_syllable_interval": 0, "syllable_interval_std": 0,
            "syllable_interval_cv": 0, "rhythm_variability": 0, "rhythm_irregularity": 0,
            "syllable_interval_range": 0, "rhythm_entropy": 0, "rhythm_consistency": 0,
            "isochrony_measure": 0, "word_rate": 0, "word_interval_variability": 0
        }
    
    def _empty_energy_features(self) -> Dict[str, float]:
        """Return empty energy features dictionary."""
        return {
            "rms_mean": 0, "rms_std": 0, "rms_max": 0, "rms_min": 0, "rms_range": 0,
            "energy_variance": 0, "energy_dynamic_range": 0, "energy_variability": 0,
            "energy_contour_slope": 0, "energy_contour_variability": 0, "energy_skewness": 0,
            "energy_kurtosis": 0, "energy_q25": 0, "energy_q75": 0, "energy_iqr": 0,
            "loudness_stability": 0, "energy_consistency": 0, "spectral_energy_mean": 0,
            "spectral_energy_std": 0, "spectral_energy_variability": 0
        }
