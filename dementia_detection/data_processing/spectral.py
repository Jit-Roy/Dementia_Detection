"""
Advanced Spectral Analysis Module
Extracts detailed spectral features including MFCC variations, spectral shape,
clarity measures, and timbre characteristics for dementia detection.
"""

import librosa
import numpy as np
import scipy.stats
from scipy.signal import butter, filtfilt
from typing import Dict, List, Optional





class AdvancedSpectralAnalyzer:
    """Extract comprehensive spectral and timbre features for dementia detection."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
    def extract_detailed_mfcc_features(self, audio: np.ndarray, n_mfcc: int = 13) -> Dict[str, float]:
        """
        Extract detailed MFCC features with advanced statistics and dynamics.
        
        Args:
            audio: Input audio array
            n_mfcc: Number of MFCC coefficients
            
        Returns:
            Dictionary of detailed MFCC features
        """
        try:
            features = {}
            
            # Extract MFCCs with higher resolution
            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=n_mfcc,
                n_fft=2048,
                hop_length=256,  # Higher resolution
                n_mels=40
            )
            
            # Delta and delta-delta coefficients
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            
            # Global MFCC statistics (phonetic clarity indicators)
            for coeff in range(n_mfcc):
                coeff_values = mfccs[coeff, :]
                delta_values = mfcc_delta[coeff, :]
                delta2_values = mfcc_delta2[coeff, :]
                
                # Static MFCC statistics
                features[f"mfcc_{coeff}_mean"] = np.mean(coeff_values)
                features[f"mfcc_{coeff}_std"] = np.std(coeff_values)
                features[f"mfcc_{coeff}_skew"] = scipy.stats.skew(coeff_values)
                features[f"mfcc_{coeff}_kurtosis"] = scipy.stats.kurtosis(coeff_values)
                features[f"mfcc_{coeff}_range"] = np.max(coeff_values) - np.min(coeff_values)
                
                # Dynamic features (important for speech clarity)
                features[f"mfcc_{coeff}_delta_mean"] = np.mean(np.abs(delta_values))
                features[f"mfcc_{coeff}_delta_std"] = np.std(delta_values)
                features[f"mfcc_{coeff}_delta2_mean"] = np.mean(np.abs(delta2_values))
                
                # Coefficient stability (dementia indicator)
                features[f"mfcc_{coeff}_stability"] = 1.0 / (1.0 + np.std(coeff_values))
                features[f"mfcc_{coeff}_dynamic_range"] = np.max(coeff_values) - np.min(coeff_values)
                
            # Cross-coefficient relationships
            features["mfcc_correlation_c0_c1"] = np.corrcoef(mfccs[0, :], mfccs[1, :])[0, 1] if len(mfccs[0, :]) > 1 else 0
            features["mfcc_correlation_c1_c2"] = np.corrcoef(mfccs[1, :], mfccs[2, :])[0, 1] if len(mfccs[1, :]) > 1 else 0
            
            # Global MFCC trajectory features
            mfcc_mean_trajectory = np.mean(mfccs, axis=0)
            features["mfcc_trajectory_variance"] = np.var(mfcc_mean_trajectory)
            features["mfcc_trajectory_range"] = np.max(mfcc_mean_trajectory) - np.min(mfcc_mean_trajectory)
            
            # Spectral consistency measures
            features["mfcc_spectral_consistency"] = 1.0 / (1.0 + np.mean([np.std(mfccs[i, :]) for i in range(n_mfcc)]))
            
            return features
            
        except Exception as e:
            
            return self._empty_mfcc_features(n_mfcc)
    
    def extract_spectral_shape_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract spectral shape and distribution features.
        
        Args:
            audio: Input audio array
            
        Returns:
            Dictionary of spectral shape features
        """
        try:
            features = {}
            
            # Basic spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate)
            spectral_flatness = librosa.feature.spectral_flatness(y=audio)[0]
            
            # Spectral centroid features (brightness indicator)
            features["spectral_centroid_mean"] = np.mean(spectral_centroid)
            features["spectral_centroid_std"] = np.std(spectral_centroid)
            features["spectral_centroid_median"] = np.median(spectral_centroid)
            features["spectral_centroid_range"] = np.max(spectral_centroid) - np.min(spectral_centroid)
            features["spectral_centroid_skew"] = scipy.stats.skew(spectral_centroid)
            
            # Spectral bandwidth features (clarity indicator)\n            features["spectral_bandwidth_mean"] = np.mean(spectral_bandwidth)
            features["spectral_bandwidth_std"] = np.std(spectral_bandwidth)
            features["spectral_bandwidth_cv"] = np.std(spectral_bandwidth) / np.mean(spectral_bandwidth) if np.mean(spectral_bandwidth) > 0 else 0
            
            # Spectral roll-off features (energy distribution)
            features["spectral_rolloff_mean"] = np.mean(spectral_rolloff)
            features["spectral_rolloff_std"] = np.std(spectral_rolloff)
            features["spectral_rolloff_range"] = np.max(spectral_rolloff) - np.min(spectral_rolloff)
            
            # Spectral contrast features (clarity vs noise)
            for i, contrast_band in enumerate(spectral_contrast):
                features[f"spectral_contrast_band_{i}_mean"] = np.mean(contrast_band)
                features[f"spectral_contrast_band_{i}_std"] = np.std(contrast_band)
            
            # Overall spectral contrast
            features["spectral_contrast_mean"] = np.mean(spectral_contrast)
            features["spectral_contrast_std"] = np.std(spectral_contrast)
            
            # Spectral flatness (harmonicity indicator)
            features["spectral_flatness_mean"] = np.mean(spectral_flatness)
            features["spectral_flatness_std"] = np.std(spectral_flatness)
            features["spectral_flatness_median"] = np.median(spectral_flatness)
            
            # Derived clarity measures
            features["spectral_clarity"] = np.mean(spectral_contrast) / (np.mean(spectral_flatness) + 1e-10)
            features["spectral_brightness"] = features["spectral_centroid_mean"] / (self.sample_rate / 2)  # Normalized
            features["spectral_sharpness"] = 1.0 / (1.0 + features["spectral_bandwidth_mean"] / 1000.0)  # Inverse bandwidth
            
            return features
            
        except Exception as e:
            
            return self._empty_spectral_features()
    
    def extract_timbre_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract timbre and voice quality features.
        
        Args:
            audio: Input audio array
            
        Returns:
            Dictionary of timbre features
        """
        try:
            features = {}
            
            # Zero-crossing rate (roughness indicator)
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features["zcr_mean"] = np.mean(zcr)
            features["zcr_std"] = np.std(zcr)
            features["zcr_variance"] = np.var(zcr)
            
            # Chroma features (harmonic content)
            chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
            features["chroma_mean"] = np.mean(chroma)
            features["chroma_std"] = np.std(chroma)
            features["chroma_variance"] = np.var(chroma)
            
            # Harmonic-percussive separation
            harmonic, percussive = librosa.effects.hpss(audio)
            
            # Harmonic content features
            harmonic_energy = np.sum(harmonic ** 2)
            percussive_energy = np.sum(percussive ** 2)
            total_energy = harmonic_energy + percussive_energy
            
            features["harmonic_ratio"] = harmonic_energy / total_energy if total_energy > 0 else 0
            features["percussive_ratio"] = percussive_energy / total_energy if total_energy > 0 else 0
            features["harmonic_to_noise_ratio"] = harmonic_energy / (percussive_energy + 1e-10)
            
            # Tonal vs noise characteristics
            features["tonality"] = features["harmonic_ratio"]
            features["noisiness"] = 1.0 - features["harmonic_ratio"]
            
            # Spectral entropy (randomness measure)
            stft = np.abs(librosa.stft(audio))
            spectral_entropy = []
            for frame in stft.T:
                if np.sum(frame) > 0:
                    frame_normalized = frame / np.sum(frame)
                    entropy = -np.sum(frame_normalized * np.log2(frame_normalized + 1e-10))
                    spectral_entropy.append(entropy)
            
            if spectral_entropy:
                features["spectral_entropy_mean"] = np.mean(spectral_entropy)
                features["spectral_entropy_std"] = np.std(spectral_entropy)
                features["spectral_entropy_range"] = np.max(spectral_entropy) - np.min(spectral_entropy)
            else:
                features["spectral_entropy_mean"] = 0
                features["spectral_entropy_std"] = 0
                features["spectral_entropy_range"] = 0
            
            # Roughness estimation (based on spectral irregularity)
            spectral_diff = np.diff(stft, axis=0)
            spectral_roughness = np.mean(np.sum(np.abs(spectral_diff), axis=0))
            features["spectral_roughness"] = spectral_roughness
            features["spectral_smoothness"] = 1.0 / (1.0 + spectral_roughness)
            
            return features
            
        except Exception as e:
            
            return self._empty_timbre_features()
    
    def extract_voice_clarity_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract voice clarity and speech quality features.
        
        Args:
            audio: Input audio array
            
        Returns:
            Dictionary of clarity features
        """
        try:
            features = {}
            
            # Fundamental frequency and harmonics analysis
            f0, voiced_flag, voiced_probs = librosa.pyin(audio, 
                                                        fmin=librosa.note_to_hz('C2'),
                                                        fmax=librosa.note_to_hz('C7'))
            
            if len(f0[~np.isnan(f0)]) > 0:
                valid_f0 = f0[~np.isnan(f0)]
                
                # F0 stability measures (clarity indicator)
                features["f0_stability"] = 1.0 / (1.0 + np.std(valid_f0) / np.mean(valid_f0))
                features["f0_jitter_approx"] = np.std(np.diff(valid_f0)) / np.mean(valid_f0)
                
                # Voicing probability measures
                features["voicing_strength"] = np.mean(voiced_probs[~np.isnan(voiced_probs)])
                features["voicing_consistency"] = np.std(voiced_probs[~np.isnan(voiced_probs)])
                
            else:
                features["f0_stability"] = 0
                features["f0_jitter_approx"] = 0
                features["voicing_strength"] = 0
                features["voicing_consistency"] = 0
            
            # Signal-to-noise ratio estimation
            # Use spectral subtraction approach
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            
            # Estimate noise floor (bottom 10% of spectral energy)
            noise_floor = np.percentile(magnitude, 10, axis=1, keepdims=True)
            signal_power = np.mean(magnitude, axis=1, keepdims=True)
            
            snr_per_freq = 10 * np.log10((signal_power / (noise_floor + 1e-10)))
            features["estimated_snr_db"] = np.mean(snr_per_freq)
            features["snr_variance"] = np.var(snr_per_freq)
            
            # Clarity measures based on spectral characteristics
            # High-frequency energy ratio (breathiness indicator)
            freq_bins = librosa.fft_frequencies(sr=self.sample_rate, n_fft=2048)
            high_freq_mask = freq_bins > 4000  # Above 4kHz
            low_freq_mask = freq_bins < 1000   # Below 1kHz
            
            high_freq_energy = np.mean(magnitude[high_freq_mask, :])
            low_freq_energy = np.mean(magnitude[low_freq_mask, :])
            features["high_freq_ratio"] = high_freq_energy / (low_freq_energy + 1e-10)
            
            # Spectral tilt (voice quality measure)
            # Linear regression slope of log spectrum
            log_magnitude = np.log(np.mean(magnitude, axis=1) + 1e-10)
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(freq_bins[:len(log_magnitude)], log_magnitude)
            features["spectral_tilt"] = slope
            features["spectral_tilt_r2"] = r_value ** 2
            
            # Formant clarity approximation
            # Look for prominent peaks in spectrum
            mean_spectrum = np.mean(magnitude, axis=1)
            from scipy.signal import find_peaks
            peaks, properties = find_peaks(mean_spectrum, height=np.mean(mean_spectrum))
            
            if len(peaks) >= 2:
                # Formant clarity based on peak prominence
                peak_heights = mean_spectrum[peaks]
                valley_depths = []
                for i in range(len(peaks) - 1):
                    valley_region = mean_spectrum[peaks[i]:peaks[i+1]]
                    if len(valley_region) > 0:
                        valley_depths.append(np.min(valley_region))
                
                if valley_depths:
                    formant_clarity = np.mean(peak_heights[:-1]) / (np.mean(valley_depths) + 1e-10)
                    features["formant_clarity"] = formant_clarity
                else:
                    features["formant_clarity"] = 0
            else:
                features["formant_clarity"] = 0
            
            return features
            
        except Exception as e:
            
            return self._empty_clarity_features()
    
    def _empty_mfcc_features(self, n_mfcc: int) -> Dict[str, float]:
        """Return empty MFCC features dictionary."""
        features = {}
        for coeff in range(n_mfcc):
            features.update({
                f"mfcc_{coeff}_mean": 0, f"mfcc_{coeff}_std": 0, f"mfcc_{coeff}_skew": 0,
                f"mfcc_{coeff}_kurtosis": 0, f"mfcc_{coeff}_range": 0, f"mfcc_{coeff}_delta_mean": 0,
                f"mfcc_{coeff}_delta_std": 0, f"mfcc_{coeff}_delta2_mean": 0, f"mfcc_{coeff}_stability": 0,
                f"mfcc_{coeff}_dynamic_range": 0
            })
        
        features.update({
            "mfcc_correlation_c0_c1": 0, "mfcc_correlation_c1_c2": 0, "mfcc_trajectory_variance": 0,
            "mfcc_trajectory_range": 0, "mfcc_spectral_consistency": 0
        })
        return features
    
    def _empty_spectral_features(self) -> Dict[str, float]:
        """Return empty spectral features dictionary."""
        features = {
            "spectral_centroid_mean": 0, "spectral_centroid_std": 0, "spectral_centroid_median": 0,
            "spectral_centroid_range": 0, "spectral_centroid_skew": 0, "spectral_bandwidth_mean": 0,
            "spectral_bandwidth_std": 0, "spectral_bandwidth_cv": 0, "spectral_rolloff_mean": 0,
            "spectral_rolloff_std": 0, "spectral_rolloff_range": 0, "spectral_contrast_mean": 0,
            "spectral_contrast_std": 0, "spectral_flatness_mean": 0, "spectral_flatness_std": 0,
            "spectral_flatness_median": 0, "spectral_clarity": 0, "spectral_brightness": 0,
            "spectral_sharpness": 0
        }
        
        # Add spectral contrast band features
        for i in range(7):  # librosa default number of contrast bands
            features[f"spectral_contrast_band_{i}_mean"] = 0
            features[f"spectral_contrast_band_{i}_std"] = 0
        
        return features
    
    def _empty_timbre_features(self) -> Dict[str, float]:
        """Return empty timbre features dictionary."""
        return {
            "zcr_mean": 0, "zcr_std": 0, "zcr_variance": 0, "chroma_mean": 0, "chroma_std": 0,
            "chroma_variance": 0, "harmonic_ratio": 0, "percussive_ratio": 0, "harmonic_to_noise_ratio": 0,
            "tonality": 0, "noisiness": 0, "spectral_entropy_mean": 0, "spectral_entropy_std": 0,
            "spectral_entropy_range": 0, "spectral_roughness": 0, "spectral_smoothness": 0
        }
    
    def _empty_clarity_features(self) -> Dict[str, float]:
        """Return empty clarity features dictionary."""
        return {
            "f0_stability": 0, "f0_jitter_approx": 0, "voicing_strength": 0, "voicing_consistency": 0,
            "estimated_snr_db": 0, "snr_variance": 0, "high_freq_ratio": 0, "spectral_tilt": 0,
            "spectral_tilt_r2": 0, "formant_clarity": 0
        }
