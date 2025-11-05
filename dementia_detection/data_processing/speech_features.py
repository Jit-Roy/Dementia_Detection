"""
Speech Feature Extraction Module
Extracts comprehensive speech features including MFCCs, pitch, energy, 
formants, speech rate, pause duration, and prosody features.
"""

import librosa
import numpy as np
import scipy.stats
from scipy.signal import find_peaks, butter, filtfilt
from typing import Dict, List, Tuple, Optional


from ..config.settings import SPEECH_FEATURES_CONFIG




class SpeechFeatureExtractor:
    """Extract comprehensive speech features for dementia detection."""
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize SpeechFeatureExtractor.
        
        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self.config = SPEECH_FEATURES_CONFIG
        
    def extract_mfcc_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract MFCC features and their statistics.
        
        Args:
            audio: Input audio array
            
        Returns:
            Dictionary of MFCC feature statistics
        """
        try:
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=self.config["mfcc_coefficients"],
                n_fft=self.config["n_fft"],
                hop_length=self.config["hop_length"],
                n_mels=self.config["mel_filters"]
            )
            
            # Compute delta and delta-delta features
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            
            # Combine all MFCC features
            all_mfccs = np.vstack([mfccs, mfcc_delta, mfcc_delta2])
            
            # Statistical features
            features = {}
            
            # Per-coefficient statistics
            for i in range(all_mfccs.shape[0]):
                coeff_name = f"mfcc_{i+1}" if i < len(mfccs) else \
                           f"mfcc_delta_{i-len(mfccs)+1}" if i < len(mfccs)*2 else \
                           f"mfcc_delta2_{i-len(mfccs)*2+1}"
                
                features[f"{coeff_name}_mean"] = np.mean(all_mfccs[i])
                features[f"{coeff_name}_std"] = np.std(all_mfccs[i])
                features[f"{coeff_name}_skew"] = scipy.stats.skew(all_mfccs[i])
                features[f"{coeff_name}_kurtosis"] = scipy.stats.kurtosis(all_mfccs[i])
            
            # Global statistics
            features["mfcc_mean_trajectory"] = np.mean(np.mean(mfccs, axis=0))
            features["mfcc_std_trajectory"] = np.std(np.mean(mfccs, axis=0))
            features["mfcc_range"] = np.max(mfccs) - np.min(mfccs)
            
            return features
            
        except Exception as e:
            
            return {}
    
    def extract_pitch_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract pitch-related features using librosa.
        
        Args:
            audio: Input audio array
            
        Returns:
            Dictionary of pitch features
        """
        try:
            # Extract fundamental frequency using librosa's piptrack
            pitches, magnitudes = librosa.piptrack(
                y=audio,
                sr=self.sample_rate,
                threshold=0.1,
                fmin=self.config["pitch_min"],
                fmax=self.config["pitch_max"]
            )
            
            # Get pitch values by selecting the pitch with highest magnitude at each frame
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            pitch_values = np.array(pitch_values)
            
            if len(pitch_values) == 0:
                
                return self._empty_pitch_features()
            
            # Basic pitch statistics
            features = {
                "pitch_mean": np.mean(pitch_values),
                "pitch_std": np.std(pitch_values),
                "pitch_min": np.min(pitch_values),
                "pitch_max": np.max(pitch_values),
                "pitch_range": np.max(pitch_values) - np.min(pitch_values),
                "pitch_median": np.median(pitch_values),
                "pitch_q25": np.percentile(pitch_values, 25),
                "pitch_q75": np.percentile(pitch_values, 75),
                "pitch_iqr": np.percentile(pitch_values, 75) - np.percentile(pitch_values, 25),
            }
            
            # Advanced pitch features
            features["pitch_skewness"] = scipy.stats.skew(pitch_values)
            features["pitch_kurtosis"] = scipy.stats.kurtosis(pitch_values)
            
            # Pitch variability measures
            features["pitch_cv"] = features["pitch_std"] / features["pitch_mean"] if features["pitch_mean"] > 0 else 0
            
            # Pitch slope (overall trend)
            if len(pitch_values) > 1:
                x = np.arange(len(pitch_values))
                slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, pitch_values)
                features["pitch_slope"] = slope
                features["pitch_slope_r2"] = r_value**2
            else:
                features["pitch_slope"] = 0
                features["pitch_slope_r2"] = 0
            
            # Voicing rate (estimate based on non-zero pitch frames)
            # Calculate total frames from audio length
            hop_length = 512
            total_frames = int(np.ceil(len(audio) / hop_length))
            voiced_frames = len(pitch_values)
            features["voicing_rate"] = voiced_frames / total_frames if total_frames > 0 else 0
            
            # Voice quality measures (simplified alternatives to jitter/shimmer)
            try:
                if len(pitch_values) > 1:
                    # Pitch stability measures (alternatives to jitter)
                    pitch_diff = np.diff(pitch_values)
                    features["pitch_stability"] = 1.0 / (1.0 + np.std(pitch_diff))
                    features["pitch_perturbation"] = np.std(pitch_diff) / np.mean(pitch_values) if np.mean(pitch_values) > 0 else 0
                    
                    # Energy stability measures (alternatives to shimmer)
                    rms_energy = librosa.feature.rms(y=audio, hop_length=512)[0]
                    if len(rms_energy) > 1:
                        energy_diff = np.diff(rms_energy)
                        features["energy_stability"] = 1.0 / (1.0 + np.std(energy_diff))
                        features["energy_perturbation"] = np.std(energy_diff) / np.mean(rms_energy) if np.mean(rms_energy) > 0 else 0
                    else:
                        features["energy_stability"] = 0
                        features["energy_perturbation"] = 0
                else:
                    features["pitch_stability"] = 0
                    features["pitch_perturbation"] = 0
                    features["energy_stability"] = 0
                    features["energy_perturbation"] = 0
                
            except Exception as e:
                
                features.update(self._empty_voice_quality_features())
            
            return features
            
        except Exception as e:
            return self._empty_pitch_features()
    
    def extract_energy_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract energy-related features.
        
        Args:
            audio: Input audio array
            
        Returns:
            Dictionary of energy features
        """
        try:
            # RMS Energy
            rms = librosa.feature.rms(
                y=audio, 
                frame_length=self.config["window_length"],
                hop_length=self.config["hop_length"]
            )[0]
            
            # Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(
                audio, 
                frame_length=self.config["window_length"],
                hop_length=self.config["hop_length"]
            )[0]
            
            # Spectral Centroid (brightness)
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio, 
                sr=self.sample_rate,
                hop_length=self.config["hop_length"]
            )[0]
            
            # Spectral Rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio, 
                sr=self.sample_rate,
                hop_length=self.config["hop_length"]
            )[0]
            
            # Spectral Bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio, 
                sr=self.sample_rate,
                hop_length=self.config["hop_length"]
            )[0]
            
            features = {}
            
            # RMS Energy statistics
            features["rms_mean"] = np.mean(rms)
            features["rms_std"] = np.std(rms)
            features["rms_max"] = np.max(rms)
            features["rms_min"] = np.min(rms)
            features["rms_range"] = np.max(rms) - np.min(rms)
            
            # ZCR statistics
            features["zcr_mean"] = np.mean(zcr)
            features["zcr_std"] = np.std(zcr)
            
            # Spectral features statistics
            features["spectral_centroid_mean"] = np.mean(spectral_centroid)
            features["spectral_centroid_std"] = np.std(spectral_centroid)
            
            features["spectral_rolloff_mean"] = np.mean(spectral_rolloff)
            features["spectral_rolloff_std"] = np.std(spectral_rolloff)
            
            features["spectral_bandwidth_mean"] = np.mean(spectral_bandwidth)
            features["spectral_bandwidth_std"] = np.std(spectral_bandwidth)
            
            # Dynamic range
            features["dynamic_range_db"] = 20 * np.log10(np.max(rms) / (np.min(rms) + 1e-10))
            
            return features
            
        except Exception as e:
            
            return {}
    
    def extract_formant_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract formant-like features using spectral analysis.
        
        Args:
            audio: Input audio array
            
        Returns:
            Dictionary of formant features
        """
        try:
            # Use spectral analysis to estimate formant-like features
            # Get spectral centroids and peaks as formant approximations
            
            # Compute STFT
            stft = librosa.stft(audio, hop_length=512, n_fft=2048)
            magnitude = np.abs(stft)
            
            features = {}
            
            # Extract spectral peaks as formant approximations
            for f_num in range(1, self.config["formant_count"] + 1):
                try:
                    formant_values = []
                    
                    # For each time frame, find spectral peaks
                    for frame in range(magnitude.shape[1]):
                        spectrum = magnitude[:, frame]
                        
                        # Find peaks in the spectrum
                        peaks, _ = find_peaks(spectrum, height=np.max(spectrum) * 0.1)
                        
                        if len(peaks) >= f_num:
                            # Convert bin to frequency
                            freq_resolution = self.sample_rate / 2048
                            formant_freq = peaks[f_num - 1] * freq_resolution
                            
                            # Basic formant frequency ranges
                            if f_num == 1 and 200 <= formant_freq <= 1000:
                                formant_values.append(formant_freq)
                            elif f_num == 2 and 800 <= formant_freq <= 2500:
                                formant_values.append(formant_freq)
                            elif f_num == 3 and 1500 <= formant_freq <= 3500:
                                formant_values.append(formant_freq)
                            elif f_num == 4 and 2500 <= formant_freq <= 4500:
                                formant_values.append(formant_freq)
                    
                    if len(formant_values) > 0:
                        features[f"f{f_num}_mean"] = np.mean(formant_values)
                        features[f"f{f_num}_std"] = np.std(formant_values)
                        features[f"f{f_num}_median"] = np.median(formant_values)
                        features[f"f{f_num}_range"] = np.max(formant_values) - np.min(formant_values)
                    else:
                        features[f"f{f_num}_mean"] = 0
                        features[f"f{f_num}_std"] = 0
                        features[f"f{f_num}_median"] = 0
                        features[f"f{f_num}_range"] = 0
                        
                except Exception as e:
                    
                    features[f"f{f_num}_mean"] = 0
                    features[f"f{f_num}_std"] = 0
                    features[f"f{f_num}_median"] = 0
                    features[f"f{f_num}_range"] = 0
            
            # Formant ratios and relationships
            if features.get("f1_mean", 0) > 0 and features.get("f2_mean", 0) > 0:
                features["f2_f1_ratio"] = features["f2_mean"] / features["f1_mean"]
            else:
                features["f2_f1_ratio"] = 0
                
            if features.get("f3_mean", 0) > 0 and features.get("f2_mean", 0) > 0:
                features["f3_f2_ratio"] = features["f3_mean"] / features["f2_mean"]
            else:
                features["f3_f2_ratio"] = 0
            
            return features
            
        except Exception as e:
            
            # Return default values for all formant features
            default_features = {}
            for i in range(1, self.config["formant_count"] + 1):
                default_features.update({
                    f"f{i}_mean": 0, f"f{i}_std": 0, 
                    f"f{i}_median": 0, f"f{i}_range": 0
                })
            return default_features
    
    def extract_timing_features(self, audio: np.ndarray, 
                               speech_segments: List[Tuple[float, float]] = None) -> Dict[str, float]:
        """
        Extract speech timing features including speech rate and pause duration.
        
        Args:
            audio: Input audio array
            speech_segments: List of (start, end) time pairs for speech segments
            
        Returns:
            Dictionary of timing features
        """
        try:
            features = {}
            
            # Total duration
            total_duration = len(audio) / self.sample_rate
            features["total_duration"] = total_duration
            
            if speech_segments is None or len(speech_segments) == 0:
                # Simple voice activity detection using energy
                rms = librosa.feature.rms(y=audio, hop_length=512)[0]
                rms_threshold = np.mean(rms) * 0.3
                
                # Convert to time-based segments
                hop_duration = 512 / self.sample_rate
                speech_frames = rms > rms_threshold
                
                # Find speech segments
                speech_segments = []
                in_speech = False
                start_time = 0
                
                for i, is_speech in enumerate(speech_frames):
                    current_time = i * hop_duration
                    
                    if is_speech and not in_speech:
                        start_time = current_time
                        in_speech = True
                    elif not is_speech and in_speech:
                        speech_segments.append((start_time, current_time))
                        in_speech = False
                
                if in_speech:
                    speech_segments.append((start_time, total_duration))
            
            # Speech and pause timing
            if speech_segments:
                speech_durations = [end - start for start, end in speech_segments]
                total_speech_time = sum(speech_durations)
                
                # Pause durations
                pause_durations = []
                if len(speech_segments) > 1:
                    for i in range(len(speech_segments) - 1):
                        pause_start = speech_segments[i][1]
                        pause_end = speech_segments[i + 1][0]
                        if pause_end > pause_start:
                            pause_durations.append(pause_end - pause_start)
                
                features["speech_time"] = total_speech_time
                features["pause_time"] = total_duration - total_speech_time
                features["speech_rate"] = total_speech_time / total_duration if total_duration > 0 else 0
                features["pause_rate"] = (total_duration - total_speech_time) / total_duration if total_duration > 0 else 0
                
                # Speech segment statistics
                features["num_speech_segments"] = len(speech_segments)
                features["mean_speech_segment_duration"] = np.mean(speech_durations) if speech_durations else 0
                features["std_speech_segment_duration"] = np.std(speech_durations) if len(speech_durations) > 1 else 0
                features["max_speech_segment_duration"] = np.max(speech_durations) if speech_durations else 0
                features["min_speech_segment_duration"] = np.min(speech_durations) if speech_durations else 0
                
                # Pause statistics
                if pause_durations:
                    features["num_pauses"] = len(pause_durations)
                    features["mean_pause_duration"] = np.mean(pause_durations)
                    features["std_pause_duration"] = np.std(pause_durations) if len(pause_durations) > 1 else 0
                    features["max_pause_duration"] = np.max(pause_durations)
                    features["min_pause_duration"] = np.min(pause_durations)
                    features["pause_to_speech_ratio"] = sum(pause_durations) / total_speech_time if total_speech_time > 0 else 0
                else:
                    features["num_pauses"] = 0
                    features["mean_pause_duration"] = 0
                    features["std_pause_duration"] = 0
                    features["max_pause_duration"] = 0
                    features["min_pause_duration"] = 0
                    features["pause_to_speech_ratio"] = 0
                
                # Articulation rate (words per minute - rough estimate)
                # This is a rough estimate based on syllables
                syllable_rate = self._estimate_syllable_rate(audio)
                features["estimated_syllable_rate"] = syllable_rate
                features["articulation_rate"] = syllable_rate * 60 / total_speech_time if total_speech_time > 0 else 0
                
            else:
                # No speech segments found
                features.update({
                    "speech_time": 0, "pause_time": total_duration, "speech_rate": 0, "pause_rate": 1,
                    "num_speech_segments": 0, "mean_speech_segment_duration": 0,
                    "std_speech_segment_duration": 0, "max_speech_segment_duration": 0,
                    "min_speech_segment_duration": 0, "num_pauses": 0,
                    "mean_pause_duration": 0, "std_pause_duration": 0,
                    "max_pause_duration": 0, "min_pause_duration": 0,
                    "pause_to_speech_ratio": 0, "estimated_syllable_rate": 0,
                    "articulation_rate": 0
                })
            
            return features
            
        except Exception as e:
            
            return {}
    
    def extract_prosody_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract prosodic features including intonation, rhythm, and variability.
        
        Args:
            audio: Input audio array
            
        Returns:
            Dictionary of prosody features
        """
        try:
            features = {}
            
            # Pitch-based prosody (using librosa)
            try:
                # Extract fundamental frequency using librosa's piptrack
                pitches, magnitudes = librosa.piptrack(
                    y=audio,
                    sr=self.sample_rate,
                    threshold=0.1,
                    fmin=self.config["pitch_min"],
                    fmax=self.config["pitch_max"]
                )
                
                # Get pitch values by selecting the pitch with highest magnitude at each frame
                pitch_values = []
                for t in range(pitches.shape[1]):
                    index = magnitudes[:, t].argmax()
                    pitch = pitches[index, t]
                    if pitch > 0:
                        pitch_values.append(pitch)
                
                pitch_values = np.array(pitch_values)
                
                if len(pitch_values) > 1:
                    # Pitch contour analysis
                    pitch_diff = np.diff(pitch_values)
                    features["pitch_contour_complexity"] = np.std(pitch_diff)
                    features["pitch_direction_changes"] = np.sum(np.diff(np.sign(pitch_diff)) != 0)
                    
                    # Intonation patterns
                    features["pitch_reset_rate"] = features["pitch_direction_changes"] / len(pitch_values)
                    
                    # Declination (overall pitch trend)
                    x = np.arange(len(pitch_values))
                    slope, _, _, _, _ = scipy.stats.linregress(x, pitch_values)
                    features["pitch_declination"] = slope
                else:
                    features["pitch_contour_complexity"] = 0
                    features["pitch_direction_changes"] = 0
                    features["pitch_reset_rate"] = 0
                    features["pitch_declination"] = 0
                    
            except Exception as e:
                
                features.update({
                    "pitch_contour_complexity": 0, "pitch_direction_changes": 0,
                    "pitch_reset_rate": 0, "pitch_declination": 0
                })
            
            # Energy-based prosody
            rms = librosa.feature.rms(y=audio, hop_length=512)[0]
            
            # Energy contour
            rms_diff = np.diff(rms)
            features["energy_contour_complexity"] = np.std(rms_diff)
            features["energy_dynamic_range"] = np.max(rms) - np.min(rms)
            
            # Rhythm analysis using onset detection
            onset_frames = librosa.onset.onset_detect(
                y=audio, 
                sr=self.sample_rate, 
                hop_length=512,
                units='frames'
            )
            
            if len(onset_frames) > 1:
                onset_times = librosa.frames_to_time(onset_frames, sr=self.sample_rate, hop_length=512)
                inter_onset_intervals = np.diff(onset_times)
                
                features["rhythm_regularity"] = 1 / (1 + np.std(inter_onset_intervals)) if len(inter_onset_intervals) > 0 else 0
                features["onset_rate"] = len(onset_frames) / (len(audio) / self.sample_rate)
                features["mean_inter_onset_interval"] = np.mean(inter_onset_intervals) if len(inter_onset_intervals) > 0 else 0
                features["std_inter_onset_interval"] = np.std(inter_onset_intervals) if len(inter_onset_intervals) > 1 else 0
            else:
                features["rhythm_regularity"] = 0
                features["onset_rate"] = 0
                features["mean_inter_onset_interval"] = 0
                features["std_inter_onset_interval"] = 0
            
            # Spectral flux for rhythmic analysis
            stft = librosa.stft(audio, hop_length=512)
            magnitude = np.abs(stft)
            spectral_flux = np.sum(np.diff(magnitude, axis=1)**2, axis=0)
            features["spectral_flux_mean"] = np.mean(spectral_flux)
            features["spectral_flux_std"] = np.std(spectral_flux)
            
            return features
            
        except Exception as e:
            
            return {}
    
    def extract_all_features(self, audio: np.ndarray, 
                           speech_segments: List[Tuple[float, float]] = None) -> Dict[str, float]:
        """
        Extract all speech features.
        
        Args:
            audio: Input audio array
            speech_segments: List of speech segments for timing analysis
            
        Returns:
            Dictionary containing all extracted features
        """
        try:
            all_features = {}
            
            all_features.update(self.extract_mfcc_features(audio))
            all_features.update(self.extract_pitch_features(audio))
            all_features.update(self.extract_energy_features(audio))
            all_features.update(self.extract_formant_features(audio))
            all_features.update(self.extract_timing_features(audio, speech_segments))
            all_features.update(self.extract_prosody_features(audio))
            
            return all_features
            
        except Exception as e:
            return {}
    
    def _estimate_syllable_rate(self, audio: np.ndarray) -> float:
        """Estimate syllable rate using onset detection."""
        try:
            onset_frames = librosa.onset.onset_detect(
                y=audio, 
                sr=self.sample_rate,
                hop_length=512
            )
            return len(onset_frames) / (len(audio) / self.sample_rate)
        except:
            return 0.0
    
    def _empty_pitch_features(self) -> Dict[str, float]:
        """Return empty pitch features dictionary."""
        return {
            "pitch_mean": 0, "pitch_std": 0, "pitch_min": 0, "pitch_max": 0,
            "pitch_range": 0, "pitch_median": 0, "pitch_q25": 0, "pitch_q75": 0,
            "pitch_iqr": 0, "pitch_skewness": 0, "pitch_kurtosis": 0,
            "pitch_cv": 0, "pitch_slope": 0, "pitch_slope_r2": 0,
            "voicing_rate": 0, "pitch_stability": 0, "pitch_perturbation": 0,
            "energy_stability": 0, "energy_perturbation": 0
        }
    
    def _empty_voice_quality_features(self) -> Dict[str, float]:
        """Return empty voice quality features."""
        return {
            "pitch_stability": 0, "pitch_perturbation": 0,
            "energy_stability": 0, "energy_perturbation": 0
        }


# Example usage
if __name__ == "__main__":
    # Example usage
    extractor = SpeechFeatureExtractor()
    
    # # Extract features from audio
    # features = extractor.extract_all_features(audio_array)
    # print(f"Extracted {len(features)} features")
    pass
