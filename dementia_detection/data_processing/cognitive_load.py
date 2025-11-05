"""
Cognitive Load Analysis Module
Analyzes speech patterns that indicate cognitive processing load,
mental effort, and cognitive decline indicators for dementia detection.
"""

import librosa
import numpy as np
import scipy.stats
from typing import Dict, List, Tuple, Optional, Union

from sklearn.preprocessing import StandardScaler
from scipy import signal
import warnings

warnings.filterwarnings('ignore')



class CognitiveLoadAnalyzer:
    """Analyze cognitive load indicators in speech for dementia detection."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.hop_length = 512
        
    def extract_processing_effort_features(self, audio: np.ndarray, 
                                         transcript: Optional[str] = None) -> Dict[str, float]:
        """
        Extract features indicating cognitive processing effort.
        
        Args:
            audio: Input audio array
            transcript: Optional transcript text
            
        Returns:
            Dictionary of processing effort features
        """
        try:
            features = {}
            
            # Temporal processing indicators
            temporal_features = self._extract_temporal_effort_features(audio)
            features.update(temporal_features)
            
            # Spectral complexity indicators
            spectral_features = self._extract_spectral_complexity_features(audio)
            features.update(spectral_features)
            
            # Prosodic effort indicators
            prosodic_features = self._extract_prosodic_effort_features(audio)
            features.update(prosodic_features)
            
            # Voice quality under cognitive load
            voice_quality_features = self._extract_voice_quality_under_load(audio)
            features.update(voice_quality_features)
            
            # Text-based cognitive effort (if transcript available)
            if transcript:
                text_effort_features = self._extract_text_effort_features(transcript)
                features.update(text_effort_features)
            else:
                features.update(self._empty_text_effort_features())
            
            return features
            
        except Exception as e:
            return self._empty_processing_effort_features()
    
    def extract_cognitive_decline_indicators(self, audio: np.ndarray,
                                           baseline_features: Optional[Dict] = None) -> Dict[str, float]:
        """
        Extract indicators of cognitive decline from speech patterns.
        
        Args:
            audio: Input audio array
            baseline_features: Optional baseline features for comparison
            
        Returns:
            Dictionary of cognitive decline indicators
        """
        try:
            features = {}
            
            # Motor speech control decline
            motor_features = self._extract_motor_decline_features(audio)
            features.update(motor_features)
            
            # Executive function decline indicators
            executive_features = self._extract_executive_decline_features(audio)
            features.update(executive_features)
            
            # Memory-related speech changes
            memory_features = self._extract_memory_related_features(audio)
            features.update(memory_features)
            
            # Attention and focus indicators
            attention_features = self._extract_attention_decline_features(audio)
            features.update(attention_features)
            
            # Language processing decline
            language_features = self._extract_language_decline_features(audio)
            features.update(language_features)
            
            # If baseline available, calculate decline ratios
            if baseline_features:
                decline_ratios = self._calculate_decline_ratios(features, baseline_features)
                features.update(decline_ratios)
            
            return features
            
        except Exception as e:
            return self._empty_cognitive_decline_features()
    
    def extract_mental_effort_markers(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract markers of increased mental effort and cognitive strain.
        
        Args:
            audio: Input audio array
            
        Returns:
            Dictionary of mental effort markers
        """
        try:
            features = {}
            
            # Voice strain indicators
            strain_features = self._extract_voice_strain_features(audio)
            features.update(strain_features)
            
            # Breathing pattern changes under cognitive load
            breathing_features = self._extract_breathing_effort_features(audio)
            features.update(breathing_features)
            
            # Articulation effort markers
            articulation_features = self._extract_articulation_effort_features(audio)
            features.update(articulation_features)
            
            # Temporal coordination under load
            coordination_features = self._extract_coordination_effort_features(audio)
            features.update(coordination_features)
            
            # Overall cognitive load score
            features["cognitive_load_composite"] = self._calculate_cognitive_load_score(features)
            
            return features
            
        except Exception as e:
            return self._empty_mental_effort_features()
    
    def extract_working_memory_indicators(self, audio: np.ndarray,
                                        task_complexity: Optional[str] = None) -> Dict[str, float]:
        """
        Extract indicators of working memory capacity and load.
        
        Args:
            audio: Input audio array
            task_complexity: Optional task complexity level ('low', 'medium', 'high')
            
        Returns:
            Dictionary of working memory indicators
        """
        try:
            features = {}
            
            # Dual-task performance indicators (speech + cognitive task)
            dual_task_features = self._extract_dual_task_features(audio)
            features.update(dual_task_features)
            
            # Working memory span indicators
            span_features = self._extract_memory_span_features(audio)
            features.update(span_features)
            
            # Cognitive interference effects
            interference_features = self._extract_interference_features(audio)
            features.update(interference_features)
            
            # Task-specific adjustments
            if task_complexity:
                complexity_features = self._adjust_for_task_complexity(features, task_complexity)
                features.update(complexity_features)
            
            return features
            
        except Exception as e:
            return self._empty_working_memory_features()
    
    def _extract_temporal_effort_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract temporal indicators of processing effort."""
        features = {}
        
        # Speech rate variability (higher variability may indicate effort)
        speech_segments = self._detect_speech_segments(audio)
        if speech_segments:
            segment_rates = []
            for start, end in speech_segments:
                segment_audio = audio[int(start * self.sample_rate):int(end * self.sample_rate)]
                if len(segment_audio) > 0:
                    # Estimate syllable rate as proxy for speech rate
                    syllable_rate = self._estimate_syllable_rate(segment_audio)
                    segment_rates.append(syllable_rate)
            
            if segment_rates:
                features["speech_rate_mean"] = np.mean(segment_rates)
                features["speech_rate_std"] = np.std(segment_rates)
                features["speech_rate_cv"] = np.std(segment_rates) / (np.mean(segment_rates) + 1e-10)
                features["speech_rate_variability"] = features["speech_rate_cv"]
            else:
                features.update({"speech_rate_mean": 0, "speech_rate_std": 0, 
                               "speech_rate_cv": 0, "speech_rate_variability": 0})
        
        # Pause pattern complexity
        pause_features = self._extract_pause_complexity(audio)
        features.update(pause_features)
        
        return features
    
    def _extract_spectral_complexity_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract spectral complexity indicators."""
        features = {}
        
        # Spectral entropy (higher entropy may indicate vocal effort/strain)
        spectral_entropy = self._calculate_spectral_entropy(audio)
        features["spectral_entropy_mean"] = np.mean(spectral_entropy)
        features["spectral_entropy_std"] = np.std(spectral_entropy)
        
        # Harmonic-to-noise ratio variability
        f0, _ = librosa.piptrack(y=audio, sr=self.sample_rate, hop_length=self.hop_length)
        hnr_values = []
        
        # Calculate HNR for windowed segments
        window_size = int(0.025 * self.sample_rate)  # 25ms windows
        hop_size = int(0.01 * self.sample_rate)      # 10ms hop
        
        for i in range(0, len(audio) - window_size, hop_size):
            window = audio[i:i + window_size]
            hnr = self._calculate_hnr(window)
            if not np.isnan(hnr):
                hnr_values.append(hnr)
        
        if hnr_values:
            features["hnr_mean"] = np.mean(hnr_values)
            features["hnr_std"] = np.std(hnr_values)
            features["hnr_variability"] = np.std(hnr_values) / (np.mean(hnr_values) + 1e-10)
        else:
            features.update({"hnr_mean": 0, "hnr_std": 0, "hnr_variability": 0})
        
        # Spectral flux (rate of spectral change - may indicate effort)
        stft = librosa.stft(audio, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        spectral_flux = np.sum(np.diff(magnitude, axis=1) ** 2, axis=0)
        
        features["spectral_flux_mean"] = np.mean(spectral_flux)
        features["spectral_flux_std"] = np.std(spectral_flux)
        
        return features
    
    def _extract_prosodic_effort_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract prosodic indicators of cognitive effort."""
        features = {}
        
        # F0 variability and control
        f0, voiced = librosa.piptrack(y=audio, sr=self.sample_rate, hop_length=self.hop_length)
        f0_values = []
        
        for t in range(f0.shape[1]):
            index = voiced[:, t].argmax()
            if voiced[index, t] > 0.1:
                f0_values.append(f0[index, t])
        
        if f0_values:
            features["f0_micro_variability"] = np.std(np.diff(f0_values))  # Frame-to-frame variability
            
            # F0 range compression (may indicate reduced vocal control)
            f0_range = np.ptp(f0_values)  # Peak-to-peak range
            f0_iqr = np.percentile(f0_values, 75) - np.percentile(f0_values, 25)
            features["f0_range"] = f0_range
            features["f0_range_compression"] = f0_iqr / (f0_range + 1e-10)
        else:
            features.update({"f0_micro_variability": 0, "f0_range": 0, "f0_range_compression": 0})
        
        # Energy control indicators
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        features["energy_variability"] = np.std(rms) / (np.mean(rms) + 1e-10)
        features["energy_micro_variability"] = np.std(np.diff(rms))
        
        return features
    
    def _extract_voice_quality_under_load(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract voice quality changes under cognitive load."""
        features = {}
        
        # Jitter and shimmer (voice stability)
        jitter = self._calculate_jitter(audio)
        shimmer = self._calculate_shimmer(audio)
        
        features["jitter"] = jitter
        features["shimmer"] = shimmer
        features["voice_instability"] = (jitter + shimmer) / 2
        
        # Breathiness indicators
        spectral_tilt = self._calculate_spectral_tilt(audio)
        features["spectral_tilt"] = spectral_tilt
        features["breathiness_indicator"] = max(0, -spectral_tilt)  # More negative tilt = more breathiness
        
        return features
    
    def _extract_text_effort_features(self, transcript: str) -> Dict[str, float]:
        """Extract text-based cognitive effort indicators."""
        features = {}
        
        words = transcript.split()
        if not words:
            return self._empty_text_effort_features()
        
        # Lexical complexity
        unique_words = len(set(words))
        features["lexical_diversity"] = unique_words / len(words)
        
        # Word length variability (may indicate word-finding difficulties)
        word_lengths = [len(word.strip('.,!?;:')) for word in words]
        features["word_length_mean"] = np.mean(word_lengths)
        features["word_length_std"] = np.std(word_lengths)
        
        # Semantic coherence approximation (repetition of content words)
        content_words = [word.lower() for word in words if len(word) > 3]
        if content_words:
            word_repetitions = len(content_words) - len(set(content_words))
            features["semantic_repetition_rate"] = word_repetitions / len(content_words)
        else:
            features["semantic_repetition_rate"] = 0
        
        return features
    
    def _extract_motor_decline_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract motor speech control decline indicators."""
        features = {}
        
        # Articulatory precision decline
        # Analyze consonant precision through spectral characteristics
        consonant_precision = self._analyze_consonant_precision(audio)
        features.update(consonant_precision)
        
        # Speech timing regularity
        timing_regularity = self._analyze_timing_regularity(audio)
        features.update(timing_regularity)
        
        return features
    
    def _extract_executive_decline_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract executive function decline indicators."""
        features = {}
        
        # Planning and organization in speech
        segment_organization = self._analyze_segment_organization(audio)
        features.update(segment_organization)
        
        # Inhibitory control (measured through speech consistency)
        inhibitory_control = self._analyze_inhibitory_control(audio)
        features.update(inhibitory_control)
        
        return features
    
    def _extract_memory_related_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract memory-related speech changes."""
        features = {}
        
        # Working memory load indicators
        wm_load = self._analyze_working_memory_load(audio)
        features.update(wm_load)
        
        # Retrieval effort markers
        retrieval_effort = self._analyze_retrieval_effort(audio)
        features.update(retrieval_effort)
        
        return features
    
    def _extract_attention_decline_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract attention and focus decline indicators."""
        features = {}
        
        # Sustained attention markers
        sustained_attention = self._analyze_sustained_attention(audio)
        features.update(sustained_attention)
        
        # Divided attention effects
        divided_attention = self._analyze_divided_attention_effects(audio)
        features.update(divided_attention)
        
        return features
    
    def _extract_language_decline_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract language processing decline indicators."""
        features = {}
        
        # Phonological processing changes
        phonological_features = self._analyze_phonological_changes(audio)
        features.update(phonological_features)
        
        # Lexical access difficulties
        lexical_features = self._analyze_lexical_access(audio)
        features.update(lexical_features)
        
        return features
    
    def _detect_speech_segments(self, audio: np.ndarray) -> List[Tuple[float, float]]:
        """Detect speech segments using energy-based VAD."""
        # Simple energy-based voice activity detection
        hop_length = 512
        frame_length = 2048
        
        # Calculate RMS energy
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Threshold for voice activity
        threshold = np.mean(rms) * 0.3
        
        # Find voiced segments
        voiced_frames = rms > threshold
        
        # Convert to time segments
        times = librosa.frames_to_time(np.arange(len(voiced_frames)), sr=self.sample_rate, hop_length=hop_length)
        
        segments = []
        in_speech = False
        start_time = 0
        
        for i, (time, is_voiced) in enumerate(zip(times, voiced_frames)):
            if is_voiced and not in_speech:
                start_time = time
                in_speech = True
            elif not is_voiced and in_speech:
                if time - start_time > 0.1:  # Minimum segment duration
                    segments.append((start_time, time))
                in_speech = False
        
        if in_speech:
            segments.append((start_time, times[-1]))
        
        return segments
    
    def _estimate_syllable_rate(self, audio: np.ndarray) -> float:
        """Estimate syllable rate from audio segment."""
        # Simple syllable rate estimation using energy peaks
        rms = librosa.feature.rms(y=audio, hop_length=512)[0]
        
        # Find peaks in energy (approximate syllable centers)
        peaks, _ = signal.find_peaks(rms, height=np.mean(rms) * 0.5, distance=5)
        
        # Calculate rate
        duration = len(audio) / self.sample_rate
        syllable_rate = len(peaks) / duration if duration > 0 else 0
        
        return syllable_rate
    
    def _extract_pause_complexity(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract pause pattern complexity features."""
        features = {}
        
        # Detect pause segments
        speech_segments = self._detect_speech_segments(audio)
        
        if len(speech_segments) < 2:
            return {"pause_complexity": 0, "pause_pattern_entropy": 0}
        
        # Calculate pause durations
        pauses = []
        for i in range(len(speech_segments) - 1):
            pause_start = speech_segments[i][1]
            pause_end = speech_segments[i + 1][0]
            pause_duration = pause_end - pause_start
            if pause_duration > 0.05:  # Minimum pause threshold
                pauses.append(pause_duration)
        
        if pauses:
            # Pause complexity as coefficient of variation
            features["pause_complexity"] = np.std(pauses) / (np.mean(pauses) + 1e-10)
            
            # Pause pattern entropy
            pause_bins = np.histogram(pauses, bins=5)[0]
            pause_probs = pause_bins / (np.sum(pause_bins) + 1e-10)
            entropy = -np.sum([p * np.log2(p + 1e-10) for p in pause_probs if p > 0])
            features["pause_pattern_entropy"] = entropy
        else:
            features.update({"pause_complexity": 0, "pause_pattern_entropy": 0})
        
        return features
    
    def _calculate_spectral_entropy(self, audio: np.ndarray) -> np.ndarray:
        """Calculate spectral entropy over time."""
        stft = librosa.stft(audio, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        
        # Normalize to get probability distribution
        magnitude_norm = magnitude / (np.sum(magnitude, axis=0, keepdims=True) + 1e-10)
        
        # Calculate entropy for each time frame
        entropy = -np.sum(magnitude_norm * np.log2(magnitude_norm + 1e-10), axis=0)
        
        return entropy
    
    def _calculate_hnr(self, window: np.ndarray) -> float:
        """Calculate Harmonic-to-Noise Ratio for a window."""
        try:
            # Autocorrelation-based HNR estimation
            autocorr = np.correlate(window, window, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Find the maximum in the expected F0 range (80-400 Hz)
            min_period = int(self.sample_rate / 400)  # samples
            max_period = int(self.sample_rate / 80)   # samples
            
            if max_period < len(autocorr):
                period_range = autocorr[min_period:max_period]
                if len(period_range) > 0:
                    max_autocorr = np.max(period_range)
                    hnr = 20 * np.log10(max_autocorr / (autocorr[0] - max_autocorr + 1e-10))
                    return hnr
            
            return 0.0
            
        except:
            return 0.0
    
    def _calculate_jitter(self, audio: np.ndarray) -> float:
        """Calculate jitter (period variability)."""
        try:
            # Extract F0 using autocorrelation
            autocorr = np.correlate(audio, audio, mode='full')
            autocorr = autocorr[len(autocorr) // 2:]
            
            # Find periods
            periods = []
            window_size = int(0.04 * self.sample_rate)  # 40ms windows
            
            for i in range(0, len(audio) - window_size, window_size // 2):
                window = audio[i:i + window_size]
                window_autocorr = np.correlate(window, window, mode='full')
                window_autocorr = window_autocorr[len(window_autocorr) // 2:]
                
                # Find peak in F0 range
                min_period = int(self.sample_rate / 400)
                max_period = int(self.sample_rate / 80)
                
                if max_period < len(window_autocorr):
                    period_range = window_autocorr[min_period:max_period]
                    if len(period_range) > 0:
                        peak_idx = np.argmax(period_range) + min_period
                        periods.append(peak_idx)
            
            if len(periods) > 1:
                period_diffs = np.diff(periods)
                jitter = np.std(period_diffs) / (np.mean(periods) + 1e-10)
                return jitter
            
            return 0.0
            
        except:
            return 0.0
    
    def _calculate_shimmer(self, audio: np.ndarray) -> float:
        """Calculate shimmer (amplitude variability)."""
        try:
            # Calculate frame-wise RMS
            frame_length = int(0.025 * self.sample_rate)
            hop_length = int(0.01 * self.sample_rate)
            
            rms_values = []
            for i in range(0, len(audio) - frame_length, hop_length):
                frame = audio[i:i + frame_length]
                rms = np.sqrt(np.mean(frame ** 2))
                if rms > 0:
                    rms_values.append(rms)
            
            if len(rms_values) > 1:
                rms_diffs = np.diff(rms_values)
                shimmer = np.std(rms_diffs) / (np.mean(rms_values) + 1e-10)
                return shimmer
            
            return 0.0
            
        except:
            return 0.0
    
    def _calculate_spectral_tilt(self, audio: np.ndarray) -> float:
        """Calculate spectral tilt (H1-H2 measure approximation)."""
        try:
            # Get magnitude spectrum
            fft = np.fft.fft(audio)
            magnitude = np.abs(fft[:len(fft) // 2])
            freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)[:len(fft) // 2]
            
            # Find low and high frequency energy
            low_freq_mask = (freqs >= 50) & (freqs <= 500)
            high_freq_mask = (freqs >= 2000) & (freqs <= 4000)
            
            low_energy = np.mean(magnitude[low_freq_mask]) if np.any(low_freq_mask) else 0
            high_energy = np.mean(magnitude[high_freq_mask]) if np.any(high_freq_mask) else 0
            
            # Calculate tilt (dB)
            if low_energy > 0 and high_energy > 0:
                tilt = 20 * np.log10(high_energy / low_energy)
                return tilt
            
            return 0.0
            
        except:
            return 0.0
    
    def _calculate_cognitive_load_score(self, features: Dict[str, float]) -> float:
        """Calculate composite cognitive load score."""
        # Weights for different feature categories (based on research)
        weights = {
            'voice_strain': 0.25,
            'temporal_variability': 0.20,
            'spectral_complexity': 0.20,
            'articulation_effort': 0.20,
            'coordination_difficulty': 0.15
        }
        
        # Normalize and combine features
        strain_score = features.get('voice_instability', 0) + features.get('breathiness_indicator', 0)
        temporal_score = features.get('speech_rate_variability', 0) + features.get('pause_complexity', 0)
        spectral_score = features.get('spectral_entropy_std', 0) + features.get('hnr_variability', 0)
        
        # Composite score
        composite_score = (
            weights['voice_strain'] * strain_score +
            weights['temporal_variability'] * temporal_score +
            weights['spectral_complexity'] * spectral_score
        )
        
        return min(1.0, composite_score)  # Normalize to [0, 1]
    
    # Placeholder methods for complex analyses (simplified implementations)
    def _analyze_consonant_precision(self, audio: np.ndarray) -> Dict[str, float]:
        """Analyze consonant precision markers."""
        # Simplified: analyze high-frequency energy stability
        stft = librosa.stft(audio, hop_length=self.hop_length)
        high_freq_energy = np.mean(np.abs(stft[stft.shape[0]//2:, :]), axis=0)
        
        return {
            "consonant_precision_mean": np.mean(high_freq_energy),
            "consonant_precision_variability": np.std(high_freq_energy) / (np.mean(high_freq_energy) + 1e-10)
        }
    
    def _analyze_timing_regularity(self, audio: np.ndarray) -> Dict[str, float]:
        """Analyze speech timing regularity."""
        segments = self._detect_speech_segments(audio)
        if len(segments) > 1:
            intervals = [segments[i][0] - segments[i-1][1] for i in range(1, len(segments))]
            regularity = 1.0 / (1.0 + np.std(intervals)) if intervals else 0
            return {"timing_regularity": regularity}
        return {"timing_regularity": 0}
    
    # Additional placeholder methods (simplified)
    def _analyze_segment_organization(self, audio: np.ndarray) -> Dict[str, float]:
        return {"segment_organization": 0.5}
    
    def _analyze_inhibitory_control(self, audio: np.ndarray) -> Dict[str, float]:
        return {"inhibitory_control": 0.5}
    
    def _analyze_working_memory_load(self, audio: np.ndarray) -> Dict[str, float]:
        return {"working_memory_load": 0.5}
    
    def _analyze_retrieval_effort(self, audio: np.ndarray) -> Dict[str, float]:
        return {"retrieval_effort": 0.5}
    
    def _analyze_sustained_attention(self, audio: np.ndarray) -> Dict[str, float]:
        return {"sustained_attention": 0.5}
    
    def _analyze_divided_attention_effects(self, audio: np.ndarray) -> Dict[str, float]:
        return {"divided_attention_effects": 0.5}
    
    def _analyze_phonological_changes(self, audio: np.ndarray) -> Dict[str, float]:
        return {"phonological_changes": 0.5}
    
    def _analyze_lexical_access(self, audio: np.ndarray) -> Dict[str, float]:
        return {"lexical_access_difficulty": 0.5}
    
    def _extract_voice_strain_features(self, audio: np.ndarray) -> Dict[str, float]:
        jitter = self._calculate_jitter(audio)
        shimmer = self._calculate_shimmer(audio)
        return {"voice_strain": (jitter + shimmer) / 2}
    
    def _extract_breathing_effort_features(self, audio: np.ndarray) -> Dict[str, float]:
        return {"breathing_effort": 0.3}
    
    def _extract_articulation_effort_features(self, audio: np.ndarray) -> Dict[str, float]:
        return {"articulation_effort": 0.4}
    
    def _extract_coordination_effort_features(self, audio: np.ndarray) -> Dict[str, float]:
        return {"coordination_effort": 0.3}
    
    def _extract_dual_task_features(self, audio: np.ndarray) -> Dict[str, float]:
        return {"dual_task_performance": 0.5}
    
    def _extract_memory_span_features(self, audio: np.ndarray) -> Dict[str, float]:
        return {"memory_span_indicator": 0.5}
    
    def _extract_interference_features(self, audio: np.ndarray) -> Dict[str, float]:
        return {"cognitive_interference": 0.5}
    
    def _adjust_for_task_complexity(self, features: Dict[str, float], complexity: str) -> Dict[str, float]:
        multiplier = {"low": 0.8, "medium": 1.0, "high": 1.2}.get(complexity, 1.0)
        return {f"adjusted_{k}": v * multiplier for k, v in features.items()}
    
    def _calculate_decline_ratios(self, current: Dict[str, float], baseline: Dict[str, float]) -> Dict[str, float]:
        ratios = {}
        for key in current:
            if key in baseline and baseline[key] != 0:
                ratios[f"{key}_decline_ratio"] = current[key] / baseline[key]
        return ratios
    
    # Empty feature dictionaries
    def _empty_processing_effort_features(self) -> Dict[str, float]:
        return {f"effort_feature_{i}": 0.0 for i in range(10)}
    
    def _empty_text_effort_features(self) -> Dict[str, float]:
        return {"lexical_diversity": 0, "word_length_mean": 0, "word_length_std": 0, "semantic_repetition_rate": 0}
    
    def _empty_cognitive_decline_features(self) -> Dict[str, float]:
        return {f"decline_feature_{i}": 0.0 for i in range(15)}
    
    def _empty_mental_effort_features(self) -> Dict[str, float]:
        return {f"mental_effort_{i}": 0.0 for i in range(8)}
    
    def _empty_working_memory_features(self) -> Dict[str, float]:
        return {f"working_memory_{i}": 0.0 for i in range(6)}
