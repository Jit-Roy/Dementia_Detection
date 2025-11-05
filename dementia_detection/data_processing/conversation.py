"""
Conversational Interaction Analysis Module
Extracts prosodic-interaction features including turn-taking behavior,
response patterns, and conversational dynamics for dementia detection.
"""

import librosa
import numpy as np
import scipy.stats
from typing import Dict, List, Tuple, Optional, Union

import re




class ConversationalInteractionAnalyzer:
    """Extract conversational and interaction features for dementia detection."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
    def extract_turn_taking_features(self, 
                                   speaker_segments: List[Dict], 
                                   total_duration: float) -> Dict[str, float]:
        """
        Extract turn-taking and conversation flow features.
        
        Args:
            speaker_segments: List of dicts with 'speaker', 'start', 'end', 'text'
            total_duration: Total conversation duration in seconds
            
        Returns:
            Dictionary of turn-taking features
        """
        try:
            features = {}
            
            if len(speaker_segments) == 0:
                return self._empty_turn_taking_features()
            
            # Basic turn statistics
            features["total_turns"] = len(speaker_segments)
            features["turns_per_minute"] = len(speaker_segments) / (total_duration / 60) if total_duration > 0 else 0
            
            # Turn durations
            turn_durations = [seg['end'] - seg['start'] for seg in speaker_segments]
            features["mean_turn_duration"] = np.mean(turn_durations)
            features["median_turn_duration"] = np.median(turn_durations)
            features["turn_duration_std"] = np.std(turn_durations)
            features["turn_duration_cv"] = np.std(turn_durations) / np.mean(turn_durations) if np.mean(turn_durations) > 0 else 0
            
            # Turn length distribution
            features["short_turns_ratio"] = sum(1 for d in turn_durations if d < 2.0) / len(turn_durations)
            features["long_turns_ratio"] = sum(1 for d in turn_durations if d > 10.0) / len(turn_durations)
            
            # Speaker-specific analysis (if multi-speaker)
            speakers = list(set(seg['speaker'] for seg in speaker_segments))
            if len(speakers) > 1:
                features["num_speakers"] = len(speakers)
                
                # Dominance patterns
                speaker_times = {speaker: sum(seg['end'] - seg['start'] for seg in speaker_segments 
                                            if seg['speaker'] == speaker) for speaker in speakers}
                
                max_speaker_time = max(speaker_times.values())
                min_speaker_time = min(speaker_times.values())
                features["speaker_dominance_ratio"] = max_speaker_time / (min_speaker_time + 1e-10)
                features["conversation_balance"] = min_speaker_time / max_speaker_time if max_speaker_time > 0 else 0
                
                # Turn transitions
                transitions = []
                for i in range(len(speaker_segments) - 1):
                    current_speaker = speaker_segments[i]['speaker']
                    next_speaker = speaker_segments[i + 1]['speaker']
                    transition_gap = speaker_segments[i + 1]['start'] - speaker_segments[i]['end']
                    
                    if current_speaker != next_speaker:
                        transitions.append(transition_gap)
                
                if transitions:
                    features["turn_transition_count"] = len(transitions)
                    features["mean_transition_gap"] = np.mean(transitions)
                    features["transition_gap_std"] = np.std(transitions)
                    features["smooth_transitions_ratio"] = sum(1 for t in transitions if 0 < t < 0.5) / len(transitions)
                    features["interrupted_turns_ratio"] = sum(1 for t in transitions if t < 0) / len(transitions)  # Overlaps
                    features["delayed_responses_ratio"] = sum(1 for t in transitions if t > 2.0) / len(transitions)
                else:
                    features.update(self._empty_transition_features())
            else:
                features["num_speakers"] = 1
                features.update(self._empty_multi_speaker_features())
            
            return features
            
        except Exception as e:
            
            return self._empty_turn_taking_features()
    
    def extract_response_latency_features(self, 
                                        question_response_pairs: List[Tuple[Dict, Dict]]) -> Dict[str, float]:
        """
        Extract response latency and timing features.
        
        Args:
            question_response_pairs: List of (question_segment, response_segment) tuples
            
        Returns:
            Dictionary of response latency features
        """
        try:
            features = {}
            
            if len(question_response_pairs) == 0:
                return self._empty_response_features()
            
            # Calculate response latencies
            latencies = []
            response_durations = []
            
            for question, response in question_response_pairs:
                latency = response['start'] - question['end']
                duration = response['end'] - response['start']
                
                latencies.append(latency)
                response_durations.append(duration)
            
            # Response latency statistics
            features["num_qa_pairs"] = len(question_response_pairs)
            features["mean_response_latency"] = np.mean(latencies)
            features["median_response_latency"] = np.median(latencies)
            features["response_latency_std"] = np.std(latencies)
            features["response_latency_cv"] = np.std(latencies) / np.mean(latencies) if np.mean(latencies) > 0 else 0
            
            # Response timing categories
            features["quick_responses_ratio"] = sum(1 for l in latencies if l < 0.5) / len(latencies)
            features["normal_responses_ratio"] = sum(1 for l in latencies if 0.5 <= l <= 2.0) / len(latencies)
            features["delayed_responses_ratio"] = sum(1 for l in latencies if l > 2.0) / len(latencies)
            features["very_delayed_responses_ratio"] = sum(1 for l in latencies if l > 5.0) / len(latencies)
            
            # Response duration analysis
            features["mean_response_duration"] = np.mean(response_durations)
            features["response_duration_std"] = np.std(response_durations)
            features["response_duration_cv"] = np.std(response_durations) / np.mean(response_durations) if np.mean(response_durations) > 0 else 0
            
            # Response consistency measures
            features["response_consistency"] = 1.0 / (1.0 + features["response_latency_cv"])
            features["response_predictability"] = np.exp(-features["response_latency_std"])
            
            return features
            
        except Exception as e:
            
            return self._empty_response_features()
    
    def extract_hesitation_patterns(self, audio: np.ndarray, 
                                  transcript: Optional[str] = None) -> Dict[str, float]:
        """
        Extract hesitation and disfluency patterns.
        
        Args:
            audio: Input audio array
            transcript: Optional transcript text
            
        Returns:
            Dictionary of hesitation features
        """
        try:
            features = {}
            
            # Audio-based hesitation detection
            hesitation_segments = self._detect_audio_hesitations(audio)
            
            # Basic hesitation statistics
            features["num_hesitations"] = len(hesitation_segments)
            features["hesitation_rate"] = len(hesitation_segments) / (len(audio) / self.sample_rate) if len(audio) > 0 else 0
            
            if len(hesitation_segments) > 0:
                hesitation_durations = [end - start for start, end in hesitation_segments]
                features["mean_hesitation_duration"] = np.mean(hesitation_durations)
                features["total_hesitation_time"] = sum(hesitation_durations)
                features["hesitation_time_ratio"] = features["total_hesitation_time"] / (len(audio) / self.sample_rate)
                
                # Hesitation patterns
                inter_hesitation_intervals = []
                for i in range(len(hesitation_segments) - 1):
                    interval = hesitation_segments[i + 1][0] - hesitation_segments[i][1]
                    inter_hesitation_intervals.append(interval)
                
                if inter_hesitation_intervals:
                    features["mean_inter_hesitation_interval"] = np.mean(inter_hesitation_intervals)
                    features["hesitation_clustering"] = 1.0 / (1.0 + np.mean(inter_hesitation_intervals))
                else:
                    features["mean_inter_hesitation_interval"] = 0
                    features["hesitation_clustering"] = 0
            else:
                features.update({
                    "mean_hesitation_duration": 0, "total_hesitation_time": 0,
                    "hesitation_time_ratio": 0, "mean_inter_hesitation_interval": 0,
                    "hesitation_clustering": 0
                })
            
            # Text-based hesitation analysis (if transcript available)
            if transcript:
                text_hesitations = self._analyze_text_hesitations(transcript)
                features.update(text_hesitations)
            else:
                features.update(self._empty_text_hesitation_features())
            
            return features
            
        except Exception as e:
            
            return self._empty_hesitation_features()
    
    def extract_interruption_patterns(self, speaker_segments: List[Dict]) -> Dict[str, float]:
        """
        Extract interruption and overlap patterns.
        
        Args:
            speaker_segments: List of speaker segment dictionaries
            
        Returns:
            Dictionary of interruption features
        """
        try:
            features = {}
            
            if len(speaker_segments) < 2:
                return self._empty_interruption_features()
            
            # Detect overlaps and interruptions
            overlaps = []
            interruptions = []
            
            for i in range(len(speaker_segments)):
                current_seg = speaker_segments[i]
                
                # Check for overlaps with other segments
                for j in range(len(speaker_segments)):
                    if i != j:
                        other_seg = speaker_segments[j]
                        
                        # Check for temporal overlap
                        overlap_start = max(current_seg['start'], other_seg['start'])
                        overlap_end = min(current_seg['end'], other_seg['end'])
                        
                        if overlap_end > overlap_start:  # There is an overlap
                            overlap_duration = overlap_end - overlap_start
                            overlaps.append({
                                'duration': overlap_duration,
                                'speaker1': current_seg['speaker'],
                                'speaker2': other_seg['speaker'],
                                'start': overlap_start,
                                'end': overlap_end
                            })
                            
                            # Determine if it's an interruption (one speaker starts before the other ends)
                            if (current_seg['start'] > other_seg['start'] and 
                                current_seg['start'] < other_seg['end'] and
                                current_seg['speaker'] != other_seg['speaker']):
                                interruptions.append({
                                    'interrupter': current_seg['speaker'],
                                    'interrupted': other_seg['speaker'],
                                    'time': current_seg['start']
                                })
            
            # Remove duplicate overlaps (same overlap detected from both directions)
            unique_overlaps = []
            for overlap in overlaps:
                is_duplicate = any(
                    abs(existing['start'] - overlap['start']) < 0.1 and
                    abs(existing['end'] - overlap['end']) < 0.1 and
                    {existing['speaker1'], existing['speaker2']} == {overlap['speaker1'], overlap['speaker2']}
                    for existing in unique_overlaps
                )
                if not is_duplicate:
                    unique_overlaps.append(overlap)
            
            # Overlap statistics
            features["num_overlaps"] = len(unique_overlaps)
            features["num_interruptions"] = len(interruptions)
            
            if unique_overlaps:
                overlap_durations = [o['duration'] for o in unique_overlaps]
                features["total_overlap_time"] = sum(overlap_durations)
                features["mean_overlap_duration"] = np.mean(overlap_durations)
                features["overlap_rate"] = len(unique_overlaps) / (len(speaker_segments) / 2) if len(speaker_segments) > 0 else 0
            else:
                features.update({
                    "total_overlap_time": 0, "mean_overlap_duration": 0, "overlap_rate": 0
                })
            
            # Interruption patterns
            if interruptions:
                features["interruption_rate"] = len(interruptions) / len(speaker_segments) if len(speaker_segments) > 0 else 0
                
                # Speaker-specific interruption patterns
                speakers = list(set(seg['speaker'] for seg in speaker_segments))
                if len(speakers) > 1:
                    interrupter_counts = {}
                    for interruption in interruptions:
                        interrupter = interruption['interrupter']
                        interrupter_counts[interrupter] = interrupter_counts.get(interrupter, 0) + 1
                    
                    if interrupter_counts:
                        features["dominant_interrupter_ratio"] = max(interrupter_counts.values()) / len(interruptions)
                    else:
                        features["dominant_interrupter_ratio"] = 0
                else:
                    features["dominant_interrupter_ratio"] = 0
            else:
                features["interruption_rate"] = 0
                features["dominant_interrupter_ratio"] = 0
            
            return features
            
        except Exception as e:
            
            return self._empty_interruption_features()
    
    def _detect_audio_hesitations(self, audio: np.ndarray) -> List[Tuple[float, float]]:
        """Detect hesitation segments in audio based on energy and spectral characteristics."""
        try:
            hop_length = 512
            
            # Calculate features for hesitation detection
            rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate, hop_length=hop_length)[0]
            zcr = librosa.feature.zero_crossing_rate(audio, hop_length=hop_length)[0]
            
            # Hesitation characteristics: low energy, low spectral centroid, high ZCR variability
            low_energy_threshold = np.mean(rms) - 0.5 * np.std(rms)
            low_centroid_threshold = np.mean(spectral_centroid) - 0.3 * np.std(spectral_centroid)
            high_zcr_variability = np.std(zcr) > np.mean(np.std(zcr))
            
            # Potential hesitation frames
            hesitation_frames = (rms < low_energy_threshold) | (spectral_centroid < low_centroid_threshold)
            
            # Convert to time segments
            times = librosa.frames_to_time(np.arange(len(hesitation_frames)), sr=self.sample_rate, hop_length=hop_length)
            
            segments = []
            in_hesitation = False
            start_time = 0
            
            for i, (time, is_hesitation) in enumerate(zip(times, hesitation_frames)):
                if is_hesitation and not in_hesitation:
                    start_time = time
                    in_hesitation = True
                elif not is_hesitation and in_hesitation:
                    if time - start_time > 0.1:  # Minimum hesitation duration
                        segments.append((start_time, time))
                    in_hesitation = False
            
            if in_hesitation and times[-1] - start_time > 0.1:
                segments.append((start_time, times[-1]))
            
            return segments
            
        except Exception as e:
            
            return []
    
    def _analyze_text_hesitations(self, transcript: str) -> Dict[str, float]:
        """Analyze hesitations in transcript text."""
        try:
            features = {}
            
            # Count filled pauses
            filled_pauses = len(re.findall(r'\b(um|uh|er|ah|mm|hmm)\b', transcript.lower()))
            
            # Count repetitions
            words = transcript.lower().split()
            repetitions = 0
            for i in range(len(words) - 1):
                if words[i] == words[i + 1]:
                    repetitions += 1
            
            # Count false starts (approximation: words followed by similar words)
            false_starts = 0
            for i in range(len(words) - 1):
                if len(words[i]) > 2 and len(words[i + 1]) > 2:
                    # Check if next word starts with same letters (potential false start)
                    if words[i][:2] == words[i + 1][:2] and words[i] != words[i + 1]:
                        false_starts += 1
            
            # Calculate rates
            total_words = len(words) if words else 1
            features["filled_pause_rate"] = filled_pauses / total_words
            features["repetition_rate"] = repetitions / total_words
            features["false_start_rate"] = false_starts / total_words
            
            # Overall disfluency rate
            features["total_disfluencies"] = filled_pauses + repetitions + false_starts
            features["disfluency_rate"] = features["total_disfluencies"] / total_words
            
            return features
            
        except Exception as e:
            
            return self._empty_text_hesitation_features()
    
    def _empty_turn_taking_features(self) -> Dict[str, float]:
        """Return empty turn-taking features."""
        base_features = {
            "total_turns": 0, "turns_per_minute": 0, "mean_turn_duration": 0,
            "median_turn_duration": 0, "turn_duration_std": 0, "turn_duration_cv": 0,
            "short_turns_ratio": 0, "long_turns_ratio": 0, "num_speakers": 0
        }
        base_features.update(self._empty_multi_speaker_features())
        return base_features
    
    def _empty_multi_speaker_features(self) -> Dict[str, float]:
        """Return empty multi-speaker features."""
        features = {
            "speaker_dominance_ratio": 0, "conversation_balance": 0
        }
        features.update(self._empty_transition_features())
        return features
    
    def _empty_transition_features(self) -> Dict[str, float]:
        """Return empty transition features."""
        return {
            "turn_transition_count": 0, "mean_transition_gap": 0, "transition_gap_std": 0,
            "smooth_transitions_ratio": 0, "interrupted_turns_ratio": 0, "delayed_responses_ratio": 0
        }
    
    def _empty_response_features(self) -> Dict[str, float]:
        """Return empty response latency features."""
        return {
            "num_qa_pairs": 0, "mean_response_latency": 0, "median_response_latency": 0,
            "response_latency_std": 0, "response_latency_cv": 0, "quick_responses_ratio": 0,
            "normal_responses_ratio": 0, "delayed_responses_ratio": 0, "very_delayed_responses_ratio": 0,
            "mean_response_duration": 0, "response_duration_std": 0, "response_duration_cv": 0,
            "response_consistency": 0, "response_predictability": 0
        }
    
    def _empty_hesitation_features(self) -> Dict[str, float]:
        """Return empty hesitation features."""
        base_features = {
            "num_hesitations": 0, "hesitation_rate": 0, "mean_hesitation_duration": 0,
            "total_hesitation_time": 0, "hesitation_time_ratio": 0, "mean_inter_hesitation_interval": 0,
            "hesitation_clustering": 0
        }
        base_features.update(self._empty_text_hesitation_features())
        return base_features
    
    def _empty_text_hesitation_features(self) -> Dict[str, float]:
        """Return empty text hesitation features."""
        return {
            "filled_pause_rate": 0, "repetition_rate": 0, "false_start_rate": 0,
            "total_disfluencies": 0, "disfluency_rate": 0
        }
    
    def _empty_interruption_features(self) -> Dict[str, float]:
        """Return empty interruption features."""
        return {
            "num_overlaps": 0, "num_interruptions": 0, "total_overlap_time": 0,
            "mean_overlap_duration": 0, "overlap_rate": 0, "interruption_rate": 0,
            "dominant_interrupter_ratio": 0
        }
