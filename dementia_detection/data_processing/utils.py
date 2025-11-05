"""
Shared utility functions for data_processing modules
Provides common helper functions to avoid code duplication
"""

import numpy as np
from typing import List, Tuple
import librosa
import re
from nltk.tokenize import sent_tokenize, word_tokenize


def detect_speech_segments(audio: np.ndarray, sample_rate: int = 16000) -> List[Tuple[float, float]]:
    """
    Detect speech segments in audio using energy-based VAD.
    
    Args:
        audio: Audio signal
        sample_rate: Sample rate of audio
        
    Returns:
        List of (start_time, end_time) tuples for speech segments
    """
    try:
        hop_length = 512
        rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        
        # Speech detection threshold
        threshold = np.mean(rms) + 0.2 * np.std(rms)
        speech_frames = rms > threshold
        
        # Convert to time segments
        times = librosa.frames_to_time(
            np.arange(len(speech_frames)), 
            sr=sample_rate, 
            hop_length=hop_length
        )
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
        # Return full audio duration as single segment
        return [(0.0, len(audio) / sample_rate)]


def estimate_syllable_rate(audio: np.ndarray, sample_rate: int = 16000) -> float:
    """
    Estimate syllable rate from audio using onset detection.
    
    Args:
        audio: Audio signal
        sample_rate: Sample rate of audio
        
    Returns:
        Syllable rate in syllables per second
    """
    try:
        onset_frames = librosa.onset.onset_detect(
            y=audio, 
            sr=sample_rate,
            hop_length=512
        )
        duration = len(audio) / sample_rate
        return len(onset_frames) / duration if duration > 0 else 0.0
    except:
        return 0.0


def segment_sentences(text: str) -> List[str]:
    """
    Segment text into sentences using NLTK.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    try:
        return sent_tokenize(text)
    except:
        # Fallback to simple rule-based segmentation
        return re.split(r'[.!?]+', text)


def tokenize_words(text: str) -> List[str]:
    """
    Tokenize text into words using NLTK.
    
    Args:
        text: Input text
        
    Returns:
        List of lowercase alphabetic words
    """
    try:
        return [word.lower() for word in word_tokenize(text) if word.isalpha()]
    except:
        # Fallback to simple whitespace tokenization
        return [word.lower() for word in text.split() if word.isalpha()]


def calculate_pause_statistics(segments: List[Tuple[float, float]]) -> dict:
    """
    Calculate pause statistics from speech segments.
    
    Args:
        segments: List of (start_time, end_time) tuples
        
    Returns:
        Dictionary with pause statistics
    """
    if len(segments) < 2:
        return {
            'num_pauses': 0,
            'mean_pause_duration': 0.0,
            'std_pause_duration': 0.0,
            'max_pause_duration': 0.0,
            'min_pause_duration': 0.0
        }
    
    # Calculate pauses between segments
    pauses = []
    for i in range(len(segments) - 1):
        pause_duration = segments[i + 1][0] - segments[i][1]
        if pause_duration > 0:
            pauses.append(pause_duration)
    
    if not pauses:
        return {
            'num_pauses': 0,
            'mean_pause_duration': 0.0,
            'std_pause_duration': 0.0,
            'max_pause_duration': 0.0,
            'min_pause_duration': 0.0
        }
    
    return {
        'num_pauses': len(pauses),
        'mean_pause_duration': float(np.mean(pauses)),
        'std_pause_duration': float(np.std(pauses)),
        'max_pause_duration': float(np.max(pauses)),
        'min_pause_duration': float(np.min(pauses))
    }


def normalize_features(features: dict) -> dict:
    """
    Normalize feature values to handle NaN and infinite values.
    
    Args:
        features: Dictionary of features
        
    Returns:
        Dictionary with normalized features
    """
    normalized = {}
    for key, value in features.items():
        if isinstance(value, (int, float)):
            if np.isnan(value) or np.isinf(value):
                normalized[key] = 0.0
            else:
                normalized[key] = float(value)
        else:
            normalized[key] = value
    return normalized
