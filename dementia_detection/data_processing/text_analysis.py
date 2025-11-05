"""
Text Analysis Pipeline Module
Handles automatic speech recognition using Whisper and text preprocessing
including tokenization, sentence segmentation, and cleaning.
"""

import whisper
import torch
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
import nltk
import spacy
from transformers import pipeline
from pathlib import Path

from ..config.settings import TEXT_CONFIG

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)


class TextAnalysisPipeline:
    """Complete text analysis pipeline including ASR and preprocessing."""
    
    def __init__(self, whisper_model: str = TEXT_CONFIG["whisper_model"]):
        """
        Initialize TextAnalysisPipeline.
        
        Args:
            whisper_model: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        """
        self.whisper_model_name = whisper_model
        self.whisper_model = None
        self.nlp = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize models
        self._load_models()
        
    def _load_models(self):
        """Load ASR and NLP models."""
        try:
            # Load Whisper model
            
            self.whisper_model = whisper.load_model(self.whisper_model_name)
            
            # Load spaCy model
            
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                
                self.nlp = None
                
        except Exception as e:
            
            raise
    
    def transcribe_audio(self, audio: np.ndarray, 
                        sample_rate: int = 16000,
                        language: str = TEXT_CONFIG["language"],
                        return_segments: bool = True) -> Dict:
        """
        Transcribe audio using Whisper ASR.
        
        Args:
            audio: Input audio array
            sample_rate: Audio sample rate
            language: Language code (e.g., 'en', 'es', 'fr')
            return_segments: Whether to return word-level segments
            
        Returns:
            Dictionary containing transcription and metadata
        """
        try:
            if self.whisper_model is None:
                raise RuntimeError("Whisper model not loaded")
            
            # Ensure audio is float32 and normalized
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
                
            # Normalize audio to [-1, 1] range
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val
            
            # Transcribe
            result = self.whisper_model.transcribe(
                audio,
                language=language if language != 'auto' else None,
                word_timestamps=return_segments,
                verbose=False
            )
            
            # Extract transcription details
            transcription_result = {
                "text": result["text"].strip(),
                "language": result.get("language", language),
                "segments": [],
                "word_count": 0,
                "total_duration": len(audio) / sample_rate,
                "speaking_rate": 0,
                "confidence_scores": []
            }
            
            # Process segments if available
            if "segments" in result and return_segments:
                for segment in result["segments"]:
                    segment_info = {
                        "start": segment.get("start", 0),
                        "end": segment.get("end", 0),
                        "text": segment.get("text", "").strip(),
                        "words": []
                    }
                    
                    # Process words if available
                    if "words" in segment:
                        for word in segment["words"]:
                            word_info = {
                                "word": word.get("word", "").strip(),
                                "start": word.get("start", 0),
                                "end": word.get("end", 0),
                                "probability": word.get("probability", 0.0)
                            }
                            segment_info["words"].append(word_info)
                            
                            # Collect confidence scores
                            if word_info["probability"] > 0:
                                transcription_result["confidence_scores"].append(word_info["probability"])
                    
                    transcription_result["segments"].append(segment_info)
            
            # Calculate word count and speaking rate
            words = self._tokenize_words(transcription_result["text"])
            transcription_result["word_count"] = len(words)
            
            if transcription_result["total_duration"] > 0:
                transcription_result["speaking_rate"] = transcription_result["word_count"] / (transcription_result["total_duration"] / 60)
            
            # Calculate average confidence
            if transcription_result["confidence_scores"]:
                transcription_result["average_confidence"] = np.mean(transcription_result["confidence_scores"])
            else:
                transcription_result["average_confidence"] = 0.0
            
            
            
            return transcription_result
            
        except Exception as e:
            
            return {
                "text": "", "language": language, "segments": [], 
                "word_count": 0, "total_duration": len(audio) / sample_rate if len(audio) > 0 else 0,
                "speaking_rate": 0, "confidence_scores": [], "average_confidence": 0.0
            }
    
    def preprocess_text(self, text: str, 
                       clean_text: bool = True,
                       normalize_whitespace: bool = True,
                       remove_filler_words: bool = True) -> Dict:
        """
        Preprocess transcribed text.
        
        Args:
            text: Input text
            clean_text: Whether to clean special characters
            normalize_whitespace: Whether to normalize whitespace
            remove_filler_words: Whether to remove filler words
            
        Returns:
            Dictionary containing preprocessed text and metadata
        """
        try:
            original_text = text
            processed_text = text
            
            # Basic cleaning
            if clean_text:
                # Remove extra whitespace
                if normalize_whitespace:
                    processed_text = re.sub(r'\s+', ' ', processed_text)
                    processed_text = processed_text.strip()
                
                # Remove special characters but keep punctuation
                processed_text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\'\"]', '', processed_text)
                
                # Fix punctuation spacing
                processed_text = re.sub(r'\s*([\.!\?])\s*', r'\1 ', processed_text)
                processed_text = re.sub(r'\s*([,:;])\s*', r'\1 ', processed_text)
            
            # Remove filler words commonly found in speech
            if remove_filler_words:
                filler_words = [
                    'um', 'uh', 'er', 'ah', 'like', 'you know', 
                    'so', 'well', 'basically', 'actually', 'literally'
                ]
                
                filler_pattern = r'\b(?:' + '|'.join(re.escape(word) for word in filler_words) + r')\b'
                processed_text = re.sub(filler_pattern, '', processed_text, flags=re.IGNORECASE)
                processed_text = re.sub(r'\s+', ' ', processed_text).strip()
            
            # Tokenization and segmentation
            sentences = self._segment_sentences(processed_text)
            words = self._tokenize_words(processed_text)
            
            # Calculate preprocessing statistics
            preprocessing_stats = {
                "original_length": len(original_text),
                "processed_length": len(processed_text),
                "compression_ratio": len(processed_text) / len(original_text) if len(original_text) > 0 else 0,
                "num_sentences": len(sentences),
                "num_words": len(words),
                "avg_sentence_length": np.mean([len(self._tokenize_words(s)) for s in sentences]) if sentences else 0,
                "avg_word_length": np.mean([len(word) for word in words]) if words else 0
            }
            
            result = {
                "original_text": original_text,
                "processed_text": processed_text,
                "sentences": sentences,
                "words": words,
                "stats": preprocessing_stats
            }
            
            return result
            
        except Exception as e:
            
            return {
                "original_text": text,
                "processed_text": text,
                "sentences": [text] if text else [],
                "words": text.split() if text else [],
                "stats": {"original_length": len(text), "processed_length": len(text), 
                         "compression_ratio": 1.0, "num_sentences": 1 if text else 0,
                         "num_words": len(text.split()) if text else 0,
                         "avg_sentence_length": len(text.split()) if text else 0,
                         "avg_word_length": 0}
            }
    
    def analyze_linguistic_structure(self, text: str) -> Dict:
        """
        Analyze linguistic structure using spaCy NLP.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing linguistic analysis
        """
        try:
            if self.nlp is None:
                
                return self._basic_linguistic_analysis(text)
            
            doc = self.nlp(text)
            
            # Basic linguistic features
            analysis = {
                "num_tokens": len(doc),
                "num_sentences": len(list(doc.sents)),
                "num_words": len([token for token in doc if token.is_alpha]),
                "num_punctuation": len([token for token in doc if token.is_punct]),
                "num_spaces": len([token for token in doc if token.is_space]),
            }
            
            # Part-of-speech analysis
            pos_counts = {}
            for token in doc:
                if token.pos_ not in pos_counts:
                    pos_counts[token.pos_] = 0
                pos_counts[token.pos_] += 1
            
            analysis["pos_counts"] = pos_counts
            
            # Calculate POS ratios
            total_words = analysis["num_words"]
            if total_words > 0:
                analysis["noun_ratio"] = pos_counts.get("NOUN", 0) / total_words
                analysis["verb_ratio"] = pos_counts.get("VERB", 0) / total_words
                analysis["adj_ratio"] = pos_counts.get("ADJ", 0) / total_words
                analysis["adv_ratio"] = pos_counts.get("ADV", 0) / total_words
                analysis["pronoun_ratio"] = pos_counts.get("PRON", 0) / total_words
            else:
                analysis.update({"noun_ratio": 0, "verb_ratio": 0, "adj_ratio": 0, 
                               "adv_ratio": 0, "pronoun_ratio": 0})
            
            # Named entity analysis
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            analysis["num_entities"] = len(entities)
            analysis["entities"] = entities
            
            # Entity type counts
            entity_counts = {}
            for _, ent_type in entities:
                if ent_type not in entity_counts:
                    entity_counts[ent_type] = 0
                entity_counts[ent_type] += 1
            analysis["entity_counts"] = entity_counts
            
            # Dependency parsing features
            dependency_labels = [token.dep_ for token in doc]
            analysis["dependency_complexity"] = len(set(dependency_labels))
            analysis["avg_dependency_distance"] = np.mean([
                abs(token.i - token.head.i) for token in doc if token.head != token
            ]) if len(doc) > 1 else 0
            
            # Sentence complexity
            sentence_lengths = [len(sent) for sent in doc.sents]
            analysis["avg_sentence_length"] = np.mean(sentence_lengths) if sentence_lengths else 0
            analysis["max_sentence_length"] = max(sentence_lengths) if sentence_lengths else 0
            analysis["sentence_length_std"] = np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0
            
            return analysis
            
        except Exception as e:
            
            return self._basic_linguistic_analysis(text)
    
    def detect_speech_errors(self, original_audio_text: str, 
                           processed_text: str,
                           word_timestamps: List[Dict] = None) -> Dict:
        """
        Detect speech errors and disfluencies.
        
        Args:
            original_audio_text: Raw transcription from ASR
            processed_text: Cleaned/processed text
            word_timestamps: Word-level timing information
            
        Returns:
            Dictionary containing error analysis
        """
        try:
            error_analysis = {
                "repetitions": 0,
                "self_corrections": 0,
                "incomplete_words": 0,
                "false_starts": 0,
                "filled_pauses": 0,
                "word_fragments": 0,
                "semantic_errors": 0
            }
            
            # Detect repetitions
            words = original_audio_text.lower().split()
            for i in range(len(words) - 1):
                if words[i] == words[i + 1] and len(words[i]) > 2:
                    error_analysis["repetitions"] += 1
            
            # Detect filled pauses and disfluencies
            filler_patterns = [
                r'\b(um|uh|er|ah)\b',
                r'\b(like|you know)\b',
                r'\b(well|so|basically)\b'
            ]
            
            for pattern in filler_patterns:
                matches = re.findall(pattern, original_audio_text.lower())
                error_analysis["filled_pauses"] += len(matches)
            
            # Detect incomplete words (words shorter than 2 chars or ending with -)
            incomplete_pattern = r'\b\w{1}(?:\s|$)|\w+-(?:\s|$)'
            incomplete_matches = re.findall(incomplete_pattern, original_audio_text)
            error_analysis["incomplete_words"] = len(incomplete_matches)
            
            # Detect self-corrections (word followed by similar but different word)
            words = original_audio_text.split()
            for i in range(len(words) - 1):
                word1, word2 = words[i].lower(), words[i + 1].lower()
                if (len(word1) > 2 and len(word2) > 2 and 
                    word1 != word2 and 
                    self._words_similar(word1, word2)):
                    error_analysis["self_corrections"] += 1
            
            # False starts detection (short utterances followed by restarts)
            sentences = re.split(r'[.!?]+', original_audio_text)
            for sentence in sentences:
                words_in_sent = sentence.strip().split()
                if 1 <= len(words_in_sent) <= 3:
                    error_analysis["false_starts"] += 1
            
            # Calculate error rates
            total_words = len(original_audio_text.split())
            if total_words > 0:
                error_analysis["repetition_rate"] = error_analysis["repetitions"] / total_words
                error_analysis["disfluency_rate"] = error_analysis["filled_pauses"] / total_words
                error_analysis["error_rate"] = (
                    error_analysis["repetitions"] + 
                    error_analysis["incomplete_words"] + 
                    error_analysis["false_starts"]
                ) / total_words
            else:
                error_analysis.update({"repetition_rate": 0, "disfluency_rate": 0, "error_rate": 0})
            
            return error_analysis
            
        except Exception as e:
            
            return {"repetitions": 0, "self_corrections": 0, "incomplete_words": 0,
                   "false_starts": 0, "filled_pauses": 0, "word_fragments": 0,
                   "semantic_errors": 0, "repetition_rate": 0, "disfluency_rate": 0, "error_rate": 0}
    
    def process_audio_to_text(self, audio: np.ndarray, 
                             sample_rate: int = 16000) -> Dict:
        """
        Complete pipeline from audio to processed text with analysis.
        
        Args:
            audio: Input audio array
            sample_rate: Audio sample rate
            
        Returns:
            Dictionary containing complete text analysis
        """
        try:
            # Step 1: Transcribe audio
            transcription = self.transcribe_audio(audio, sample_rate)
            
            if not transcription["text"]:
                
                return self._empty_text_result()
            
            # Step 2: Preprocess text
            preprocessing = self.preprocess_text(transcription["text"])
            
            # Step 3: Linguistic analysis
            linguistic_analysis = self.analyze_linguistic_structure(preprocessing["processed_text"])
            
            # Step 4: Error detection
            error_analysis = self.detect_speech_errors(
                transcription["text"],
                preprocessing["processed_text"],
                transcription.get("segments", [])
            )
            
            # Combine all results
            result = {
                "transcription": transcription,
                "preprocessing": preprocessing,
                "linguistic_analysis": linguistic_analysis,
                "error_analysis": error_analysis,
                "processing_success": True
            }
            
            return result
            
        except Exception as e:
            return self._empty_text_result()
    
    def _segment_sentences(self, text: str) -> List[str]:
        """Segment text into sentences using NLTK."""
        try:
            from nltk.tokenize import sent_tokenize
            return sent_tokenize(text)
        except Exception:
            # Fallback to simple rule-based segmentation
            return re.split(r'[.!?]+', text)
    
    def _tokenize_words(self, text: str) -> List[str]:
        """Tokenize text into words using NLTK."""
        try:
            from nltk.tokenize import word_tokenize
            return [word.lower() for word in word_tokenize(text) if word.isalpha()]
        except Exception:
            # Fallback to simple whitespace tokenization
            return [word.lower() for word in text.split() if word.isalpha()]
    
    def _words_similar(self, word1: str, word2: str, threshold: float = 0.7) -> bool:
        """Check if two words are similar (for self-correction detection)."""
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, word1, word2).ratio()
        return similarity >= threshold
    
    def _basic_linguistic_analysis(self, text: str) -> Dict:
        """Basic linguistic analysis when spaCy is not available."""
        words = self._tokenize_words(text)
        sentences = self._segment_sentences(text)
        
        return {
            "num_tokens": len(text.split()),
            "num_sentences": len(sentences),
            "num_words": len(words),
            "avg_sentence_length": np.mean([len(s.split()) for s in sentences]) if sentences else 0,
            "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
            "pos_counts": {},
            "noun_ratio": 0, "verb_ratio": 0, "adj_ratio": 0,
            "adv_ratio": 0, "pronoun_ratio": 0,
            "num_entities": 0, "entities": [], "entity_counts": {},
            "dependency_complexity": 0, "avg_dependency_distance": 0,
            "max_sentence_length": max([len(s.split()) for s in sentences]) if sentences else 0,
            "sentence_length_std": np.std([len(s.split()) for s in sentences]) if len(sentences) > 1 else 0
        }
    
    def _empty_text_result(self) -> Dict:
        """Return empty result structure."""
        return {
            "transcription": {
                "text": "", "language": "en", "segments": [],
                "word_count": 0, "total_duration": 0, "speaking_rate": 0,
                "confidence_scores": [], "average_confidence": 0.0
            },
            "preprocessing": {
                "original_text": "", "processed_text": "", "sentences": [], "words": [],
                "stats": {"original_length": 0, "processed_length": 0, "compression_ratio": 0,
                         "num_sentences": 0, "num_words": 0, "avg_sentence_length": 0, "avg_word_length": 0}
            },
            "linguistic_analysis": self._basic_linguistic_analysis(""),
            "error_analysis": {
                "repetitions": 0, "self_corrections": 0, "incomplete_words": 0,
                "false_starts": 0, "filled_pauses": 0, "word_fragments": 0,
                "semantic_errors": 0, "repetition_rate": 0, "disfluency_rate": 0, "error_rate": 0
            },
            "processing_success": False
        }


# Example usage
if __name__ == "__main__":
    # Example usage
    pipeline = TextAnalysisPipeline()
    
    # # Process audio to text
    # result = pipeline.process_audio_to_text(audio_array)
    # print(f"Transcribed: {result['transcription']['text']}")
    # print(f"Word count: {result['transcription']['word_count']}")
    pass
