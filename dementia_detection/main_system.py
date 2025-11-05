"""
Main Integration Module
Provides a unified interface for the complete dementia detection system,
integrating all ML/DL features for end-to-end processing.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any

from pathlib import Path
import json
from datetime import datetime

# Import all processing modules
from .data_processing.speech_preprocessing import AudioPreprocessor
from .data_processing.speech_features import SpeechFeatureExtractor
from .data_processing.text_analysis import TextAnalysisPipeline
from .data_processing.language_features import LanguageFeatureExtractor
from .data_processing.cognitive_tests import CognitiveTestProcessor
from .data_processing.prosody import AdvancedProsodyAnalyzer
from .data_processing.spectral import AdvancedSpectralAnalyzer
from .data_processing.conversation import ConversationalInteractionAnalyzer
from .data_processing.cognitive_load import CognitiveLoadAnalyzer
from .ml_models.models import ModelManager
from .ml_models.multimodal_fusion import MultimodalFusion

from .config.settings import *




class DementiaDetectionSystem:
    """Complete dementia detection system with multimodal analysis."""
    
    def __init__(self):
        """Initialize the complete system."""
        
        
        # Initialize all processing components
        self.audio_preprocessor = AudioPreprocessor()
        self.speech_feature_extractor = SpeechFeatureExtractor()
        self.text_pipeline = TextAnalysisPipeline()
        self.language_feature_extractor = LanguageFeatureExtractor()
        self.cognitive_processor = CognitiveTestProcessor()
        
        # Initialize advanced analysis components
        self.advanced_prosody = AdvancedProsodyAnalyzer()
        self.advanced_spectral = AdvancedSpectralAnalyzer()
        self.conversational_analyzer = ConversationalInteractionAnalyzer()
        self.cognitive_load_analyzer = CognitiveLoadAnalyzer()
        
        # Initialize ML components
        self.model_manager = ModelManager()
        self.multimodal_fusion = MultimodalFusion()
        
        # System state
        self.is_trained = False
        self.trained_models = {}
        self.feature_history = {}
        
        
    
    def process_audio_sample(self, audio_file_path: str, 
                           extract_speech_features: bool = True,
                           extract_text_features: bool = True) -> Dict[str, Any]:
        """
        Process a single audio sample through the complete pipeline.
        
        Args:
            audio_file_path: Path to audio file
            extract_speech_features: Whether to extract speech features
            extract_text_features: Whether to extract text/language features
            
        Returns:
            Dictionary containing all extracted features and analysis
        """
        try:
            
            
            # Step 1: Audio preprocessing
            preprocessing_result = self.audio_preprocessor.preprocess_audio(audio_file_path)
            audio = preprocessing_result["audio"]
            sample_rate = preprocessing_result["sample_rate"]
            speech_segments = preprocessing_result["speech_segments"]
            
            results = {
                "file_path": audio_file_path,
                "preprocessing": preprocessing_result,
                "speech_features": {},
                "text_features": {},
                "language_features": {},
                "processing_timestamp": datetime.now().isoformat()
            }
            
            # Step 2: Comprehensive speech feature extraction
            if extract_speech_features:
                
                
                # Basic speech features
                speech_features = self.speech_feature_extractor.extract_all_features(
                    audio, speech_segments
                )
                
                # Advanced prosodic analysis
                prosody_features = {}
                prosody_features.update(self.advanced_prosody.phonation_features(audio))
                prosody_features.update(self.advanced_prosody.pause_distribution_features(audio, speech_segments))
                prosody_features.update(self.advanced_prosody.speech_rhythm_features(audio, speech_segments))
                prosody_features.update(self.advanced_prosody.energy_dynamics_features(audio))
                
                # Advanced spectral analysis
                spectral_features = {}
                spectral_features.update(self.advanced_spectral.detailed_mfcc_features(audio))
                spectral_features.update(self.advanced_spectral.spectral_shape_features(audio))
                spectral_features.update(self.advanced_spectral.timbre_features(audio))
                spectral_features.update(self.advanced_spectral.voice_clarity_features(audio))
                
                # Cognitive load analysis
                cognitive_load_features = {}
                cognitive_load_features.update(self.cognitive_load_analyzer.extract_processing_effort_features(audio))
                cognitive_load_features.update(self.cognitive_load_analyzer.extract_mental_effort_markers(audio))
                
                # Combine all speech features
                results["speech_features"] = speech_features
                results["advanced_prosody_features"] = prosody_features
                results["advanced_spectral_features"] = spectral_features
                results["cognitive_load_features"] = cognitive_load_features
                
                total_features = len(speech_features) + len(prosody_features) + len(spectral_features) + len(cognitive_load_features)
                
            
            # Step 3: Text analysis and language features
            if extract_text_features:
                
                
                # Speech-to-text and preprocessing
                text_analysis = self.text_pipeline.process_audio_to_text(audio, sample_rate)
                results["text_features"] = text_analysis
                
                if text_analysis["processing_success"] and text_analysis["transcription"]["text"]:
                    # Language feature extraction
                    
                    processed_text = text_analysis["preprocessing"]["processed_text"]
                    sentences = text_analysis["preprocessing"]["sentences"] 
                    words = text_analysis["preprocessing"]["words"]
                    transcript = text_analysis["transcription"]["text"]
                    
                    # Basic language features
                    language_features = self.language_feature_extractor.extract_all_language_features(
                        processed_text, sentences, words
                    )
                    
                    # Conversational analysis (if speaker segments available)
                    conversational_features = {}
                    if hasattr(text_analysis, 'speaker_segments'):
                        speaker_segments = text_analysis.get('speaker_segments', [])
                        total_duration = len(audio) / sample_rate
                        
                        conversational_features.update(
                            self.conversational_analyzer.extract_turn_taking_features(speaker_segments, total_duration)
                        )
                        conversational_features.update(
                            self.conversational_analyzer.extract_interruption_patterns(speaker_segments)
                        )
                    
                    # Hesitation pattern analysis
                    hesitation_features = self.conversational_analyzer.extract_hesitation_patterns(audio, transcript)
                    
                    # Advanced cognitive load analysis with text
                    text_cognitive_features = self.cognitive_load_analyzer.extract_processing_effort_features(audio, transcript)
                    
                    results["language_features"] = language_features
                    results["conversational_features"] = conversational_features
                    results["hesitation_features"] = hesitation_features
                    results["text_cognitive_features"] = text_cognitive_features
                    
                    total_text_features = (len(language_features) + len(conversational_features) + 
                                         len(hesitation_features) + len(text_cognitive_features))
                    
                else:
                    pass
            
            return results
            
        except Exception as e:
            raise
    
    def process_cognitive_tests(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process cognitive test results.
        
        Args:
            test_data: Dictionary containing test results
                {
                    "word_recall": {...},
                    "digit_span": {...},
                    "reaction_time": {...},
                    "clock_drawing": {...},
                    "patient_info": {"age": int, ...}
                }
                
        Returns:
            Dictionary containing processed cognitive features
        """
        try:
            
            
            results = {
                "processing_timestamp": datetime.now().isoformat(),
                "cognitive_features": {},
                "test_summaries": {}
            }
            
            # Process each type of cognitive test
            if "word_recall" in test_data:
                
                word_recall_features = self.cognitive_processor.process_word_recall_test(
                    test_data["word_recall"]
                )
                results["cognitive_features"].update({
                    f"word_recall_{k}": v for k, v in word_recall_features.items()
                })
                results["test_summaries"]["word_recall"] = word_recall_features
            
            if "digit_span" in test_data:
                
                digit_span_features = self.cognitive_processor.process_digit_span_test(
                    test_data["digit_span"]
                )
                results["cognitive_features"].update({
                    f"digit_span_{k}": v for k, v in digit_span_features.items()
                })
                results["test_summaries"]["digit_span"] = digit_span_features
            
            if "reaction_time" in test_data:
                
                rt_features = self.cognitive_processor.process_reaction_time_test(
                    test_data["reaction_time"]
                )
                results["cognitive_features"].update({
                    f"reaction_time_{k}": v for k, v in rt_features.items()
                })
                results["test_summaries"]["reaction_time"] = rt_features
            
            if "clock_drawing" in test_data:
                
                clock_features = self.cognitive_processor.process_clock_drawing_test(
                    test_data["clock_drawing"]
                )
                results["cognitive_features"].update({
                    f"clock_drawing_{k}": v for k, v in clock_features.items()
                })
                results["test_summaries"]["clock_drawing"] = clock_features
            
            # Generate overall cognitive summary
            cognitive_summary = self.cognitive_processor.generate_cognitive_summary(
                results["test_summaries"]
            )
            results["cognitive_features"].update({
                f"summary_{k}": v for k, v in cognitive_summary.items()
            })
            
            return results
            
        except Exception as e:
            raise
    
    def train_system(self, training_data: Dict[str, Any], 
                    model_types: List[str] = ["random_forest", "xgboost"],
                    fusion_approaches: List[str] = ["early", "late", "deep"],
                    save_models: bool = True,
                    model_save_path: str = None) -> Dict[str, Any]:
        """
        Train the complete system on provided data.
        
        Args:
            training_data: Dictionary containing training samples
                {
                    "sample_id_1": {
                        "audio_file": "path/to/audio.wav",
                        "cognitive_tests": {...},
                        "label": 0 or 1,
                        "metadata": {...}
                    },
                    ...
                }
            model_types: List of ML model types to train
            fusion_approaches: List of fusion approaches to compare
            save_models: Whether to save trained models
            model_save_path: Path to save models
            
        Returns:
            Dictionary containing training results and model performance
        """
        try:
            # Step 1: Process all training samples
            processed_samples = {}
            labels = {}
            
            for sample_id, sample_data in training_data.items():
                
                
                try:
                    # Process audio
                    if "audio_file" in sample_data:
                        audio_results = self.process_audio_sample(
                            sample_data["audio_file"],
                            extract_speech_features=True,
                            extract_text_features=True
                        )
                    else:
                        audio_results = {"speech_features": {}, "language_features": {}}
                    
                    # Process cognitive tests
                    if "cognitive_tests" in sample_data:
                        cognitive_results = self.process_cognitive_tests(sample_data["cognitive_tests"])
                    else:
                        cognitive_results = {"cognitive_features": {}}
                    
                    # Combine features
                    processed_samples[sample_id] = {
                        "speech_features": audio_results.get("speech_features", {}),
                        "text_features": audio_results.get("language_features", {}),
                        "cognitive_features": cognitive_results.get("cognitive_features", {})
                    }
                    
                    labels[sample_id] = sample_data["label"]
                    
                except Exception as e:
                    
                    continue
            
            # Step 2: Prepare multimodal data
            speech_features = {sid: data["speech_features"] for sid, data in processed_samples.items()}
            text_features = {sid: data["text_features"] for sid, data in processed_samples.items()}
            cognitive_features = {sid: data["cognitive_features"] for sid, data in processed_samples.items()}
            
            processed_data = self.multimodal_fusion.prepare_multimodal_data(
                speech_features, text_features, cognitive_features, labels
            )
            
            # Step 3: Train and compare different approaches
            training_results = {}
            
            for approach in fusion_approaches:
                
                
                if approach == "early":
                    for model_type in model_types:
                        result = self.multimodal_fusion.early_fusion(
                            processed_data, model_type, 'classification'
                        )
                        training_results[f"early_fusion_{model_type}"] = result
                
                elif approach == "late":
                    for model_type in model_types:
                        result = self.multimodal_fusion.late_fusion(
                            processed_data, model_type, 'classification', 'weighted_average'
                        )
                        training_results[f"late_fusion_{model_type}"] = result
                
                elif approach == "deep":
                    for fusion_type in ["concat", "attention"]:
                        result = self.multimodal_fusion.deep_multimodal_fusion(
                            processed_data, fusion_type
                        )
                        training_results[f"deep_fusion_{fusion_type}"] = result
            
            # Step 4: Select best model
            best_model = None
            best_score = 0
            
            for model_name, results in training_results.items():
                if 'metrics' in results:
                    score = results['metrics'].get('test_auc', 0)
                elif 'fusion_score' in results:
                    score = results['fusion_score']
                else:
                    continue
                
                if score > best_score:
                    best_score = score
                    best_model = model_name
            
            # Step 5: Save models if requested
            if save_models and model_save_path:
                save_path = Path(model_save_path)
                save_path.mkdir(parents=True, exist_ok=True)
                
                for model_name in training_results.keys():
                    try:
                        self.multimodal_fusion.save_fusion_model(model_name, str(save_path))
                    except Exception as e:
                        pass
            
            # Compile final results
            final_results = {
                "training_completed": True,
                "num_samples": len(processed_samples),
                "model_results": training_results,
                "best_model": best_model,
                "best_score": best_score,
                "feature_dimensions": {
                    "speech": processed_data["speech"].shape[1],
                    "text": processed_data["text"].shape[1],
                    "cognitive": processed_data["cognitive"].shape[1]
                }
            }
            
            self.is_trained = True
            self.trained_models = training_results
            
            return final_results
            
        except Exception as e:
            raise
    
    def predict_dementia_risk(self, sample_data: Dict[str, Any], 
                             model_name: str = None) -> Dict[str, Any]:
        """
        Predict dementia risk for a new sample.
        
        Args:
            sample_data: Dictionary containing sample data
                {
                    "audio_file": "path/to/audio.wav",  # optional
                    "cognitive_tests": {...},           # optional  
                    "speech_features": {...},           # optional (if pre-computed)
                    "text_features": {...},             # optional (if pre-computed)
                    "cognitive_features": {...}         # optional (if pre-computed)
                }
            model_name: Name of specific model to use (if None, uses best model)
            
        Returns:
            Dictionary containing risk prediction and analysis
        """
        try:
            if not self.is_trained:
                raise RuntimeError("System must be trained before making predictions")
            
            
            
            # Process features if not provided
            speech_features = sample_data.get("speech_features")
            text_features = sample_data.get("text_features") 
            cognitive_features = sample_data.get("cognitive_features")
            
            if not speech_features or not text_features:
                if "audio_file" in sample_data:
                    audio_results = self.process_audio_sample(sample_data["audio_file"])
                    speech_features = audio_results.get("speech_features", {})
                    text_features = audio_results.get("language_features", {})
                else:
                    speech_features = speech_features or {}
                    text_features = text_features or {}
            
            if not cognitive_features:
                if "cognitive_tests" in sample_data:
                    cognitive_results = self.process_cognitive_tests(sample_data["cognitive_tests"])
                    cognitive_features = cognitive_results.get("cognitive_features", {})
                else:
                    cognitive_features = {}
            
            # Select model
            if model_name is None:
                # Use best model from training
                model_name = max(self.trained_models.keys(), 
                               key=lambda k: self.trained_models[k].get('metrics', {}).get('test_auc', 0))
            
            if model_name not in self.multimodal_fusion.models:
                raise ValueError(f"Model {model_name} not found or not trained")
            
            # Make prediction
            prediction = self.multimodal_fusion.predict_multimodal(
                speech_features, text_features, cognitive_features, model_name
            )
            
            # Generate detailed analysis
            analysis = self._generate_risk_analysis(
                speech_features, text_features, cognitive_features, prediction
            )
            
            result = {
                "prediction": prediction,
                "analysis": analysis,
                "model_used": model_name,
                "prediction_timestamp": datetime.now().isoformat(),
                "feature_summary": {
                    "speech_features_count": len(speech_features),
                    "text_features_count": len(text_features),
                    "cognitive_features_count": len(cognitive_features)
                }
            }
            
            
            return result
            
        except Exception as e:
            
            raise
    
    def generate_report(self, prediction_result: Dict[str, Any], 
                       include_recommendations: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive assessment report.
        
        Args:
            prediction_result: Result from predict_dementia_risk()
            include_recommendations: Whether to include clinical recommendations
            
        Returns:
            Dictionary containing formatted report
        """
        try:
            prediction = prediction_result["prediction"]
            analysis = prediction_result["analysis"]
            
            # Generate report sections
            report = {
                "report_generated": datetime.now().isoformat(),
                "executive_summary": {
                    "risk_score": prediction["risk_score"],
                    "risk_category": prediction["risk_category"],
                    "confidence_level": prediction["confidence"],
                    "overall_assessment": self._generate_assessment_summary(prediction)
                },
                "detailed_findings": {
                    "speech_analysis": analysis.get("speech_analysis", {}),
                    "language_analysis": analysis.get("language_analysis", {}),
                    "cognitive_analysis": analysis.get("cognitive_analysis", {})
                },
                "key_indicators": analysis.get("key_indicators", []),
                "technical_details": {
                    "model_used": prediction_result["model_used"],
                    "feature_counts": prediction_result["feature_summary"],
                    "processing_quality": analysis.get("processing_quality", {})
                }
            }
            
            if include_recommendations:
                report["recommendations"] = self._generate_recommendations(prediction, analysis)
            
            return report
            
        except Exception as e:
            
            raise
    
    # Helper methods
    def _generate_risk_analysis(self, speech_features: Dict, text_features: Dict, 
                               cognitive_features: Dict, prediction: Dict) -> Dict[str, Any]:
        """Generate detailed risk analysis from features and prediction."""
        analysis = {
            "speech_analysis": {},
            "language_analysis": {},
            "cognitive_analysis": {},
            "key_indicators": [],
            "processing_quality": {}
        }
        
        # Speech analysis
        if speech_features:
            speech_indicators = []
            
            # Check speech rate
            if "estimated_syllable_rate" in speech_features:
                rate = speech_features["estimated_syllable_rate"]
                if rate < 2.0:  # Very slow speech
                    speech_indicators.append("Reduced speech rate detected")
            
            # Check pause patterns
            if "pause_to_speech_ratio" in speech_features:
                ratio = speech_features["pause_to_speech_ratio"]
                if ratio > 0.5:  # More pauses than speech
                    speech_indicators.append("Increased pause duration")
            
            # Check voice quality
            if "jitter_local" in speech_features:
                jitter = speech_features["jitter_local"]
                if jitter > 0.02:  # High jitter indicates voice instability
                    speech_indicators.append("Voice instability detected")
            
            analysis["speech_analysis"] = {
                "indicators": speech_indicators,
                "speech_rate": speech_features.get("estimated_syllable_rate", 0),
                "pause_ratio": speech_features.get("pause_to_speech_ratio", 0)
            }
        
        # Language analysis
        if text_features:
            language_indicators = []
            
            # Check vocabulary richness
            if "type_token_ratio" in text_features:
                ttr = text_features["type_token_ratio"]
                if ttr < 0.4:  # Low lexical diversity
                    language_indicators.append("Reduced vocabulary diversity")
            
            # Check semantic coherence
            if "avg_sentence_similarity" in text_features:
                coherence = text_features["avg_sentence_similarity"]
                if coherence < 0.3:  # Low coherence
                    language_indicators.append("Reduced semantic coherence")
            
            analysis["language_analysis"] = {
                "indicators": language_indicators,
                "vocabulary_diversity": text_features.get("type_token_ratio", 0),
                "semantic_coherence": text_features.get("avg_sentence_similarity", 0)
            }
        
        # Cognitive analysis
        if cognitive_features:
            cognitive_indicators = []
            
            # Check memory performance
            memory_fields = [k for k in cognitive_features.keys() if "recall" in k and "accuracy" in k]
            if memory_fields:
                avg_memory = np.mean([cognitive_features[field] for field in memory_fields])
                if avg_memory < 0.6:  # Below 60% accuracy
                    cognitive_indicators.append("Memory performance below expected range")
            
            analysis["cognitive_analysis"] = {
                "indicators": cognitive_indicators,
                "memory_composite": cognitive_features.get("summary_memory_composite", 0)
            }
        
        # Combine key indicators
        all_indicators = (analysis["speech_analysis"].get("indicators", []) +
                         analysis["language_analysis"].get("indicators", []) +
                         analysis["cognitive_analysis"].get("indicators", []))
        
        analysis["key_indicators"] = all_indicators
        
        return analysis
    
    def _generate_assessment_summary(self, prediction: Dict) -> str:
        """Generate human-readable assessment summary."""
        risk_score = prediction["risk_score"]
        confidence = prediction["confidence"]
        
        if risk_score < 0.3:
            base_text = "Analysis suggests low likelihood of cognitive impairment"
        elif risk_score < 0.7:
            base_text = "Analysis indicates moderate risk that warrants further evaluation"
        else:
            base_text = "Analysis suggests elevated risk of cognitive impairment"
        
        confidence_text = f"high confidence" if confidence > 0.7 else f"moderate confidence" if confidence > 0.4 else f"low confidence"
        
        return f"{base_text} with {confidence_text} (confidence: {confidence:.1%})."
    
    def _generate_recommendations(self, prediction: Dict, analysis: Dict) -> List[str]:
        """Generate clinical recommendations based on analysis."""
        recommendations = []
        risk_score = prediction["risk_score"]
        
        if risk_score > 0.7:
            recommendations.extend([
                "Consider referral for comprehensive neuropsychological evaluation",
                "Recommend follow-up assessment in 3-6 months",
                "Consider brain imaging studies if clinically indicated"
            ])
        elif risk_score > 0.4:
            recommendations.extend([
                "Monitor cognitive status with regular assessments",
                "Consider lifestyle interventions to support cognitive health",
                "Follow-up assessment in 6-12 months recommended"
            ])
        else:
            recommendations.extend([
                "Continue routine cognitive health monitoring",
                "Maintain healthy lifestyle practices",
                "Re-assess if new symptoms emerge"
            ])
        
        # Add specific recommendations based on key indicators
        key_indicators = analysis.get("key_indicators", [])
        
        if any("speech" in indicator.lower() for indicator in key_indicators):
            recommendations.append("Consider speech-language pathology evaluation")
        
        if any("memory" in indicator.lower() for indicator in key_indicators):
            recommendations.append("Focus on memory assessment in clinical evaluation")
        
        return recommendations


# Example usage and system integration
if __name__ == "__main__":
    # Example usage
    system = DementiaDetectionSystem()
    
    # # Train the system
    # training_data = {...}  # Dictionary with training samples
    # results = system.train_system(training_data)
    # print(f"Training completed. Best model AUC: {results['best_score']:.3f}")
    # 
    # # Make predictions
    # sample = {"audio_file": "path/to/audio.wav", "cognitive_tests": {...}}
    # prediction = system.predict_dementia_risk(sample)
    # print(f"Risk score: {prediction['prediction']['risk_score']:.3f}")
    # 
    # # Generate report
    # report = system.generate_report(prediction)
    # print(f"Assessment: {report['executive_summary']['overall_assessment']}")
    pass
