"""
Cognitive Test Processing Module
Handles processing and analysis of various cognitive tests including
word recall, digit span, reaction times, and drawing assessments.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union

from datetime import datetime, timedelta
import json
from scipy import stats
from sklearn.preprocessing import StandardScaler

from ..config.settings import COGNITIVE_TEST_CONFIG




class CognitiveTestProcessor:
    """Process and analyze various cognitive test results."""
    
    def __init__(self):
        """Initialize CognitiveTestProcessor with normative data."""
        self.config = COGNITIVE_TEST_CONFIG
        
        # Load normative data (in a real system, this would come from a database)
        self._load_normative_data()
        
    def _load_normative_data(self):
        """Load normative data for cognitive tests."""
        # Normative data by age group: (mean, std) for healthy controls
        # These are example values - in production, use validated norms
        
        self.normative_data = {
            "word_recall": {
                (18, 39): {"immediate": (14.2, 2.1), "delayed": (12.8, 2.3)},
                (40, 59): {"immediate": (13.8, 2.2), "delayed": (12.1, 2.4)},
                (60, 79): {"immediate": (12.9, 2.5), "delayed": (11.2, 2.6)},
                (80, 100): {"immediate": (11.5, 2.8), "delayed": (9.8, 2.9)}
            },
            "digit_span": {
                (18, 39): {"forward": (7.2, 1.1), "backward": (5.8, 1.2)},
                (40, 59): {"forward": (6.9, 1.2), "backward": (5.5, 1.3)},
                (60, 79): {"forward": (6.4, 1.3), "backward": (5.1, 1.4)},
                (80, 100): {"forward": (5.9, 1.4), "backward": (4.6, 1.5)}
            },
            "reaction_time": {
                (18, 39): {"simple": (285, 45), "choice": (420, 65)},
                (40, 59): {"simple": (295, 50), "choice": (440, 70)},
                (60, 79): {"simple": (315, 55), "choice": (475, 80)},
                (80, 100): {"simple": (340, 65), "choice": (520, 95)}
            },
            "clock_drawing": {
                (18, 39): {"total_score": (9.2, 0.8)},
                (40, 59): {"total_score": (9.1, 0.9)},
                (60, 79): {"total_score": (8.8, 1.1)},
                (80, 100): {"total_score": (8.3, 1.3)}
            }
        }
    
    def get_age_group(self, age: int) -> Tuple[int, int]:
        """
        Get age group for normative comparison.
        
        Args:
            age: Patient age
            
        Returns:
            Age group tuple (min_age, max_age)
        """
        for age_range in self.config["age_groups"]:
            if age_range[0] <= age <= age_range[1]:
                return age_range
        
        # Default to oldest group if age exceeds ranges
        return self.config["age_groups"][-1]
    
    def process_word_recall_test(self, test_data: Dict) -> Dict[str, float]:
        """
        Process word recall test results.
        
        Args:
            test_data: Dictionary containing test results
                {
                    "words_presented": List[str],
                    "immediate_recall": List[str],
                    "delayed_recall": List[str],
                    "recognition_items": List[str],
                    "recognition_responses": List[bool],
                    "age": int,
                    "test_date": str
                }
                
        Returns:
            Dictionary of processed features
        """
        try:
            words_presented = test_data.get("words_presented", [])
            immediate_recall = test_data.get("immediate_recall", [])
            delayed_recall = test_data.get("delayed_recall", [])
            recognition_items = test_data.get("recognition_items", [])
            recognition_responses = test_data.get("recognition_responses", [])
            age = test_data.get("age", 65)
            
            # Basic scores
            num_words_presented = len(words_presented)
            immediate_score = len(immediate_recall)
            delayed_score = len(delayed_recall)
            
            # Accuracy calculations
            correct_immediate = self._calculate_recall_accuracy(words_presented, immediate_recall)
            correct_delayed = self._calculate_recall_accuracy(words_presented, delayed_recall)
            
            immediate_accuracy = correct_immediate / num_words_presented if num_words_presented > 0 else 0
            delayed_accuracy = correct_delayed / num_words_presented if num_words_presented > 0 else 0
            
            # Recognition memory
            recognition_accuracy = 0
            recognition_hits = 0
            recognition_false_alarms = 0
            
            if recognition_items and recognition_responses:
                correct_recognition = 0
                for item, response in zip(recognition_items, recognition_responses):
                    if item in words_presented and response:  # Hit
                        correct_recognition += 1
                        recognition_hits += 1
                    elif item not in words_presented and not response:  # Correct rejection
                        correct_recognition += 1
                    elif item not in words_presented and response:  # False alarm
                        recognition_false_alarms += 1
                
                recognition_accuracy = correct_recognition / len(recognition_items)
            
            # Memory retention (delayed vs immediate)
            retention_rate = delayed_accuracy / immediate_accuracy if immediate_accuracy > 0 else 0
            
            # Error analysis
            intrusions_immediate = len([word for word in immediate_recall if word not in words_presented])
            intrusions_delayed = len([word for word in delayed_recall if word not in words_presented])
            
            # Serial position effects
            serial_position_immediate = self._analyze_serial_position(words_presented, immediate_recall)
            serial_position_delayed = self._analyze_serial_position(words_presented, delayed_recall)
            
            # Clustering analysis (semantic or phonemic clustering)
            semantic_clustering = self._analyze_clustering(words_presented, immediate_recall, "semantic")
            
            # Normative comparison
            age_group = self.get_age_group(age)
            normative_scores = self._compare_to_norms(
                {"immediate": immediate_accuracy, "delayed": delayed_accuracy},
                "word_recall",
                age_group
            )
            
            # Compile results
            features = {
                # Raw scores
                "words_presented_count": num_words_presented,
                "immediate_recall_count": immediate_score,
                "delayed_recall_count": delayed_score,
                "correct_immediate": correct_immediate,
                "correct_delayed": correct_delayed,
                
                # Accuracy measures
                "immediate_recall_accuracy": immediate_accuracy,
                "delayed_recall_accuracy": delayed_accuracy,
                "recognition_accuracy": recognition_accuracy,
                "retention_rate": retention_rate,
                
                # Error measures
                "intrusions_immediate": intrusions_immediate,
                "intrusions_delayed": intrusions_delayed,
                "intrusion_rate_immediate": intrusions_immediate / immediate_score if immediate_score > 0 else 0,
                "intrusion_rate_delayed": intrusions_delayed / delayed_score if delayed_score > 0 else 0,
                
                # Recognition measures
                "recognition_hits": recognition_hits,
                "recognition_false_alarms": recognition_false_alarms,
                "recognition_discrimination": recognition_hits - recognition_false_alarms,
                
                # Serial position effects
                "primacy_effect_immediate": serial_position_immediate.get("primacy", 0),
                "recency_effect_immediate": serial_position_immediate.get("recency", 0),
                "primacy_effect_delayed": serial_position_delayed.get("primacy", 0),
                "recency_effect_delayed": serial_position_delayed.get("recency", 0),
                
                # Clustering
                "semantic_clustering_score": semantic_clustering,
                
                # Normative scores
                "immediate_z_score": normative_scores.get("immediate_z", 0),
                "delayed_z_score": normative_scores.get("delayed_z", 0),
                "immediate_percentile": normative_scores.get("immediate_percentile", 50),
                "delayed_percentile": normative_scores.get("delayed_percentile", 50)
            }
            
            return features
            
        except Exception as e:
            
            return self._empty_word_recall_features()
    
    def process_digit_span_test(self, test_data: Dict) -> Dict[str, float]:
        """
        Process digit span test results.
        
        Args:
            test_data: Dictionary containing test results
                {
                    "forward_trials": List[Dict],  # [{"digits": [1,2,3], "response": [1,2,3], "correct": bool}]
                    "backward_trials": List[Dict],
                    "age": int,
                    "test_date": str
                }
                
        Returns:
            Dictionary of processed features
        """
        try:
            forward_trials = test_data.get("forward_trials", [])
            backward_trials = test_data.get("backward_trials", [])
            age = test_data.get("age", 65)
            
            # Process forward span
            forward_features = self._process_digit_span_direction(forward_trials, "forward")
            
            # Process backward span
            backward_features = self._process_digit_span_direction(backward_trials, "backward")
            
            # Combined features
            total_trials = len(forward_trials) + len(backward_trials)
            total_correct = forward_features["correct_count"] + backward_features["correct_count"]
            
            overall_accuracy = total_correct / total_trials if total_trials > 0 else 0
            
            # Span difference (working memory load effect)
            span_difference = forward_features["max_span"] - backward_features["max_span"]
            
            # Normative comparison
            age_group = self.get_age_group(age)
            normative_scores = self._compare_to_norms(
                {"forward": forward_features["max_span"], "backward": backward_features["max_span"]},
                "digit_span",
                age_group
            )
            
            # Compile results
            features = {
                # Forward span features
                **{f"forward_{k}": v for k, v in forward_features.items()},
                
                # Backward span features  
                **{f"backward_{k}": v for k, v in backward_features.items()},
                
                # Combined features
                "total_trials": total_trials,
                "total_correct": total_correct,
                "overall_accuracy": overall_accuracy,
                "span_difference": span_difference,
                "working_memory_index": backward_features["max_span"] / forward_features["max_span"] if forward_features["max_span"] > 0 else 0,
                
                # Normative scores
                "forward_z_score": normative_scores.get("forward_z", 0),
                "backward_z_score": normative_scores.get("backward_z", 0),
                "forward_percentile": normative_scores.get("forward_percentile", 50),
                "backward_percentile": normative_scores.get("backward_percentile", 50)
            }
            
            return features
            
        except Exception as e:
            
            return self._empty_digit_span_features()
    
    def process_reaction_time_test(self, test_data: Dict) -> Dict[str, float]:
        """
        Process reaction time test results.
        
        Args:
            test_data: Dictionary containing test results
                {
                    "simple_rt_trials": List[Dict],  # [{"stimulus_time": float, "response_time": float, "correct": bool}]
                    "choice_rt_trials": List[Dict],
                    "age": int,
                    "test_date": str
                }
                
        Returns:
            Dictionary of processed features
        """
        try:
            simple_trials = test_data.get("simple_rt_trials", [])
            choice_trials = test_data.get("choice_rt_trials", [])
            age = test_data.get("age", 65)
            
            # Process simple RT
            simple_features = self._process_rt_condition(simple_trials, "simple")
            
            # Process choice RT
            choice_features = self._process_rt_condition(choice_trials, "choice")
            
            # Choice RT cost (difference between choice and simple RT)
            choice_cost = choice_features["mean_rt"] - simple_features["mean_rt"]
            
            # Variability measures
            intraindividual_variability_simple = simple_features["cv_rt"]
            intraindividual_variability_choice = choice_features["cv_rt"]
            
            # Normative comparison
            age_group = self.get_age_group(age)
            normative_scores = self._compare_to_norms(
                {"simple": simple_features["mean_rt"], "choice": choice_features["mean_rt"]},
                "reaction_time",
                age_group
            )
            
            # Compile results
            features = {
                # Simple RT features
                **{f"simple_{k}": v for k, v in simple_features.items()},
                
                # Choice RT features
                **{f"choice_{k}": v for k, v in choice_features.items()},
                
                # Comparative measures
                "choice_rt_cost": choice_cost,
                "rt_cost_ratio": choice_cost / simple_features["mean_rt"] if simple_features["mean_rt"] > 0 else 0,
                
                # Variability measures
                "simple_rt_variability": intraindividual_variability_simple,
                "choice_rt_variability": intraindividual_variability_choice,
                "variability_difference": intraindividual_variability_choice - intraindividual_variability_simple,
                
                # Normative scores
                "simple_rt_z_score": normative_scores.get("simple_z", 0),
                "choice_rt_z_score": normative_scores.get("choice_z", 0),
                "simple_rt_percentile": normative_scores.get("simple_percentile", 50),
                "choice_rt_percentile": normative_scores.get("choice_percentile", 50)
            }
            
            return features
            
        except Exception as e:
            
            return self._empty_reaction_time_features()
    
    def process_clock_drawing_test(self, test_data: Dict) -> Dict[str, float]:
        """
        Process clock drawing test results.
        
        Args:
            test_data: Dictionary containing test results
                {
                    "image_path": str,  # Path to drawing image
                    "manual_scores": Dict,  # Manual scoring if available
                    "coordinates": List[Tuple],  # Drawing coordinates if available
                    "time_taken": float,  # Time to complete drawing
                    "age": int,
                    "test_date": str
                }
                
        Returns:
            Dictionary of processed features
        """
        try:
            manual_scores = test_data.get("manual_scores", {})
            coordinates = test_data.get("coordinates", [])
            time_taken = test_data.get("time_taken", 0)
            age = test_data.get("age", 65)
            
            # If manual scores are available, use them
            if manual_scores:
                features = self._process_manual_clock_scores(manual_scores)
            else:
                # Basic geometric analysis of coordinates if available
                features = self._analyze_drawing_coordinates(coordinates)
            
            # Add timing features
            features.update({
                "drawing_time": time_taken,
                "drawing_speed": len(coordinates) / time_taken if time_taken > 0 and coordinates else 0
            })
            
            # Normative comparison
            age_group = self.get_age_group(age)
            total_score = features.get("total_score", 0)
            
            if total_score > 0:
                normative_scores = self._compare_to_norms(
                    {"total_score": total_score},
                    "clock_drawing",
                    age_group
                )
                features.update({
                    "clock_z_score": normative_scores.get("total_score_z", 0),
                    "clock_percentile": normative_scores.get("total_score_percentile", 50)
                })
            else:
                features.update({"clock_z_score": 0, "clock_percentile": 50})
            
            return features
            
        except Exception as e:
            
            return self._empty_clock_drawing_features()
    
    def analyze_longitudinal_trends(self, test_history: List[Dict]) -> Dict[str, float]:
        """
        Analyze longitudinal trends in cognitive test performance.
        
        Args:
            test_history: List of test results over time
                
        Returns:
            Dictionary of trend analysis features
        """
        try:
            if len(test_history) < 2:
                
                return self._empty_longitudinal_features()
            
            # Sort by date
            sorted_history = sorted(test_history, key=lambda x: x.get("test_date", ""))
            
            # Extract key measures over time
            dates = [datetime.fromisoformat(test.get("test_date", "2000-01-01")) for test in sorted_history]
            
            # Calculate time intervals (in months)
            time_intervals = []
            for i in range(1, len(dates)):
                interval = (dates[i] - dates[i-1]).days / 30.44  # Average days per month
                time_intervals.append(interval)
            
            total_follow_up = (dates[-1] - dates[0]).days / 30.44
            
            # Analyze trends for different cognitive measures
            trend_features = {}
            
            # Define key measures to track
            key_measures = [
                "immediate_recall_accuracy", "delayed_recall_accuracy",
                "forward_max_span", "backward_max_span",
                "simple_mean_rt", "choice_mean_rt",
                "total_score"  # clock drawing
            ]
            
            for measure in key_measures:
                values = [test.get(measure) for test in sorted_history if test.get(measure) is not None]
                
                if len(values) >= 2:
                    # Linear trend analysis
                    x = np.arange(len(values))
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                    
                    # Annualized change rate
                    if len(time_intervals) > 0:
                        avg_interval = np.mean(time_intervals)
                        annualized_slope = slope * (12 / avg_interval) if avg_interval > 0 else 0
                    else:
                        annualized_slope = 0
                    
                    trend_features.update({
                        f"{measure}_slope": slope,
                        f"{measure}_r_squared": r_value**2,
                        f"{measure}_p_value": p_value,
                        f"{measure}_annualized_change": annualized_slope,
                        f"{measure}_percent_change": ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
                    })
                    
                    # Variability measures
                    trend_features[f"{measure}_coefficient_variation"] = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
                    
                    # Decline detection (significant negative slope)
                    trend_features[f"{measure}_significant_decline"] = 1 if (slope < 0 and p_value < 0.05) else 0
            
            # Overall cognitive trend summary
            decline_count = sum(1 for key, value in trend_features.items() 
                              if key.endswith("_significant_decline") and value == 1)
            
            trend_features.update({
                "total_follow_up_months": total_follow_up,
                "number_of_assessments": len(sorted_history),
                "avg_assessment_interval": np.mean(time_intervals) if time_intervals else 0,
                "measures_showing_decline": decline_count,
                "proportion_measures_declining": decline_count / len(key_measures)
            })
            
            return trend_features
            
        except Exception as e:
            
            return self._empty_longitudinal_features()
    
    def generate_cognitive_summary(self, all_test_results: Dict) -> Dict[str, float]:
        """
        Generate comprehensive cognitive assessment summary.
        
        Args:
            all_test_results: Dictionary containing all test results
            
        Returns:
            Summary features across cognitive domains
        """
        try:
            summary = {}
            
            # Memory domain (word recall)
            memory_scores = []
            if "word_recall" in all_test_results:
                memory_data = all_test_results["word_recall"]
                memory_scores.extend([
                    memory_data.get("immediate_recall_accuracy", 0),
                    memory_data.get("delayed_recall_accuracy", 0)
                ])
                summary["memory_composite"] = np.mean(memory_scores) if memory_scores else 0
            
            # Working memory domain (digit span)
            working_memory_scores = []
            if "digit_span" in all_test_results:
                digit_data = all_test_results["digit_span"]
                working_memory_scores.extend([
                    digit_data.get("forward_accuracy", 0),
                    digit_data.get("backward_accuracy", 0)
                ])
                summary["working_memory_composite"] = np.mean(working_memory_scores) if working_memory_scores else 0
            
            # Processing speed domain (reaction time)
            processing_speed_scores = []
            if "reaction_time" in all_test_results:
                rt_data = all_test_results["reaction_time"]
                # Invert RT scores (lower RT = better performance)
                simple_rt_norm = 1 / (1 + rt_data.get("simple_mean_rt", 1000) / 1000)
                choice_rt_norm = 1 / (1 + rt_data.get("choice_mean_rt", 1000) / 1000)
                processing_speed_scores.extend([simple_rt_norm, choice_rt_norm])
                summary["processing_speed_composite"] = np.mean(processing_speed_scores) if processing_speed_scores else 0
            
            # Visuospatial domain (clock drawing)
            if "clock_drawing" in all_test_results:
                clock_data = all_test_results["clock_drawing"]
                # Normalize clock score (assuming max score is 10)
                summary["visuospatial_composite"] = clock_data.get("total_score", 0) / 10
            
            # Overall cognitive composite
            domain_scores = [v for k, v in summary.items() if k.endswith("_composite")]
            summary["global_cognitive_composite"] = np.mean(domain_scores) if domain_scores else 0
            
            # Risk indicators
            risk_factors = 0
            
            # Check for low performance in each domain
            if summary.get("memory_composite", 1) < 0.7:  # Below 70% accuracy
                risk_factors += 1
            if summary.get("working_memory_composite", 1) < 0.7:
                risk_factors += 1
            if summary.get("processing_speed_composite", 1) < 0.5:
                risk_factors += 1
            if summary.get("visuospatial_composite", 1) < 0.7:
                risk_factors += 1
            
            summary["cognitive_risk_score"] = risk_factors / 4  # Normalize to 0-1
            
            return summary
            
        except Exception as e:
            
            return {"memory_composite": 0, "working_memory_composite": 0,
                   "processing_speed_composite": 0, "visuospatial_composite": 0,
                   "global_cognitive_composite": 0, "cognitive_risk_score": 0}
    
    # Helper methods
    def _calculate_recall_accuracy(self, presented: List[str], recalled: List[str]) -> int:
        """Calculate number of correctly recalled words."""
        return len([word for word in recalled if word in presented])
    
    def _analyze_serial_position(self, presented: List[str], recalled: List[str]) -> Dict[str, float]:
        """Analyze serial position effects in word recall."""
        if not presented or not recalled:
            return {"primacy": 0, "recency": 0, "middle": 0}
        
        # Define regions (first 20%, last 20%, middle 60%)
        n = len(presented)
        primacy_region = n // 5
        recency_region = n // 5
        
        primacy_words = set(presented[:primacy_region])
        recency_words = set(presented[-recency_region:])
        middle_words = set(presented[primacy_region:-recency_region])
        
        primacy_recalled = len([w for w in recalled if w in primacy_words])
        recency_recalled = len([w for w in recalled if w in recency_words])
        middle_recalled = len([w for w in recalled if w in middle_words])
        
        return {
            "primacy": primacy_recalled / len(primacy_words) if primacy_words else 0,
            "recency": recency_recalled / len(recency_words) if recency_words else 0,
            "middle": middle_recalled / len(middle_words) if middle_words else 0
        }
    
    def _analyze_clustering(self, presented: List[str], recalled: List[str], cluster_type: str) -> float:
        """Analyze clustering in recall (semantic, phonemic, etc.)."""
        # Simplified clustering analysis
        # In practice, would use semantic similarity matrices or phonemic analysis
        
        if len(recalled) < 2:
            return 0
        
        # Simple clustering score based on consecutive word relationships
        clustering_score = 0
        for i in range(len(recalled) - 1):
            if cluster_type == "semantic":
                # Placeholder: check if words are semantically related
                # In practice, use word embeddings or semantic categories
                if self._are_semantically_related(recalled[i], recalled[i+1]):
                    clustering_score += 1
        
        return clustering_score / (len(recalled) - 1) if len(recalled) > 1 else 0
    
    def _are_semantically_related(self, word1: str, word2: str) -> bool:
        """Check if two words are semantically related (placeholder)."""
        # Placeholder implementation
        # In practice, use word embeddings, WordNet, or semantic categories
        return False
    
    def _process_digit_span_direction(self, trials: List[Dict], direction: str) -> Dict[str, float]:
        """Process digit span trials for one direction (forward/backward)."""
        if not trials:
            return {"accuracy": 0, "max_span": 0, "correct_count": 0, "trial_count": 0}
        
        correct_count = sum(1 for trial in trials if trial.get("correct", False))
        accuracy = correct_count / len(trials)
        
        # Calculate maximum span achieved
        max_span = 0
        current_span_length = 0
        
        for trial in trials:
            span_length = len(trial.get("digits", []))
            if trial.get("correct", False):
                max_span = max(max_span, span_length)
                current_span_length = span_length
            
        return {
            "accuracy": accuracy,
            "max_span": max_span,
            "correct_count": correct_count,
            "trial_count": len(trials)
        }
    
    def _process_rt_condition(self, trials: List[Dict], condition: str) -> Dict[str, float]:
        """Process reaction time trials for one condition."""
        if not trials:
            return {"mean_rt": 0, "median_rt": 0, "std_rt": 0, "cv_rt": 0, "accuracy": 0}
        
        # Filter out incorrect trials and outliers
        correct_trials = [t for t in trials if t.get("correct", False)]
        
        if not correct_trials:
            return {"mean_rt": 0, "median_rt": 0, "std_rt": 0, "cv_rt": 0, "accuracy": 0}
        
        rts = [t["response_time"] - t["stimulus_time"] for t in correct_trials 
               if "response_time" in t and "stimulus_time" in t]
        
        # Remove outliers (RTs < 100ms or > 3000ms)
        rts = [rt for rt in rts if 100 <= rt <= 3000]
        
        if not rts:
            return {"mean_rt": 0, "median_rt": 0, "std_rt": 0, "cv_rt": 0, "accuracy": 0}
        
        mean_rt = np.mean(rts)
        median_rt = np.median(rts)
        std_rt = np.std(rts)
        cv_rt = std_rt / mean_rt if mean_rt > 0 else 0
        accuracy = len(correct_trials) / len(trials)
        
        return {
            "mean_rt": mean_rt,
            "median_rt": median_rt,
            "std_rt": std_rt,
            "cv_rt": cv_rt,
            "accuracy": accuracy
        }
    
    def _process_manual_clock_scores(self, scores: Dict) -> Dict[str, float]:
        """Process manually scored clock drawing features."""
        # Standard clock drawing scoring components
        features = {
            "circle_score": scores.get("circle", 0),
            "numbers_score": scores.get("numbers", 0),
            "hands_score": scores.get("hands", 0),
            "time_score": scores.get("time_setting", 0),
            "total_score": scores.get("total", 0)
        }
        
        # Additional computed features
        features["number_placement_accuracy"] = scores.get("number_placement", 0)
        features["hand_positioning_accuracy"] = scores.get("hand_positioning", 0)
        features["overall_organization"] = scores.get("organization", 0)
        
        return features
    
    def _analyze_drawing_coordinates(self, coordinates: List[Tuple]) -> Dict[str, float]:
        """Basic geometric analysis of drawing coordinates."""
        if not coordinates:
            return {"total_score": 0, "drawing_complexity": 0, "spatial_accuracy": 0}
        
        # Basic geometric features
        x_coords = [coord[0] for coord in coordinates]
        y_coords = [coord[1] for coord in coordinates]
        
        # Bounding box
        width = max(x_coords) - min(x_coords) if x_coords else 0
        height = max(y_coords) - min(y_coords) if y_coords else 0
        
        # Drawing complexity (path length)
        total_distance = 0
        for i in range(1, len(coordinates)):
            dx = coordinates[i][0] - coordinates[i-1][0]
            dy = coordinates[i][1] - coordinates[i-1][1]
            total_distance += np.sqrt(dx**2 + dy**2)
        
        return {
            "total_score": 5,  # Default neutral score
            "drawing_complexity": total_distance,
            "spatial_accuracy": min(width, height) / max(width, height) if max(width, height) > 0 else 0,
            "bounding_box_area": width * height
        }
    
    def _compare_to_norms(self, scores: Dict, test_type: str, age_group: Tuple[int, int]) -> Dict[str, float]:
        """Compare scores to normative data."""
        normative_scores = {}
        
        if test_type in self.normative_data and age_group in self.normative_data[test_type]:
            norms = self.normative_data[test_type][age_group]
            
            for measure, score in scores.items():
                if measure in norms:
                    norm_mean, norm_std = norms[measure]
                    z_score = (score - norm_mean) / norm_std if norm_std > 0 else 0
                    percentile = stats.norm.cdf(z_score) * 100
                    
                    normative_scores[f"{measure}_z"] = z_score
                    normative_scores[f"{measure}_percentile"] = percentile
        
        return normative_scores
    
    # Empty feature dictionaries
    def _empty_word_recall_features(self) -> Dict[str, float]:
        """Return empty word recall features."""
        return {
            "words_presented_count": 0, "immediate_recall_count": 0, "delayed_recall_count": 0,
            "correct_immediate": 0, "correct_delayed": 0, "immediate_recall_accuracy": 0,
            "delayed_recall_accuracy": 0, "recognition_accuracy": 0, "retention_rate": 0,
            "intrusions_immediate": 0, "intrusions_delayed": 0, "intrusion_rate_immediate": 0,
            "intrusion_rate_delayed": 0, "recognition_hits": 0, "recognition_false_alarms": 0,
            "recognition_discrimination": 0, "primacy_effect_immediate": 0, "recency_effect_immediate": 0,
            "primacy_effect_delayed": 0, "recency_effect_delayed": 0, "semantic_clustering_score": 0,
            "immediate_z_score": 0, "delayed_z_score": 0, "immediate_percentile": 50, "delayed_percentile": 50
        }
    
    def _empty_digit_span_features(self) -> Dict[str, float]:
        """Return empty digit span features."""
        return {
            "forward_accuracy": 0, "forward_max_span": 0, "forward_correct_count": 0, "forward_trial_count": 0,
            "backward_accuracy": 0, "backward_max_span": 0, "backward_correct_count": 0, "backward_trial_count": 0,
            "total_trials": 0, "total_correct": 0, "overall_accuracy": 0, "span_difference": 0,
            "working_memory_index": 0, "forward_z_score": 0, "backward_z_score": 0,
            "forward_percentile": 50, "backward_percentile": 50
        }
    
    def _empty_reaction_time_features(self) -> Dict[str, float]:
        """Return empty reaction time features."""
        return {
            "simple_mean_rt": 0, "simple_median_rt": 0, "simple_std_rt": 0, "simple_cv_rt": 0, "simple_accuracy": 0,
            "choice_mean_rt": 0, "choice_median_rt": 0, "choice_std_rt": 0, "choice_cv_rt": 0, "choice_accuracy": 0,
            "choice_rt_cost": 0, "rt_cost_ratio": 0, "simple_rt_variability": 0, "choice_rt_variability": 0,
            "variability_difference": 0, "simple_rt_z_score": 0, "choice_rt_z_score": 0,
            "simple_rt_percentile": 50, "choice_rt_percentile": 50
        }
    
    def _empty_clock_drawing_features(self) -> Dict[str, float]:
        """Return empty clock drawing features."""
        return {
            "circle_score": 0, "numbers_score": 0, "hands_score": 0, "time_score": 0, "total_score": 0,
            "number_placement_accuracy": 0, "hand_positioning_accuracy": 0, "overall_organization": 0,
            "drawing_time": 0, "drawing_speed": 0, "drawing_complexity": 0, "spatial_accuracy": 0,
            "bounding_box_area": 0, "clock_z_score": 0, "clock_percentile": 50
        }
    
    def _empty_longitudinal_features(self) -> Dict[str, float]:
        """Return empty longitudinal features."""
        return {
            "total_follow_up_months": 0, "number_of_assessments": 0, "avg_assessment_interval": 0,
            "measures_showing_decline": 0, "proportion_measures_declining": 0
        }


# Example usage
if __name__ == "__main__":
    # Example usage
    processor = CognitiveTestProcessor()
    
    # # Process word recall test
    # word_recall_data = {
    #     "words_presented": ["apple", "chair", "ocean", "happy", "bright"],
    #     "immediate_recall": ["apple", "chair", "ocean"],
    #     "delayed_recall": ["apple", "ocean"],
    #     "age": 65
    # }
    # features = processor.process_word_recall_test(word_recall_data)
    # print(f"Word recall features: {features}")
    pass
