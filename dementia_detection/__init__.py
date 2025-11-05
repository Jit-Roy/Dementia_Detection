"""
Dementia Detection System - Main Package
A comprehensive ML/DL system for dementia risk assessment using multimodal analysis
"""

__version__ = "0.1.0"
__author__ = "Dementia Detection Team"

# Import main system class
from .main_system import DementiaDetectionSystem

# Import core components
from .data_processing.speech_preprocessing import AudioPreprocessor
from .data_processing.speech_features import SpeechFeatureExtractor
from .data_processing.text_analysis import TextAnalysisPipeline
from .data_processing.language_features import LanguageFeatureExtractor
from .data_processing.cognitive_tests import CognitiveTestProcessor

from .ml_models.models import ModelManager
from .ml_models.multimodal_fusion import MultimodalFusion

from .config.settings import Settings

# Define what gets imported with "from dementia_detection import *"
__all__ = [
    'DementiaDetectionSystem',
    'AudioPreprocessor',
    'SpeechFeatureExtractor', 
    'TextAnalysisPipeline',
    'LanguageFeatureExtractor',
    'CognitiveTestProcessor',
    'ModelManager',
    'MultimodalFusion',
    'Settings'
]
