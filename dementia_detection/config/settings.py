# Configuration settings for Dementia Detection System

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "ml_models" / "saved_models"
LOGS_DIR = BASE_DIR / "logs"

# Audio processing settings
AUDIO_CONFIG = {
    "sample_rate": 16000,
    "max_audio_length": 300,  # 5 minutes in seconds
    "chunk_size": 1024,
    "noise_reduction_strength": 0.5,
    "silence_threshold": 0.01,
    "min_speech_duration": 0.5,  # minimum speech segment length in seconds
}

# Speech feature extraction settings
SPEECH_FEATURES_CONFIG = {
    "mfcc_coefficients": 13,
    "n_fft": 2048,
    "hop_length": 512,
    "window_length": 2048,
    "mel_filters": 26,
    "pitch_min": 75,  # Hz
    "pitch_max": 400,  # Hz
    "formant_count": 4,
}

# Text analysis settings
TEXT_CONFIG = {
    "max_text_length": 10000,
    "sentence_embedding_model": "all-MiniLM-L6-v2",
    "whisper_model": "base",  # base, small, medium, large
    "language": "en",
}

# NLP features settings
NLP_FEATURES_CONFIG = {
    "min_word_length": 2,
    "max_parse_depth": 10,
    "coherence_window": 3,  # sentences
    "vocabulary_size_threshold": 100,
}

# Cognitive test settings
COGNITIVE_TEST_CONFIG = {
    "word_recall_max_words": 20,
    "digit_span_max_digits": 9,
    "reaction_time_threshold": 2000,  # ms
    "age_groups": [(18, 39), (40, 59), (60, 79), (80, 100)],
}

# Machine Learning settings
ML_CONFIG = {
    "train_test_split": 0.2,
    "validation_split": 0.1,
    "random_state": 42,
    "cv_folds": 5,
    "early_stopping_patience": 10,
    "max_iterations": 1000,
}

# Model settings
MODEL_CONFIG = {
    "risk_threshold": 0.5,
    "confidence_threshold": 0.7,
    "ensemble_weights": {
        "speech": 0.4,
        "text": 0.3,
        "cognitive": 0.3
    }
}

# Security settings
SECURITY_CONFIG = {
    "encryption_key_size": 256,
    "jwt_expiry_hours": 24,
    "max_file_size": 100 * 1024 * 1024,  # 100MB
}

# API settings
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": True,
    "cors_origins": ["*"],
    "rate_limit": "100/minute",
}

# Database settings
DATABASE_CONFIG = {
    "url": os.getenv("DATABASE_URL", "postgresql://user:password@localhost/dementia_db"),
    "pool_size": 5,
    "max_overflow": 10,
    "echo": False,
}

# Storage settings
STORAGE_CONFIG = {
    "type": "local",  # local, s3, minio
    "local_path": DATA_DIR / "uploads",
    "s3_bucket": os.getenv("S3_BUCKET_NAME", "dementia-data"),
    "s3_region": os.getenv("AWS_REGION", "us-east-1"),
}

# Experiment tracking
EXPERIMENT_CONFIG = {
    "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    "wandb_project": "dementia-detection",
    "experiment_name": "dementia_risk_model",
}


class Settings:
    """Configuration settings wrapper for the Dementia Detection System."""
    
    def __init__(self):
        """Initialize settings with all configuration dictionaries."""
        self.base_dir = BASE_DIR
        self.data_dir = DATA_DIR
        self.models_dir = MODELS_DIR
        self.logs_dir = LOGS_DIR
        
        self.audio = AUDIO_CONFIG
        self.speech_features = SPEECH_FEATURES_CONFIG
        self.text = TEXT_CONFIG
        self.nlp_features = NLP_FEATURES_CONFIG
        self.cognitive_test = COGNITIVE_TEST_CONFIG
        self.ml = ML_CONFIG
        self.model = MODEL_CONFIG
        self.security = SECURITY_CONFIG
        self.api = API_CONFIG
        self.database = DATABASE_CONFIG
        self.storage = STORAGE_CONFIG
        self.experiment = EXPERIMENT_CONFIG
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.data_dir, self.models_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get(self, section, key=None, default=None):
        """Get configuration value by section and key."""
        section_config = getattr(self, section, None)
        if section_config is None:
            return default
        
        if key is None:
            return section_config
        
        return section_config.get(key, default)
    
    def update(self, section, key, value):
        """Update configuration value."""
        section_config = getattr(self, section, None)
        if section_config is not None:
            section_config[key] = value
