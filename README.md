# Dementia Detection System

A comprehensive machine learning and deep learning system for dementia detection using multimodal analysis including speech, text, and cognitive test data.

## Features

- **Speech Analysis**: Advanced audio preprocessing and feature extraction including MFCC, pitch, energy, formants, prosody, and spectral features
- **Text Analysis**: Automatic speech recognition using OpenAI Whisper, NLP processing with spaCy, and comprehensive language feature extraction
- **Cognitive Assessment**: Processing of standardized neuropsychological tests (word recall, digit span, reaction time, clock drawing)
- **Multimodal Fusion**: Early fusion, late fusion, and deep learning-based fusion strategies
- **Machine Learning Models**: Support for classical ML (Random Forest, XGBoost, LightGBM, SVM) and deep learning (PyTorch, TensorFlow)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Jit-Roy/Dementia_Detection.git
cd Dementia_Detection
```

2. Create a conda environment (recommended):
```bash
conda create -n dementia_detection python=3.10
conda activate dementia_detection
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Processing Cognitive Tests

```python
# Process cognitive tests
cognitive_results = system.process_cognitive_tests({
    "word_recall": word_recall_data,
    "digit_span": digit_span_data,
    "reaction_time": reaction_time_data,
    "clock_drawing": clock_drawing_data
})
```

## Features Extracted

### Speech Features
- MFCCs (13 coefficients)
- Pitch (F0) statistics
- Energy and intensity
- Formants (F1-F4)
- Speech timing (pauses, rate)
- Prosody (intonation, rhythm)
- Voice quality (jitter, shimmer, HNR)
- Spectral features (centroid, rolloff, flux)

### Language Features
- Lexical richness (TTR, MTLD)
- Syntactic complexity
- Semantic coherence
- Discourse markers
- POS distributions
- Named entities

### Cognitive Load Indicators
- Processing effort
- Mental effort markers
- Working memory indicators
- Cognitive decline features