# EMOD - Emotion Recognition System

This repository implements a two-stage emotion recognition system using the IEMOCAP dataset. The system first converts input to valence-arousal-dominance (VAD) tuples, then categorizes emotions based on these values.

## Project Structure

```
src/
├── data/
│   ├── data_loader.py       # Functions to load and preprocess IEMOCAP data
│   └── data_utils.py        # Utility functions for data handling
├── models/
│   ├── vad_predictor.py     # Text-to-VAD model using pre-trained transformers
│   ├── emotion_classifier.py # VAD-to-emotion classifier
│   ├── pipeline.py          # End-to-end pipeline combining both stages
│   ├── audio_processor.py   # Audio feature extraction and VAD prediction
│   ├── multimodal_fusion.py # Fusion of text and audio modalities
│   ├── multimodal_pipeline.py # Multimodal pipeline
│   └── vad_fine_tuner.py    # Fine-tuning for VAD prediction
├── utils/
│   ├── metrics.py           # Evaluation metrics
│   └── visualization.py     # Functions for visualizing results
└── results/
    └── model_outputs/       # Directory to store model outputs
```

## Implementation Approaches

### 1. Original Implementation (26 Categories)

- Two-stage approach: Text → VAD → Emotion
- Zero-shot prediction of VAD values using BART-large-MNLI
- Random Forest classifier for emotion prediction
- End-to-End Accuracy: 32.62%

### 2. Reduced Categories Implementation (4 Categories)

- Same two-stage approach
- Mapped 26 categories to 4 (Angry, Happy, Neutral, Sad)
- End-to-End Accuracy: 47.06%

### 3. Fine-Tuning Implementation

- Added a regression head to pre-trained language models
- Implemented training and evaluation functions

### 4. Multimodal Implementation

- Added audio processing and feature extraction
- Implemented early and late fusion strategies

## Usage

### Running with Original Categories (26)

```bash
python run_emotion_recognition.py --mode train --data_path IEMOCAP_Final.csv --vad_model facebook/bart-large-mnli
```

### Running with Reduced Categories (4)

```bash
python run_reduced_categories.py --vad_model facebook/bart-large-mnli
```

### Running Fine-Tuning

```bash
python run_fine_tuning.py --model_name roberta-base --epochs 5
```

### Running with Fine-Tuned Model

```bash
python run_with_fine_tuned.py --fine_tuned_model_dir results/fine_tuning/run_YYYYMMDD_HHMMSS
```

### Running Multimodal Approach

```bash
python run_multimodal.py --mode train --data_path IEMOCAP_Final.csv --fusion_type early
```

### Visualizing Results

```bash
python visualize_results.py --results_dir results/run_YYYYMMDD_HHMMSS
```

### Comparing Approaches

```bash
python compare_approaches.py --zero_shot_dir results/run_YYYYMMDD_HHMMSS --fine_tuned_dir results/fine_tuned/run_YYYYMMDD_HHMMSS
```

## Results Summary

### Stage 1 (Text to VAD) Performance
- **MSE**: 1.6072
- **RMSE**: 1.2678
- **MAE**: 1.0041

### Stage 2 (VAD to Emotion) Performance
- **26 Categories**: Validation Accuracy = 58.86%
- **4 Categories**: Validation Accuracy = 71.51%

### End-to-End (Text to Emotion) Performance
- **26 Categories**: Test Accuracy = 32.62%
- **4 Categories**: Test Accuracy = 47.06%

## Feature Importance
- **Valence**: 72.71%
- **Arousal**: 12.48%
- **Dominance**: 14.82%

## Requirements

```
torch>=1.8.0
transformers>=4.10.0
scikit-learn>=0.24.0
pandas>=1.2.0
numpy>=1.19.0
matplotlib>=3.3.0
seaborn>=0.11.0
librosa>=0.8.0
tqdm>=4.60.0
scipy>=1.6.0
```

## Documentation

- **EMOD_PROJECT_LOG.md**: Comprehensive log of all activities and results
- **EXPERIMENT_RESULTS.md**: Detailed results of the initial experiments
- **FINE_TUNING.md**: Documentation of the fine-tuning approach

For more details, see the documentation files in the repository.
