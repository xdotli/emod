# EMOD - Emotion Recognition System

This repository implements a two-stage emotion recognition system using the IEMOCAP dataset. The system first converts input to valence-arousal-dominance (VAD) tuples, then categorizes emotions based on these values.

## Project Structure

```
./
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
├── scripts/
│   ├── run_emotion_recognition.py  # Run original implementation (26 categories)
│   ├── run_reduced_categories.py   # Run with reduced categories (4)
│   ├── run_fine_tuning.py          # Run fine-tuning for VAD prediction
│   ├── run_with_fine_tuned.py      # Run with fine-tuned model
│   ├── run_multimodal.py           # Run multimodal approach
│   ├── visualize_results.py        # Visualize results
│   ├── compare_approaches.py       # Compare different approaches
│   ├── simple_fine_tune.py         # Simplified fine-tuning script
│   ├── preprocess_reduced_categories.py # Preprocess data with reduced categories
│   └── evaluate_reduced_categories.py   # Evaluate reduced categories approach
├── docs/
│   ├── EMOD_PROJECT_LOG.md         # Comprehensive log of all activities
│   ├── EXPERIMENT_RESULTS.md       # Detailed results of the initial experiments
│   └── FINE_TUNING.md              # Documentation of the fine-tuning approach
├── results/
│   └── model_outputs/              # Directory to store model outputs
└── requirements.txt                # Project dependencies
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
python scripts/run_emotion_recognition.py --mode train --data_path IEMOCAP_Final.csv --vad_model facebook/bart-large-mnli
```

### Running with Reduced Categories (4)

```bash
python scripts/run_reduced_categories.py --vad_model facebook/bart-large-mnli
```

### Running Fine-Tuning

```bash
python scripts/run_fine_tuning.py --model_name roberta-base --epochs 5
```

### Running with Fine-Tuned Model

```bash
python scripts/run_with_fine_tuned.py --fine_tuned_model_dir results/fine_tuning/run_YYYYMMDD_HHMMSS
```

### Running Multimodal Approach

```bash
python scripts/run_multimodal.py --mode train --data_path IEMOCAP_Final.csv --fusion_type early
```

### Visualizing Results

```bash
python scripts/visualize_results.py --results_dir results/run_YYYYMMDD_HHMMSS
```

### Comparing Approaches

```bash
python scripts/compare_approaches.py --zero_shot_dir results/run_YYYYMMDD_HHMMSS --fine_tuned_dir results/fine_tuned/run_YYYYMMDD_HHMMSS
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

## Documentation

For more detailed information, see the documentation files in the `docs/` directory:

- **EMOD_PROJECT_LOG.md**: Comprehensive log of all activities and results
- **EXPERIMENT_RESULTS.md**: Detailed results of the initial experiments
- **FINE_TUNING.md**: Documentation of the fine-tuning approach
