# EMOD - Emotion Recognition System

A two-stage emotion recognition system for predicting emotions from text and audio data using the IEMOCAP dataset.

## Project Overview

EMOD (Emotion MODeling) is a comprehensive emotion recognition system that implements a two-stage architecture:
1. **Stage 1**: Prediction of Valence-Arousal-Dominance (VAD) dimensions from input features
2. **Stage 2**: Classification of discrete emotions based on VAD predictions

Two variants of the system were implemented:
- **Text-only**: Uses only textual transcripts for emotion recognition
- **Multimodal**: Combines textual and acoustic features with early fusion

## Repository Structure

```
emod/
├── src/
│   ├── data/                # Data processing utilities
│   ├── models/              # Model implementations
│   ├── utils/               # Evaluation and utility functions
│   ├── scripts/             # Helper scripts
│   └── results/             # Experimental results
│       ├── text_only/       # Text-only model results
│       ├── multimodal/      # Multimodal model results
│       └── comparison/      # Comparison between approaches
├── emod.py                  # Text-only model training script
├── emod_multimodal.py       # Multimodal model training script
├── compare_emod.py          # Results comparison script
└── requirements.txt         # Project dependencies
```

## Experimental Setup

### Datasets
- **IEMOCAP**: Interactive Emotional Dyadic Motion Capture Database
- Contains multimodal conversations with emotion annotations
- Features both categorical emotion labels and dimensional VAD ratings

### Model Architecture

#### Stage 1: VAD Prediction
- **Text-only**: Fine-tuned RoBERTa with valence, arousal, and dominance heads
- **Multimodal**: Early fusion of RoBERTa embeddings with acoustic features

#### Stage 2: Emotion Classification
- Ensemble of Random Forest and Gaussian Naive Bayes classifiers
- Maps VAD predictions to discrete emotion categories
- Categories: angry, happy, neutral, sad

## Experiments and Results

Two main experiments were conducted:

### Experiment 1: Text-only Approach
- **Model**: RoBERTa-base fine-tuned for VAD prediction
- **Results**:
  - Valence Prediction: MSE = 0.4892, R² = 0.4760
  - Arousal Prediction: MSE = 0.3957, R² = 0.2176
  - Dominance Prediction: MSE = 0.4931, R² = 0.2359
  - Emotion Classification: Accuracy = 59.41%, F1 (weighted) = 0.6052

### Experiment 2: Multimodal Approach
- **Model**: Early fusion of RoBERTa-base and acoustic features
- **Audio Features**: MFCCs, spectral features, temporal features, pitch-related features
- **Results**:
  - Valence Prediction: MSE = 0.4598, R² = 0.5075
  - Arousal Prediction: MSE = 0.4073, R² = 0.1946
  - Dominance Prediction: MSE = 0.5123, R² = 0.2061
  - Emotion Classification: Accuracy = 61.41%, F1 (weighted) = 0.6179

### Comparison of Approaches

| Metric | Text-only | Multimodal | Improvement |
|--------|-----------|------------|-------------|
| Valence MSE | 0.4892 | 0.4598 | +6.01% |
| Arousal MSE | 0.3957 | 0.4073 | -2.93% |
| Dominance MSE | 0.4931 | 0.5123 | -3.90% |
| Classification Accuracy | 59.41% | 61.41% | +3.36% |
| F1 Score (weighted) | 0.6052 | 0.6179 | +2.09% |

#### Per-Emotion Performance (F1 Score)
| Emotion | Text-only | Multimodal | Improvement |
|---------|-----------|------------|-------------|
| Angry | 0.6957 | 0.7213 | +3.68% |
| Happy | 0.6953 | 0.7115 | +2.33% |
| Neutral | 0.4008 | 0.4262 | +6.33% |
| Sad | 0.4932 | 0.4381 | -11.16% |

## Key Findings

1. **VAD Prediction**:
   - Multimodal approach excels at valence prediction (+6.01%)
   - Text-only approach is better for arousal (+2.93%) and dominance (+3.90%)

2. **Emotion Classification**:
   - Multimodal approach improves overall accuracy by 3.36%
   - Performance gains in "Angry," "Happy," and "Neutral" emotions
   - Text-only approach performs better for "Sad" emotion detection

3. **Tradeoffs**:
   - Multimodal approach provides modest improvements at the cost of increased complexity
   - Text features contribute most significantly to emotion recognition
   - Audio features help primarily with valence prediction

## Usage

### Requirements

```
pip install -r requirements.txt
```

### Text-only Model

```
python emod.py --data_path IEMOCAP_Final.csv --output_dir src/results/text_only --epochs 10 --save_model
```

### Multimodal Model

```
python emod_multimodal.py --data_path IEMOCAP_Final.csv --audio_base_path Datasets/IEMOCAP_full_release --output_dir src/results/multimodal --fusion_type early --epochs 10 --save_model
```

### Comparing Results

```
python src/scripts/compare_emod.py --text_results src/results/text_only --multimodal_results src/results/multimodal --output_dir src/results/comparison
```
