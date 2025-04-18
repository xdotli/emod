# Multimodal Emotion Recognition System

This implementation provides a multimodal approach for emotion recognition using the IEMOCAP dataset, combining both text and audio modalities:

1. **Text Modality**: Zero-shot prediction of Valence-Arousal-Dominance (VAD) values from text using pre-trained language models.
2. **Audio Modality**: Extraction of audio features and prediction of VAD values using traditional machine learning models.
3. **Fusion**: Combination of text and audio modalities for improved emotion recognition.

## Project Structure

```
src/
├── data/
│   ├── data_loader.py       # Functions to load and preprocess IEMOCAP data
│   └── data_utils.py        # Utility functions for data handling
├── models/
│   ├── vad_predictor.py     # Text-to-VAD model using pre-trained transformers
│   ├── audio_processor.py   # Audio feature extraction and VAD prediction
│   ├── emotion_classifier.py # VAD-to-emotion classifier
│   ├── multimodal_fusion.py # Fusion of text and audio modalities
│   ├── pipeline.py          # Text-only pipeline
│   └── multimodal_pipeline.py # Multimodal pipeline
├── utils/
│   ├── metrics.py           # Evaluation metrics
│   └── visualization.py     # Functions for visualizing results
└── results/
    └── model_outputs/       # Directory to store model outputs
```

## Usage

### Training and Evaluation

To train and evaluate the multimodal pipeline:

```bash
python run_multimodal.py --mode train --data_path IEMOCAP_Final.csv --output_dir results/multimodal --vad_model facebook/bart-large-mnli --audio_feature_type mfcc --fusion_type early
```

Key arguments:
- `--data_path`: Path to the IEMOCAP_Final.csv file
- `--output_dir`: Directory to save results
- `--vad_model`: Model for text VAD prediction (choices: 'roberta-base', 'facebook/bart-large-mnli')
- `--audio_feature_type`: Type of audio features to extract (choices: 'mfcc', 'spectral', 'wav2vec')
- `--fusion_type`: Type of multimodal fusion (choices: 'early', 'late')
- `--batch_size`: Batch size for processing

### Prediction

To run the multimodal pipeline for prediction:

```bash
python run_multimodal.py --mode predict --data_path IEMOCAP_Final.csv --model_dir results/multimodal/run_YYYYMMDD_HHMMSS
```

## Models

### Text-to-VAD (Stage 1)

The implementation provides two zero-shot approaches for predicting VAD values from text:

1. **RoBERTa-based**: Uses RoBERTa embeddings and cosine similarity to predict VAD values.
2. **BART-based**: Uses BART for natural language inference to predict VAD values.

### Audio-to-VAD

The implementation provides three approaches for extracting audio features:

1. **MFCC**: Mel-frequency cepstral coefficients for audio representation.
2. **Spectral**: Spectral features like centroid, bandwidth, rolloff, and contrast.
3. **Wav2Vec**: Features extracted using the pre-trained Wav2Vec2 model.

These features are then used to train a regression model (Ridge, SVR, or Random Forest) to predict VAD values.

### Multimodal Fusion

Two fusion approaches are implemented:

1. **Early Fusion**: Concatenation of text and audio VAD values before classification.
2. **Late Fusion**: Weighted combination of predictions from separate text and audio classifiers.

## Evaluation Metrics

The system evaluates performance using various metrics:

- **VAD Prediction**: MSE, RMSE, MAE for each VAD dimension
- **Emotion Classification**: Accuracy, F1-score (macro and weighted), confusion matrix

## Visualization

The implementation includes various visualization tools:

- Confusion matrix for emotion classification
- VAD distribution plots
- VAD values by emotion
- Performance comparison between text-only and multimodal approaches

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- librosa
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
