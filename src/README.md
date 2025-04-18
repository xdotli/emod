# Two-Stage Emotion Recognition System

This implementation provides a two-stage approach for emotion recognition using the IEMOCAP dataset:

1. **Stage 1 (Text-to-VAD)**: Zero-shot prediction of Valence-Arousal-Dominance (VAD) values from text using pre-trained language models.
2. **Stage 2 (VAD-to-Emotion)**: Classification of emotions from VAD values using a Random Forest classifier.

## Project Structure

```
src/
├── data/
│   ├── data_loader.py       # Functions to load and preprocess IEMOCAP data
│   └── data_utils.py        # Utility functions for data handling
├── models/
│   ├── vad_predictor.py     # Text-to-VAD model using pre-trained transformers
│   ├── emotion_classifier.py # VAD-to-emotion classifier
│   └── pipeline.py          # End-to-end pipeline combining both stages
├── utils/
│   ├── metrics.py           # Evaluation metrics
│   └── visualization.py     # Functions for visualizing results
└── results/
    └── model_outputs/       # Directory to store model outputs
```

## Usage

### Training and Evaluation

To train and evaluate the complete pipeline:

```bash
python src/main.py --data_path IEMOCAP_Final.csv --output_dir results --vad_model facebook/bart-large-mnli
```

Key arguments:
- `--data_path`: Path to the IEMOCAP_Final.csv file
- `--output_dir`: Directory to save results
- `--vad_model`: Model for VAD prediction (choices: 'roberta-base', 'facebook/bart-large-mnli')
- `--batch_size`: Batch size for processing
- `--n_estimators`: Number of trees in Random Forest
- `--skip_vad_eval`: Skip VAD evaluation (faster)

### Running the Pipeline

To run the pipeline in different modes:

```bash
# Training mode
python run.py --mode train --data_path IEMOCAP_Final.csv --output_dir results

# Prediction mode
python run.py --mode predict --data_path IEMOCAP_Final.csv --model_dir results/run_YYYYMMDD_HHMMSS
```

## Models

### Text-to-VAD (Stage 1)

The implementation provides two zero-shot approaches for predicting VAD values:

1. **RoBERTa-based**: Uses RoBERTa embeddings and cosine similarity to predict VAD values.
2. **BART-based**: Uses BART for natural language inference to predict VAD values.

### VAD-to-Emotion (Stage 2)

A Random Forest classifier is trained to predict emotion labels from VAD values.

## Evaluation Metrics

The system evaluates performance using various metrics:

- **VAD Prediction**: MSE, RMSE, MAE for each VAD dimension
- **Emotion Classification**: Accuracy, F1-score (macro and weighted), confusion matrix

## Visualization

The implementation includes various visualization tools:

- Confusion matrix for emotion classification
- VAD distribution plots
- VAD values by emotion
- t-SNE visualization of VAD values

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
