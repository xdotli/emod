# Two-Stage Emotion Recognition System

This repository contains a two-stage emotion recognition system that:
1. First converts audio and text modalities to valence-arousal-dominance (VAD) tuples
2. Then categorizes emotions based on those VAD values

## Overview

The system is designed to recognize emotions from the IEMOCAP dataset using a two-stage approach:

1. **Stage 1: VAD Prediction**
   - Text-based VAD prediction using a transformer model (RoBERTa)
   - Audio-based VAD prediction using a CNN model
   - Multimodal fusion of text and audio for improved VAD prediction

2. **Stage 2: Emotion Classification**
   - Rule-based emotion classification from VAD values
   - Neural network-based emotion classification from VAD values

## Project Structure

```
replicate_1/
├── data/
│   ├── __init__.py
│   └── data_processor.py
├── models/
│   ├── __init__.py
│   ├── text_vad_model.py
│   ├── audio_vad_model.py
│   ├── multimodal_vad_model.py
│   └── emotion_classifier.py
├── utils/
│   ├── __init__.py
│   ├── vad_emotion_mapping.py
│   └── visualization.py
├── main.py
├── demo.py
└── README.md
```

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- torchaudio
- scikit-learn
- pandas
- numpy
- matplotlib

## Usage

### Training

To train the emotion recognition system, run the `main.py` script:

```bash
python main.py --data_path IEMOCAP_Final.csv --mode text --batch_size 32 --epochs 20 --output_dir output
```

Options:
- `--data_path`: Path to the IEMOCAP_Final.csv file
- `--mode`: Mode of operation (text, audio, multimodal)
- `--batch_size`: Batch size for training
- `--epochs`: Number of training epochs
- `--learning_rate`: Learning rate
- `--seed`: Random seed
- `--output_dir`: Output directory for models and results

### Demo

To test the emotion recognition system on a new text, run the `demo.py` script:

```bash
python demo.py --text "I am feeling very happy today!" --model_dir output
```

Options:
- `--text`: Text to analyze
- `--model_dir`: Directory containing trained models
- `--use_rule_based`: Use rule-based emotion classification instead of trained classifier

## Models

### Text VAD Model

The text-based VAD prediction model uses a RoBERTa transformer to extract features from text and predict valence, arousal, and dominance values. The model architecture consists of:

1. RoBERTa base model for feature extraction
2. Shared fully connected layers
3. Separate branches for valence, arousal, and dominance prediction

### Audio VAD Model

The audio-based VAD prediction model uses a CNN to extract features from mel spectrograms and predict valence, arousal, and dominance values. The model architecture consists of:

1. CNN layers for feature extraction from mel spectrograms
2. Shared fully connected layers
3. Separate branches for valence, arousal, and dominance prediction

### Multimodal VAD Model

The multimodal VAD prediction model combines the text and audio modalities for improved VAD prediction. The model architecture consists of:

1. Pretrained text VAD model
2. Pretrained audio VAD model
3. Fusion layers to combine text and audio VAD predictions

### Emotion Classifier

The emotion classifier takes VAD values as input and predicts emotion categories (happy, sad, angry, neutral). Two approaches are implemented:

1. **Rule-based classifier**: Uses predefined VAD thresholds for each emotion
2. **Neural network classifier**: Learns to map VAD values to emotion categories

## Evaluation

The system is evaluated using the following metrics:

- **VAD prediction**: MSE, RMSE, MAE, R²
- **Emotion classification**: Accuracy, F1-score, confusion matrix

## References

- IEMOCAP dataset: https://sail.usc.edu/iemocap/
- Valence-Arousal-Dominance (VAD) model of emotion: Russell, J. A. (1980). A circumplex model of affect. Journal of Personality and Social Psychology, 39(6), 1161-1178.
