# Emotion Recognition using VAD Prediction

This repository implements emotion recognition from text using a two-step approach:

1. First, predict continuous VAD (Valence-Arousal-Dominance) values from text using a transformer model
2. Then, map these VAD values to emotion categories using either a rule-based approach or a machine learning classifier

## Directory Structure

```
.
├── data/                   # Processed datasets
├── checkpoints/            # Saved models
├── logs/                   # Training logs and evaluation results
├── prepare_iemocap_vad.py  # Dataset preparation script
├── text_vad.py             # Text to VAD prediction model
├── vad_emotion_pipeline.py # Complete pipeline for emotion prediction
├── README.md               # This file
```

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn
```

## Usage

### 1. Prepare the IEMOCAP Dataset

Process the IEMOCAP dataset to extract utterances, transcripts, emotion labels, and VAD values:

```bash
python prepare_iemocap_vad.py --iemocap_dir path/to/IEMOCAP_full_release --output_dir data
```

This will create `data/iemocap_vad.csv` with the processed data.

### 2. Train the Text-to-VAD Model

Train a transformer model to predict VAD values from text:

```bash
python text_vad.py --data_path data/iemocap_vad.csv --model_name roberta-base --num_epochs 10
```

Options:

- `--data_path`: Path to the processed dataset CSV
- `--model_name`: Pre-trained model name from Hugging Face (default: roberta-base)
- `--batch_size`: Batch size for training (default: 16)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--num_epochs`: Number of training epochs (default: 10)

The trained model will be saved to `checkpoints/text_vad_best.pt`.

### 3. Run the Complete Pipeline

Run the full text-to-emotion pipeline with either rule-based or ML-based VAD-to-emotion mapping:

```bash
# Using rule-based mapping
python vad_emotion_pipeline.py --data_path data/iemocap_vad.csv --vad_model_path checkpoints/text_vad_best.pt

# Using ML classifier
python vad_emotion_pipeline.py --data_path data/iemocap_vad.csv --vad_model_path checkpoints/text_vad_best.pt --use_ml_classifier --classifier_type rf
```

Options:

- `--data_path`: Path to the processed dataset CSV
- `--vad_model_name`: Pre-trained model name (default: roberta-base)
- `--vad_model_path`: Path to the trained VAD model
- `--use_ml_classifier`: Use ML classifier for VAD to emotion mapping
- `--classifier_type`: Type of classifier (rf for RandomForest, svm for SVM)
- `--classifier_path`: Path to a trained classifier (optional)

## VAD-to-Emotion Mapping

The repository includes two approaches for mapping VAD values to emotion categories:

1. **Rule-based mapping**: Uses predefined rules to map VAD values to emotions based on their position in the 3D VAD space.
2. **ML-based mapping**: Trains a classifier (RandomForest or SVM) to learn the mapping from VAD values to emotions.

## Evaluation Metrics

The pipeline evaluates both:

1. VAD prediction quality using MSE, RMSE, MAE, and R²
2. Emotion classification quality using accuracy, precision, recall, F1 score, and confusion matrix

## Customization

- To use a different transformer model, change the `--model_name` parameter (e.g., `bert-base-uncased`, `distilroberta-base`)
- To use a different dataset, prepare a CSV file with columns: text, emotion, valence, arousal, dominance
- To customize the VAD-to-emotion mapping rules, modify the `vad_to_emotion()` function in `text_vad.py`
