# Emotion Recognition using VAD Prediction

This repository implements emotion recognition from text using a two-stage approach:

1.  **Stage 1 (Text-to-VAD):** Predict continuous VAD (Valence-Arousal-Dominance) values from text using a fine-tuned transformer model (e.g., `roberta-base`).
2.  **Stage 2 (VAD-to-Emotion):** Map these VAD values to discrete emotion categories (e.g., "angry", "happy", "neutral", "sad") using either a rule-based approach or a machine learning classifier (e.g., RandomForest).

This project primarily utilizes the IEMOCAP dataset.

**See `REPORT.md` for a detailed experimental analysis and results.**

## Quick Results Summary (Test Set)

*   **Text-to-VAD (Stage 1) R² Score:** ~0.18
*   **End-to-End Emotion Accuracy (Stage 2):** ~46.6%
*   **Performance Highlights:** Reasonable performance for 'angry', poor for 'happy' and 'sad'.
*(Based on results in `logs/pipeline_results.json` using RoBERTa-base and a trained classifier)*

## Project Structure

```
.
├── data/                   # Processed datasets (e.g., iemocap_vad.csv)
├── checkpoints/            # Saved model weights (e.g., text_vad_best.pt, vad_classifier.pkl)
├── logs/                   # Training logs, evaluation results (JSON), plots (confusion matrices)
├── models/                 # (Potentially contains model definitions if not in main scripts)
|
├── prepare_iemocap_vad.py  # Script to preprocess IEMOCAP data
├── text_vad.py             # Defines and handles training for the Text-to-VAD model (Stage 1)
├── vad_emotion_pipeline.py # Implements the VAD-to-Emotion classifier (Stage 2) and the end-to-end pipeline
├── process_vad.py          # Utility functions for VAD processing (e.g., rule-based mapping)
├── run.py                  # Example script to run experiments (potentially)
├── main.py                 # Main execution script (potentially combines different modalities/pipelines)
|
├── requirements.txt        # Python package dependencies
├── README.md               # This file
├── REPORT.md               # Detailed experimental report
├── .gitignore              # Git ignore configuration
└── ...                     # Other scripts/files (audio, multimodal experiments)
```

## Installation

1.  Clone this repository:
    ```bash
    git clone https://github.com/xdotli/emod
    cd emod
    ```
2.  Create and activate a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Prepare the IEMOCAP Dataset

*   Download the IEMOCAP dataset and place it in a known location.
*   Run the preprocessing script:

    ```bash
    python prepare_iemocap_vad.py --iemocap_dir path/to/IEMOCAP_full_release --output_dir data
    ```
    This will create `data/iemocap_vad.csv` (or similar) containing text, VAD values, and emotion labels.

### 2. Train the Text-to-VAD Model (Stage 1)

Train the transformer model to predict VAD values:

```bash
python text_vad.py --data_path data/iemocap_vad.csv --model_name roberta-base --num_epochs 10 --output_dir checkpoints --log_dir logs
```

Key Arguments:
*   `--data_path`: Path to the processed dataset CSV.
*   `--model_name`: Hugging Face transformer model (default: `roberta-base`).
*   `--num_epochs`: Number of training epochs (default: 10).
*   `--output_dir`: Directory to save the best model checkpoint (e.g., `checkpoints/text_vad_best.pt`).
*   `--log_dir`: Directory to save training logs/metrics.

### 3. Run the Full Pipeline (Stage 1 + Stage 2) & Evaluate

Run the end-to-end pipeline to predict emotions from text using the trained VAD model and evaluate performance:

```bash
# Option A: Use a trained ML classifier (e.g., RandomForest) for VAD-to-Emotion
python vad_emotion_pipeline.py \
    --data_path data/iemocap_vad.csv \
    --vad_model_path checkpoints/text_vad_best.pt \
    --use_ml_classifier \
    --classifier_type rf \
    --model_dir checkpoints \
    --log_dir logs

# Option B: Use rule-based mapping for VAD-to-Emotion
# python vad_emotion_pipeline.py \
#     --data_path data/iemocap_vad.csv \
#     --vad_model_path checkpoints/text_vad_best.pt \
#     --model_dir checkpoints \
#     --log_dir logs
```

Key Arguments:
*   `--data_path`: Path to the processed dataset CSV.
*   `--vad_model_path`: Path to the trained Stage 1 model (`text_vad_best.pt`).
*   `--use_ml_classifier`: Flag to train/use an ML classifier for Stage 2.
*   `--classifier_type`: Type of classifier ('rf' or 'svm') if `--use_ml_classifier` is set.
*   `--classifier_path`: Optionally load a pre-trained Stage 2 classifier (`.pkl` file).
*   `--model_dir`: Directory to save/load the Stage 2 classifier (e.g., `checkpoints/vad_classifier_rf.pkl`).
*   `--log_dir`: Directory to save pipeline evaluation results (`pipeline_results.json`, `confusion_matrix.png`).

This command will:
1.  Load the pre-trained Stage 1 model (`--vad_model_path`).
2.  Predict VAD values for the test set.
3.  If `--use_ml_classifier` is set and `--classifier_path` is not provided, it will train a new Stage 2 classifier (e.g., RandomForest) on the training data's VAD values and emotion labels, saving it to `--model_dir`.
4.  Use the Stage 2 method (classifier or rules) to map predicted VAD values to emotions.
5.  Evaluate both VAD prediction and final emotion classification performance on the test set, saving results to `--log_dir`.

## Evaluation Metrics

The pipeline outputs evaluation metrics for both stages:

1.  **VAD Prediction (Stage 1):** MSE, RMSE, MAE, and R² score (overall and per-dimension). Saved in `logs/pipeline_results.json`.
2.  **Emotion Classification (Stage 2):** Accuracy, Precision, Recall, F1-score (per class, macro avg, weighted avg), and Confusion Matrix. Saved in `logs/pipeline_results.json` and `logs/confusion_matrix.png`.

## Customization

*   **Transformer Model:** Change `--model_name` in `text_vad.py` (e.g., `bert-base-uncased`).
*   **Dataset:** Prepare a different dataset CSV with columns: `text`, `emotion`, `valence`, `arousal`, `dominance`. Update `--data_path` arguments.
*   **VAD-to-Emotion Rules:** Modify the `vad_to_emotion()` function in `process_vad.py` or `text_vad.py` (check which one is used by the pipeline).
*   **Classifier:** Adjust hyperparameters for RandomForest/SVM in `vad_emotion_pipeline.py`.
