# EMOD: Enhanced Multimodal Emotion Detection

This repository implements emotion recognition using a two-stage approach with state-of-the-art models and LLM integration:

1.  **Stage 1 (Modality-to-VAD):** Predict continuous VAD (Valence-Arousal-Dominance) values from text and audio using fine-tuned transformer models.
2.  **Stage 2 (VAD-to-Emotion):** Map these VAD values to discrete emotion categories (e.g., "angry", "happy", "neutral", "sad") using either a rule-based approach, a machine learning classifier, or advanced LLMs.

This project primarily utilizes the IEMOCAP dataset and has been enhanced with state-of-the-art models and LLM integration via OpenRouter.

**See `docs/report.md` for a comprehensive report on the enhancements and results.**

## Quick Results Summary (Test Set)

*   **Text-to-VAD (Stage 1) R² Score:** ~0.18
*   **End-to-End Emotion Accuracy (Stage 2):** ~46.6%
*   **Performance Highlights:** Reasonable performance for 'angry', poor for 'happy' and 'sad'.
*(Based on results in `logs/pipeline_results.json` using RoBERTa-base and a trained classifier)*

## Project Structure

```
emod/
├── benchmark_llms.py        # Benchmarking script for LLMs
├── benchmark_llms_real.py   # Benchmarking with real emotional text
├── data/                    # Processed datasets and samples
│   └── emotional_samples.csv # Sample emotional text for benchmarking
├── checkpoints/             # Saved model weights
├── docs/                    # Documentation
│   ├── images/              # Images for documentation
│   └── report.md            # Comprehensive project report
├── logs/                    # Training logs, evaluation results, plots
├── main.py                  # Main implementation of the two-stage approach
├── models/                  # Core models
│   ├── __init__.py
│   ├── audio_model.py       # Audio processing models
│   ├── fusion_model.py      # Multimodal fusion models
│   └── text_model.py        # Text processing models
├── prepare_iemocap_vad.py   # Script to preprocess IEMOCAP data
├── process_vad.py           # Utility functions for VAD processing
├── requirements.txt         # Project dependencies
├── run.py                   # Script to run the original pipeline
├── run_enhanced.py          # Script to run the enhanced pipeline
├── run_sota_analysis.py     # Script to run SOTA analysis
├── sota_models/             # State-of-the-art models
│   ├── __init__.py
│   ├── emotion_detection.py # SOTA emotion detection models
│   ├── integration.py       # Integration with existing pipeline
│   └── transcription.py     # SOTA transcription models
├── text_vad.py              # Text-to-VAD model training
├── utils/                   # Utility functions
│   ├── __init__.py
│   ├── config.py            # Configuration utilities
│   └── openrouter_client.py # OpenRouter API client
├── vad_emotion_pipeline.py  # VAD-to-Emotion pipeline
├── vad_to_emotion_model.py  # ML model for VAD-to-Emotion mapping
└── .env.example             # Example environment variables file
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

### 1. Set Up Environment Variables

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
# Edit .env to add your OpenRouter API key and other settings
```

### 2. Prepare the IEMOCAP Dataset

*   Download the IEMOCAP dataset and place it in a known location.
*   Run the preprocessing script:

    ```bash
    python prepare_iemocap_vad.py --iemocap_dir path/to/IEMOCAP_full_release --output_dir data
    ```
    This will create `data/iemocap_vad.csv` (or similar) containing text, VAD values, and emotion labels.

### 3. Run SOTA Analysis

#### For analyzing text:

```bash
./run_sota_analysis.py --text "I'm feeling really happy today!" --pretty
```

#### For analyzing audio:

```bash
./run_sota_analysis.py --audio path/to/audio.wav --pretty
```

#### For comparing LLM analyses:

```bash
./run_sota_analysis.py --text "I'm feeling really happy today!" --compare-llms --pretty
```

### 4. Run Enhanced Pipeline

```bash
./run_enhanced.py --evaluate --use-sota
```

### 5. Benchmark LLMs

```bash
./benchmark_llms_real.py --models anthropic/claude-2.0 openai/gpt-4o-2024-05-13 openai/gpt-4 --data-path data/emotional_samples.csv
```

### 6. Original Pipeline (Legacy)

#### Train the Text-to-VAD Model (Stage 1)

Train the transformer model to predict VAD values:

```bash
python text_vad.py --data_path data/iemocap_vad.csv --model_name roberta-base --num_epochs 10 --output_dir checkpoints --log_dir logs
```

Key Arguments:
*   `--data_path`: Path to the processed dataset CSV.
*   `--model_name`: Hugging Face transformer model (default: `roberta-base`).
*   `--num_epochs`: Number of training epochs (default: 10).
*   `--output_dir`: Directory to save the best model checkpoint.
*   `--log_dir`: Directory to save training logs/metrics.

#### Run the Full Pipeline (Stage 1 + Stage 2) & Evaluate

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
python vad_emotion_pipeline.py \
    --data_path data/iemocap_vad.csv \
    --vad_model_path checkpoints/text_vad_best.pt \
    --model_dir checkpoints \
    --log_dir logs
```

## Evaluation Metrics

The pipeline outputs evaluation metrics for both stages:

1.  **VAD Prediction (Stage 1):** MSE, RMSE, MAE, and R² score (overall and per-dimension). Saved in `logs/pipeline_results.json`.
2.  **Emotion Classification (Stage 2):** Accuracy, Precision, Recall, F1-score (per class, macro avg, weighted avg), and Confusion Matrix. Saved in `logs/pipeline_results.json` and `logs/confusion_matrix.png`.

## Enhanced Features

### State-of-the-Art Models

* **Text Emotion Detection**: DeBERTa-v3 model fine-tuned for emotion classification
* **Audio Emotion Detection**: AST (Audio Spectrogram Transformer) model for audio emotion recognition
* **Transcription**: Whisper Large v3 for state-of-the-art speech-to-text conversion

### LLM Integration

* **OpenRouter Client**: Access to powerful LLMs like Claude 3.7 Sonnet, GPT-4o, and DeepSeek
* **Emotion Analysis**: Deep insights into emotional content using LLMs
* **Multi-LLM Comparison**: Compare analyses from different LLMs

### Multimodal Fusion

* **Advanced Fusion Techniques**: Combine text and audio modalities for improved accuracy
* **Attention Mechanisms**: Dynamic weighting of modalities based on confidence

## Customization

*   **LLM Selection:** Change the LLM model in `.env` or via command-line arguments.
*   **Transformer Model:** Change `--model_name` in `text_vad.py` (e.g., `bert-base-uncased`).
*   **Dataset:** Prepare a different dataset CSV with columns: `text`, `emotion`, `valence`, `arousal`, `dominance`.
*   **VAD-to-Emotion Rules:** Modify the `vad_to_emotion()` function in `process_vad.py`.
*   **Fusion Weights:** Adjust the weights in `sota_models/emotion_detection.py`.

## Performance

Based on our benchmark results with real IEMOCAP dataset utterances, here's how the different LLMs performed:

### GPT-4o (OpenAI)

- **Accuracy**: 50.0%
- **F1 Score (Macro)**: 51.4%
- **F1 Score (Weighted)**: 51.2%
- **Average Response Time**: 1.02 seconds

### Claude 2.0 (Anthropic)

- **Accuracy**: 50.0%
- **F1 Score (Macro)**: 45.2%
- **F1 Score (Weighted)**: 48.3%
- **Average Response Time**: 3.16 seconds

### GPT-4 (OpenAI)

- **Accuracy**: 45.0%
- **F1 Score (Macro)**: 49.0%
- **F1 Score (Weighted)**: 46.0%
- **Average Response Time**: 2.67 seconds

The relatively low performance highlights the challenge of emotion recognition in real-world conversational data, where context, tone, and other non-verbal cues play a crucial role. GPT-4o still outperforms the other models in terms of F1 score and response time.

See the comprehensive report for detailed analysis and visualizations.
