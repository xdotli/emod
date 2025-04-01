# Experimental Report: Two-Stage Emotion Detection (Text -> VAD -> Emotion)

**Date:** April 1, 2024

**Project:** Emotion Recognition from Text via Valence-Arousal-Dominance Prediction

## 1. Introduction

This report details the experimental setup and performance results for a two-stage pipeline designed for emotion recognition from textual data, specifically targeting the IEMOCAP dataset. The pipeline operates in two sequential stages:

1.  **Stage 1 (Text-to-VAD):** A fine-tuned transformer model (`roberta-base`) predicts continuous Valence-Arousal-Dominance (VAD) values from input text utterances.
2.  **Stage 2 (VAD-to-Emotion):** The predicted VAD values are mapped to discrete emotion categories (e.g., "angry", "happy", "neutral", "sad") using a machine learning classifier (RandomForest or SVM, based on configuration).

This approach leverages the continuous VAD space as an intermediate representation, potentially capturing nuances that might be lost in direct text-to-emotion classification.

## 2. Experimental Setup

*   **Dataset:** IEMOCAP (Interactive Emotional Dyadic Motion Capture) dataset, preprocessed using `prepare_iemocap_vad.py` to include text transcripts, ground-truth emotion labels, and corresponding VAD annotations.
*   **Stage 1 Model:** `roberta-base` transformer model fine-tuned for regression on VAD dimensions (implemented in `text_vad.py`).
*   **Stage 2 Model:** A machine learning classifier (likely RandomForest based on defaults/logs) trained on ground-truth VAD values and corresponding emotion labels from the training portion of the dataset (implemented in `VADtoEmotionClassifier` within `vad_emotion_pipeline.py`).
*   **Evaluation:** The pipeline was evaluated on a held-out test set from the IEMOCAP data. Performance metrics were collected for both the VAD prediction stage and the final emotion classification stage.
*   **Software:** Python, PyTorch, Transformers, Scikit-learn, Pandas.
*   **Hardware:** Execution performed on device: `mps` (Apple Silicon GPU).

## 3. Performance Results (Test Set)

The following tables summarize the performance of the pipeline on the test set, as recorded in `logs/pipeline_results.json`.

### 3.1 Stage 1: Text-to-VAD Prediction Performance

This stage evaluates the accuracy of the `roberta-base` model in predicting VAD values from text.

| Metric        | Overall | Valence | Arousal | Dominance |
|---------------|---------|---------|---------|-----------|
| MSE           | 0.1252  | 0.1526  | 0.0993  | 0.1237    |
| RMSE          | 0.3538  | -       | -       | -         |
| MAE           | 0.2874  | -       | -       | -         |
| R² Score      | 0.1795  | 0.2557  | 0.1682  | 0.1147    |

**Description:**

The VAD prediction model achieves moderate performance. The overall R² score of approximately 0.18 indicates that the model explains about 18% of the variance in the VAD values based on the text. Performance varies across dimensions, with Valence being predicted slightly better (R² ≈ 0.26) than Arousal (R² ≈ 0.17) and Dominance (R² ≈ 0.11). Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE) provide measures of the average prediction error magnitude.

### 3.2 Stage 2: VAD-to-Emotion Classification Performance

This stage evaluates the performance of the VAD-to-Emotion classifier (using VAD values predicted by Stage 1) against the ground-truth emotion labels.

**Overall Accuracy:** 46.56%

**Classification Report:**

| Emotion   | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| angry     | 0.588     | 0.757  | 0.662    | 181     |
| happy     | 0.812     | 0.111  | 0.195    | 117     |
| neutral   | 0.335     | 0.580  | 0.424    | 150     |
| sad       | 0.000     | 0.000  | 0.000    | 61      |
|-----------|-----------|--------|----------|---------|
| Macro Avg | 0.434     | 0.362  | 0.320    | 509     |
| Wgt Avg   | 0.494     | 0.466  | 0.405    | 509     |

**Confusion Matrix:**

```
Predicted | angry | happy | neutral | sad
----------|-------|-------|---------|-----
Actual    |
angry     |  137  |   0   |   44    |  0
happy     |   18  |   13  |   86    |  0
neutral   |   60  |   3   |   87    |  0
sad       |   18  |   0   |   43    |  0
```
*(Extracted from `logs/pipeline_results.json`)*

A visualization (`logs/confusion_matrix.png`) is also available.

**Description:**

The overall accuracy of the VAD-to-Emotion stage (and thus the end-to-end pipeline on the test set) is 46.56%. Performance varies significantly across emotion categories:
*   **Angry:** Relatively well-identified with good recall (0.757) and moderate precision (0.588).
*   **Happy:** Poorly identified, with very low recall (0.111) despite high precision (0.812), suggesting the model rarely predicts happy, but when it does, it's often correct. Many 'happy' instances are misclassified as 'neutral' (86 cases).
*   **Neutral:** Moderate recall (0.580) but low precision (0.335), indicating it's often predicted, but frequently confuses other emotions (especially happy and angry) for neutral.
*   **Sad:** Completely missed by the model (0 recall, 0 precision, 0 F1-score). All 'sad' instances are misclassified, primarily as 'neutral'.

The low performance, especially for 'happy' and 'sad', suggests challenges either in the VAD prediction stage (Stage 1 not capturing differentiating features well enough) or in the VAD-to-Emotion mapping stage (Stage 2 classifier struggling with the predicted VAD distributions).
The Macro Average F1-score (0.320) is low, reflecting the poor performance on minority or difficult classes like 'sad' and 'happy'. The Weighted Average F1-score (0.405) is slightly higher due to the better performance on the more frequent 'angry' class.

## 4. Discussion & Conclusion

The two-stage pipeline demonstrates a proof-of-concept for emotion recognition via VAD prediction. However, the test set performance indicates significant room for improvement.

*   **Stage 1:** The VAD prediction R² scores suggest the text-based model captures some VAD-related information but struggles with nuance, particularly for Dominance.
*   **Stage 2:** The final emotion classification accuracy is modest (46.56%), hampered by poor performance on 'happy' and especially 'sad' categories. The confusion matrix highlights a tendency to misclassify 'happy' and 'sad' as 'neutral' or 'angry'.

**Potential Improvements:**

1.  **Stage 1 Model:** Experiment with different transformer architectures, larger models, or VAD-specific pre-training. Incorporate multimodal features (e.g., audio) if available.
2.  **Stage 2 Mapping:** Explore more sophisticated VAD-to-emotion mapping techniques, potentially non-linear models or adjustments to the classification boundaries. Analyze the distribution of predicted VAD values for each true emotion to understand misclassifications.
3.  **Data Augmentation/Balancing:** Address potential class imbalance issues, particularly for 'sad'.
4.  **Error Analysis:** Further investigate the examples provided in `pipeline_results.json` to identify patterns in misclassifications.

In conclusion, while the pipeline structure is sound, further refinement of both stages is necessary to achieve robust emotion recognition performance on the IEMOCAP dataset. 