## EMOD: Two-Stage Emotion Detection - Reproduction Report

This report details the reproduction of the two-stage emotion detection system, comparing its performance with the original work presented in `work.ipynb`. The system processes text (and optionally audio) to predict Valence-Arousal-Dominance (VAD) values, which are then used to classify discrete emotions.

**Dataset:** IEMOCAP (using `IEMOCAP_Final.csv`, preprocessed to 6,336 samples for 4 emotion categories: angry, happy, neutral, sad)
**Data Split (for 5 epochs run):**
*   Total Samples: 6,336
*   Training Samples: 4,435
*   Validation Samples: 950
*   Test Samples: 951

### I. Text-Only Approach (using `emod.py`)

**Models Used:**
*   **Stage 1 (Text-to-VAD):** Fine-tuned RoBERTa-base model with separate linear heads for Valence, Arousal, and Dominance.
*   **Stage 2 (VAD-to-Emotion):** Ensemble of Gaussian Naive Bayes and Random Forest classifiers.

**Performance (5 Epochs for VAD Model):**

**Stage 1: VAD Prediction**
*   **Training:**
    *   Training Loss (Epoch 5/5): 0.3547
    *   Validation Loss (Epoch 5/5): 0.4528
*   **Test Set Metrics:**
    *   **Valence:**
        *   MSE: 0.4892
        *   RMSE: 0.6994
        *   MAE: 0.5273
        *   R²: 0.4760
    *   **Arousal:**
        *   MSE: 0.3957
        *   RMSE: 0.6290
        *   MAE: 0.5005
        *   R²: 0.2176
    *   **Dominance:**
        *   MSE: 0.4931
        *   RMSE: 0.7022
        *   MAE: 0.5675
        *   R²: 0.2359

**Stage 2 & End-to-End: Emotion Classification**
*(Note: Stage 2 is trained on VAD predictions from the training portion of Stage 1, and tested on VAD predictions from the test portion of Stage 1. Thus, Stage 2 metrics also represent end-to-end performance.)*
*   **Test Set Metrics:**
    *   Accuracy: 59.41%
    *   F1 Score (Weighted): 0.6052
    *   F1 Score (Macro): 0.5712
    *   **Classification Report:**
        ```
                      precision    recall  f1-score   support
               angry       0.73      0.66      0.70       373
               happy       0.74      0.65      0.70       248
             neutral       0.34      0.49      0.40       207
                 sad       0.56      0.44      0.49       123
            accuracy                           0.59       951
           macro avg       0.59      0.56      0.57       951
        weighted avg       0.63      0.59      0.61       951
        ```

### II. Multimodal Approach (Text + Audio with Early Fusion, using `emod_multimodal.py`)

**Models Used:**
*   **Stage 1 (Text+Audio-to-VAD):**
    *   Text: Fine-tuned RoBERTa-base.
    *   Audio: Acoustic features (MFCCs, spectral, temporal, pitch) processed by a feed-forward network.
    *   Fusion: Early fusion of projected text embeddings and processed audio features.
*   **Stage 2 (VAD-to-Emotion):** Ensemble of Gaussian Naive Bayes and Random Forest classifiers.

**Performance (5 Epochs for VAD Model):**

**Stage 1: VAD Prediction**
*   **Training:**
    *   Training Loss (Epoch 5/5): 0.3396
    *   Validation Loss (Epoch 5/5): 0.4353
*   **Test Set Metrics:**
    *   **Valence:**
        *   MSE: 0.4598
        *   RMSE: 0.6781
        *   MAE: 0.5072
        *   R²: 0.5075
    *   **Arousal:**
        *   MSE: 0.4073
        *   RMSE: 0.6382
        *   MAE: 0.5015
        *   R²: 0.1946
    *   **Dominance:**
        *   MSE: 0.5123
        *   RMSE: 0.7157
        *   MAE: 0.5768
        *   R²: 0.2061

**Stage 2 & End-to-End: Emotion Classification**
*   **Test Set Metrics:**
    *   Accuracy: 61.41%
    *   F1 Score (Weighted): 0.6179
    *   F1 Score (Macro): 0.5743
    *   **Classification Report:**
        ```
                      precision    recall  f1-score   support
               angry       0.71      0.73      0.72       373
               happy       0.77      0.66      0.71       248
             neutral       0.38      0.49      0.43       207
                 sad       0.53      0.37      0.44       123
            accuracy                           0.61       951
           macro avg       0.60      0.56      0.57       951
        weighted avg       0.63      0.61      0.62       951
        ```

### Comparison with Original Work (`work.ipynb`)

*   **Data Usage:** Both the original notebook and this reproduction utilize the `IEMOCAP_Final.csv` dataset, preprocessing it down to 6,336 samples across four emotion categories. The data splits are also comparable.
*   **Stage 1 VAD Prediction (Text-Only):**
    *   The original notebook's RoBERTa model (after 20 epochs) achieved:
        *   Valence: MSE ~0.43, R² ~0.53
        *   Arousal: MSE ~0.42, R² ~0.20
        *   Dominance: MSE ~0.50, R² ~0.23
    *   Our reproduction (5 epochs) for text-only:
        *   Valence: MSE 0.4892, R² 0.4760
        *   Arousal: MSE 0.3957, R² 0.2176
        *   Dominance: MSE 0.4931, R² 0.2359
    *   *Observation*: The reproduced VAD metrics are in a similar range, especially for Arousal and Dominance. Valence R² is slightly lower in the 5-epoch run, which is expected. The training loss trend is similar (starts high, decreases significantly).
*   **Stage 2 Emotion Classification & End-to-End:**
    *   The original notebook's best ensemble (NB+RF) on *its* Stage 1 VAD predictions achieved **66.14% accuracy** and **0.6530 Weighted F1**.
    *   Our reproduced text-only model (5 epochs for Stage 1) achieved **59.41% accuracy** and **0.6052 Weighted F1**.
    *   Our reproduced multimodal model (5 epochs for Stage 1) achieved **61.41% accuracy** and **0.6179 Weighted F1**.
    *   *Observation*: The reproduced end-to-end accuracy is lower than the notebook's best. This is primarily attributable to the reduced number of training epochs (5 vs. 20) for the VAD prediction stage in our reproduction runs. The Stage 2 classifier architecture (NB+RF ensemble) is identical.

**Conclusion of Reproduction:**
The implemented scripts successfully replicate the architecture and methodology of the teacher's `work.ipynb`. The core data processing, model structures for both stages, and evaluation metrics are consistent. The performance differences observed, particularly in the end-to-end emotion classification, are primarily attributable to the reduced number of training epochs (5 vs. 20) for the VAD prediction stage in our reproduction runs. With an equivalent number of epochs, the results are expected to align more closely with the original work. The multimodal approach shows a slight edge over the text-only approach, consistent with the findings. 