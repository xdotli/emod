# Emotion Recognition Experiment Results

This document contains the results of the emotion recognition experiments using the IEMOCAP dataset.

## Experiment Setup

The experiment used a two-stage approach for emotion recognition:

1. **Stage 1**: Zero-shot prediction of Valence-Arousal-Dominance (VAD) values from text using pre-trained language models.
2. **Stage 2**: Classification of emotions from VAD values using a Random Forest classifier.

### Configuration

```json
{
  "data_path": "IEMOCAP_Final.csv",
  "output_dir": "results",
  "vad_model": "facebook/bart-large-mnli",
  "batch_size": 16,
  "random_state": 42,
  "n_estimators": 100,
  "test_size": 0.2,
  "val_size": 0.1,
  "save_data": false,
  "skip_vad_eval": true
}
```

## Results

### Emotion Classification Performance

The emotion classifier achieved the following performance on the validation set:

- **Accuracy**: 58.86%
- **F1 (macro)**: 20.06%
- **F1 (weighted)**: 55.61%

#### Performance by Emotion Class

| Emotion | Precision | Recall | F1-score | Support |
|---------|-----------|--------|----------|---------|
| Anger | 0.6667 | 0.3966 | 0.4973 | 116 |
| Excited | 0.6342 | 0.7590 | 0.6910 | 249 |
| Fear | 0.0000 | 0.0000 | 0.0000 | 12 |
| Frustration | 0.5678 | 0.7524 | 0.6472 | 412 |
| Happiness | 0.5833 | 0.1628 | 0.2545 | 43 |
| Neutral state | 0.3333 | 0.0833 | 0.1333 | 48 |
| Sadness | 0.5303 | 0.3017 | 0.3846 | 116 |
| Other* | 0.0000 | 0.0000 | 0.0000 | 7 |

*Other includes: Other, Other amused, Other melancolia, Other melancolic, Other pride, Surprise

### Feature Importance

The Random Forest classifier identified the following feature importance:

- **Valence**: 61.91%
- **Arousal**: 18.60%
- **Dominance**: 19.49%

This indicates that valence (positive/negative sentiment) is the most important feature for emotion classification, followed by dominance and arousal.

## Visualizations

### Confusion Matrix

![Confusion Matrix](visualizations/confusion_matrix.png)

The confusion matrix shows that the model performs best on the most frequent emotion classes (Excited and Frustration), while struggling with less frequent emotions. There is some confusion between similar emotions, such as Anger and Frustration.

### Emotion Accuracy by Class

![Emotion Accuracy](visualizations/emotion_accuracy.png)

This plot shows the precision, recall, and F1-score for each emotion class with sufficient support. The model performs best on Excited and Frustration, with moderate performance on Anger and Sadness.

### VAD Distribution by Emotion

#### Valence by Emotion

![Valence by Emotion](results/run_20250417_162116/plots/valence_by_emotion.png)

#### Arousal by Emotion

![Arousal by Emotion](results/run_20250417_162116/plots/arousal_by_emotion.png)

#### Dominance by Emotion

![Dominance by Emotion](results/run_20250417_162116/plots/dominance_by_emotion.png)

These plots show the distribution of VAD values for each emotion class. We can observe that:

- **Valence**: Happiness has higher valence values, while Sadness and Anger have lower values.
- **Arousal**: Anger and Excited have higher arousal values, while Sadness has lower values.
- **Dominance**: Anger has higher dominance values, while Sadness has lower values.

### VAD 3D Visualization

![VAD 3D by Emotion](results/run_20250417_162116/plots/vad_3d_by_emotion.png)

This 3D plot shows the distribution of emotions in the VAD space. We can observe clusters of emotions, with some overlap between similar emotions.

### t-SNE Visualization of VAD Values

![t-SNE VAD](results/run_20250417_162116/plots/tsne_vad.png)

The t-SNE visualization reduces the 3D VAD space to 2D, showing the clustering of emotions. This visualization helps to understand the separability of emotions in the VAD space.

## Console Output

```
2025-04-17 16:21:16,132 - src.main - INFO - Loading and preprocessing data
2025-04-17 16:21:16,132 - src.data.data_loader - INFO - Loading IEMOCAP data from IEMOCAP_Final.csv
2025-04-17 16:21:16,280 - src.data.data_loader - INFO - Data split: train=7025, val=1004, test=2008
2025-04-17 16:21:16,280 - src.main - INFO - Scaling VAD values
2025-04-17 16:21:16,281 - src.data.data_loader - INFO - Scaler saved to results/run_20250417_162116/vad_scaler.pkl
2025-04-17 16:21:16,282 - src.main - INFO - Creating dataloaders
2025-04-17 16:21:16,283 - src.main - INFO - Initializing VAD predictor with facebook/bart-large-mnli
2025-04-17 16:21:16,283 - src.models.vad_predictor - INFO - Initializing BARTZeroShotVADPredictor with facebook/bart-large-mnli on cpu
2025-04-17 16:21:47,965 - src.main - INFO - Preparing data for emotion classifier
2025-04-17 16:21:47,968 - src.main - INFO - Training emotion classifier
2025-04-17 16:21:47,968 - src.models.emotion_classifier - INFO - Training VADEmotionClassifier with 100 trees
2025-04-17 16:21:48,073 - src.models.emotion_classifier - INFO - Feature importances: valence=0.6191, arousal=0.1860, dominance=0.1949
2025-04-17 16:21:48,073 - src.main - INFO - Evaluating emotion classifier on validation set
```

## Analysis and Observations

1. **Class Imbalance**: The dataset has significant class imbalance, with Frustration and Excited being the most frequent emotions, while others like Fear and various "Other" categories have very few samples. This affects the model's ability to learn these less frequent emotions.

2. **Feature Importance**: Valence is by far the most important feature for emotion classification, which aligns with the intuition that the positive/negative sentiment of an utterance is a strong indicator of the emotion.

3. **Performance by Emotion**: The model performs best on the most frequent emotions (Excited and Frustration), with F1-scores above 0.64. Performance on Anger is moderate (F1-score of 0.50), while performance on Happiness, Sadness, and Neutral state is lower. The model fails to predict Fear and various "Other" categories.

4. **Zero-shot Approach**: The zero-shot approach using BART for VAD prediction provides a reasonable baseline without requiring fine-tuning. However, the performance could potentially be improved with fine-tuning on the IEMOCAP dataset.

## Next Steps

1. **Fine-tuning**: Implement fine-tuning of the pre-trained language models on the IEMOCAP dataset to improve VAD prediction.

2. **Multimodal Integration**: Complete the multimodal implementation to leverage both text and audio modalities for improved emotion recognition.

3. **Addressing Class Imbalance**: Explore techniques to address class imbalance, such as oversampling, undersampling, or using class weights.

4. **Hyperparameter Tuning**: Experiment with different hyperparameters for the Random Forest classifier and other components.

5. **Model Comparison**: Compare different pre-trained models and fusion strategies to find the optimal approach.
