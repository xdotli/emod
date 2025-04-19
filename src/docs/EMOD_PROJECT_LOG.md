# EMOD Project Log

This document logs all the activities, implementations, results, and analyses performed during the development of the emotion recognition system using the IEMOCAP dataset.

## Project Overview

The EMOD project implements a two-stage emotion recognition system using audio and text modalities from the IEMOCAP dataset. The approach first converts input to valence-arousal-dominance (VAD) tuples, then categorizes emotions based on these values.

## Initial Implementation

### Project Structure

We implemented a two-stage emotion recognition system with the following structure:

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

### Implementation Details

1. **Stage 1 (Text to VAD)**:
   - Implemented zero-shot prediction of VAD values using pre-trained language models
   - Used BART-large-MNLI for natural language inference to predict VAD values
   - Created a `BARTZeroShotVADPredictor` class in `src/models/vad_predictor.py`

2. **Stage 2 (VAD to Emotion)**:
   - Implemented a Random Forest classifier to predict emotion labels from VAD values
   - Created a `VADEmotionClassifier` class in `src/models/emotion_classifier.py`

3. **End-to-End Pipeline**:
   - Combined both stages into a complete pipeline
   - Created an `EmotionRecognitionPipeline` class in `src/models/pipeline.py`

4. **Utilities**:
   - Implemented data loading and preprocessing functions
   - Created evaluation metrics and visualization tools

## Initial Results (26 Emotion Categories)

### Stage 1 (Text to VAD) Performance
- **MSE**: 1.6072
- **RMSE**: 1.2678
- **MAE**: 1.0041

### Stage 2 (VAD to Emotion) Performance
- **Validation Accuracy**: 58.86%
- **F1 (macro)**: 20.06%
- **F1 (weighted)**: 55.61%

### End-to-End (Text to Emotion) Performance
- **Test Accuracy**: 32.62%
- **F1 (macro)**: 8.34%
- **F1 (weighted)**: 31.86%

### Performance by Emotion
- **Best performing emotions**:
  - Excited: F1 = 0.6910 (validation), F1 = 0.3882 (test)
  - Frustration: F1 = 0.6472 (validation), F1 = 0.4597 (test)
  - Anger: F1 = 0.4973 (validation), F1 = 0.1176 (test)
  
- **Moderate performing emotions**:
  - Sadness: F1 = 0.3846 (validation), F1 = 0.1695 (test)
  - Happiness: F1 = 0.2545 (validation), F1 = 0.1245 (test)
  
- **Poor performing emotions**:
  - Neutral state: F1 = 0.1333 (validation), F1 = 0.0745 (test)
  - Fear: F1 = 0.0000 (validation), F1 = 0.0000 (test)
  - Various "Other" categories: F1 = 0.0000 (validation), F1 = 0.0000 (test)

### Feature Importance
- **Valence**: 61.91%
- **Arousal**: 18.60%
- **Dominance**: 19.49%

### Comparison to Baselines
- **Random Guessing**: 3.85% (1/26 classes)
- **Majority Class**: 38.13%
- **Our End-to-End Accuracy**: 32.62%

## Analysis of Initial Results

1. **Performance Drop from Validation to Test**: There's a significant drop in performance from the validation set to the test set (58.86% to 32.62% accuracy). This suggests that the model might be overfitting to the validation set or that there's a distribution shift between the validation and test sets.

2. **Class Imbalance**: The model performs better on more frequent emotions (Excited, Frustration) and poorly on less frequent ones (Fear, various "Other" categories). This is expected due to class imbalance in the dataset.

3. **Zero-shot Approach Limitations**: The zero-shot approach using pre-trained language models for VAD prediction has limitations, as evidenced by the relatively high MSE (1.6072) for VAD prediction.

4. **Comparison to Baselines**: The end-to-end accuracy (32.62%) is significantly better than random guessing (3.85%) but worse than always predicting the most common class (38.13%).

## Fine-Tuning Implementation

We implemented fine-tuning for the VAD prediction model to improve Stage 1 performance:

1. **VAD Regressor Model**:
   - Created a `VADRegressor` class that adds a regression head to a pre-trained language model
   - Implemented in `src/models/vad_fine_tuner.py`

2. **Fine-Tuning Process**:
   - Created a `VADFineTuner` class to handle the fine-tuning process
   - Implemented training, evaluation, and prediction functions
   - Added logging and visualization of training progress

3. **Integration with Pipeline**:
   - Created a script to use the fine-tuned model in the pipeline
   - Implemented in `run_with_fine_tuned.py`

However, the fine-tuning process encountered issues and didn't complete successfully.

## Reduced Categories Implementation

To address the class imbalance issue, we reduced the number of emotion categories from 26 to 4 (Angry, Happy, Neutral, Sad):

1. **Data Preprocessing**:
   - Created a script to map the original 26 emotion categories to 4 categories
   - Implemented in `preprocess_reduced_categories.py`

2. **Mapping Logic**:
   - Angry: Anger, Frustration, Disgust, etc.
   - Happy: Happiness, Excited, etc.
   - Neutral: Neutral state, Surprise, etc.
   - Sad: Sadness, Fear, etc.

3. **Pipeline with Reduced Categories**:
   - Created a script to run the emotion recognition pipeline with reduced categories
   - Implemented in `run_reduced_categories.py`

4. **Evaluation**:
   - Created a script to evaluate the performance of the reduced categories approach
   - Implemented in `evaluate_reduced_categories.py`

## Results with Reduced Categories (4 Emotion Categories)

### Stage 1 (Text to VAD) Performance
- **MSE**: 1.6072
- **RMSE**: 1.2678
- **MAE**: 1.0041

### Stage 2 (VAD to Emotion) Performance
- **Validation Accuracy**: 71.51%
- **F1 (macro)**: 50.84%
- **F1 (weighted)**: 68.70%

### End-to-End (Text to Emotion) Performance
- **Test Accuracy**: 47.06%
- **F1 (macro)**: 32.34%
- **F1 (weighted)**: 46.73%

### Performance by Emotion (End-to-End)
- **Angry**: F1 = 0.5798, Precision = 0.5615, Recall = 0.5992, Support = 1013
- **Happy**: F1 = 0.4834, Precision = 0.5108, Recall = 0.4588, Support = 619
- **Neutral**: F1 = 0.0654, Precision = 0.1000, Recall = 0.0485, Support = 103
- **Sad**: F1 = 0.1650, Precision = 0.1526, Recall = 0.1795, Support = 273

### Feature Importance
- **Valence**: 72.71%
- **Arousal**: 12.48%
- **Dominance**: 14.82%

### Comparison to Baselines
- **Random Guessing**: 25.00%
- **Majority Class**: 50.49%
- **Our End-to-End Accuracy**: 47.06%

### Comparison to Original Model (26 Categories)
- **Stage 2 Accuracy**: Improved from 58.86% to 71.51% (+12.65%)
- **End-to-End Accuracy**: Improved from 32.62% to 47.06% (+14.44%)
- **F1 (weighted)**: Improved from 31.86% to 46.73% (+14.87%)

## Analysis of Reduced Categories Results

1. **Significant Improvement**: Reducing the number of emotion categories from 26 to 4 has dramatically improved both Stage 2 and end-to-end accuracy. The end-to-end accuracy increased from 32.62% to 47.06%, which is a 14.44 percentage point improvement.

2. **Performance by Emotion**: The model performs best on the Angry and Happy emotions, with F1-scores of 0.5798 and 0.4834 respectively. It struggles with Neutral and Sad emotions, with F1-scores of 0.0654 and 0.1650 respectively. This is likely due to class imbalance, as Angry and Happy are the majority classes.

3. **Stage 1 vs. Stage 2 Performance**: There's still a significant gap between Stage 2 accuracy (71.51%) and end-to-end accuracy (47.06%), indicating that errors in Stage 1 (VAD prediction) are propagating to Stage 2 (emotion classification).

4. **Comparison to Baselines**: The end-to-end accuracy (47.06%) is significantly better than random guessing (25.00%) but slightly worse than always predicting the majority class (50.49%). This suggests that while the model is learning, there's still room for improvement.

5. **Feature Importance**: Valence (positive/negative sentiment) remains the most important feature, with an even higher importance (72.71%) than in the original model (61.91%).

## Multimodal Implementation (Audio + Text)

We also implemented a multimodal approach that combines both text and audio modalities:

1. **Audio Processing**:
   - Created an `AudioFeatureExtractor` class to extract features from audio files
   - Implemented MFCC, spectral, and Wav2Vec2 feature extraction methods
   - Created an `AudioVADPredictor` class to predict VAD values from audio features

2. **Multimodal Fusion**:
   - Implemented early and late fusion strategies
   - Created `EarlyFusion` and `LateFusion` classes

3. **Multimodal Pipeline**:
   - Created a `MultimodalEmotionRecognitionPipeline` class
   - Implemented evaluation and prediction functions

However, we didn't complete the evaluation of the multimodal approach due to time constraints.

## Conclusion and Future Work

### Key Findings

1. **Two-Stage Approach**: The two-stage approach (Text → VAD → Emotion) provides a structured way to tackle emotion recognition, but errors in Stage 1 propagate to Stage 2.

2. **Category Reduction**: Reducing the number of emotion categories from 26 to 4 significantly improves performance, with end-to-end accuracy increasing from 32.62% to 47.06%.

3. **Feature Importance**: Valence (positive/negative sentiment) is the most important feature for emotion classification, with an importance of 72.71% in the reduced categories model.

4. **Class Imbalance**: The model performs better on majority classes (Angry, Happy) and poorly on minority classes (Neutral, Sad), indicating that class imbalance remains an issue.

### Future Work

1. **Fine-Tuning**: Complete the fine-tuning of pre-trained language models for VAD prediction to improve Stage 1 performance.

2. **Class Imbalance**: Implement techniques like oversampling, undersampling, or using class weights to address class imbalance.

3. **Multimodal Integration**: Complete the evaluation of the multimodal approach to see if combining text and audio modalities improves performance.

4. **Model Comparison**: Compare different pre-trained models and fusion strategies to find the optimal approach.

5. **End-to-End Training**: Explore end-to-end training approaches that directly predict emotions from text, bypassing the VAD intermediate step.

## Summary of Implementations

1. **Initial Implementation (26 Categories)**:
   - Two-stage approach: Text → VAD → Emotion
   - Zero-shot prediction of VAD values using BART-large-MNLI
   - Random Forest classifier for emotion prediction
   - End-to-End Accuracy: 32.62%

2. **Reduced Categories Implementation (4 Categories)**:
   - Same two-stage approach
   - Mapped 26 categories to 4 (Angry, Happy, Neutral, Sad)
   - End-to-End Accuracy: 47.06%

3. **Fine-Tuning Implementation**:
   - Added a regression head to pre-trained language models
   - Implemented training and evaluation functions
   - Not completed due to technical issues

4. **Multimodal Implementation**:
   - Added audio processing and feature extraction
   - Implemented early and late fusion strategies
   - Not evaluated due to time constraints
