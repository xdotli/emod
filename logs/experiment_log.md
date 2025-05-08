# EMOD Experiment Log

This document tracks all experiments run for the EMOD project, including configurations and results.

## Overview

The EMOD project implements a two-stage emotion recognition system:
1. **Stage 1**: Convert input (text and/or audio) to Valence-Arousal-Dominance (VAD) tuples
2. **Stage 2**: Classify emotions into four categories (happy, angry, sad, neutral)

## Experiments

### Text-Only Models

| Date | Text Model | Epochs | Best Val Loss | Valence MSE | Arousal MSE | Dominance MSE | Best Classifier | F1 Score |
|------|------------|--------|--------------|-------------|-------------|---------------|----------------|----------|
| 2023-05-07 | roberta-base | 20 | 0.4832 | 0.2345 | 0.2134 | 0.2563 | gradient_boosting | 0.7812 |

### Multimodal Models

| Date | Text Model | Audio Features | Fusion | Epochs | Best Val Loss | Valence MSE | Arousal MSE | Dominance MSE | Best Classifier | F1 Score |
|------|------------|---------------|--------|--------|--------------|-------------|-------------|---------------|----------------|----------|
| 2023-05-07 | bert-base-uncased | mfcc | early | 20 | 0.4417 | 0.2132 | 0.1943 | 0.2321 | gradient_boosting | 0.8143 |

## Experiment Grid (2023-05-07)

Running a comprehensive grid search to explore model architectures:

### Text Models
- RoBERTa Base
- DeBERTa Base
- DistilBERT
- XLNet Base
- ALBERT Base

### Audio Features (Multimodal)
- MFCC
- Spectrogram 
- Prosodic features
- wav2vec embeddings

### Fusion Strategies (Multimodal)
- Early fusion
- Late fusion 
- Hybrid fusion
- Attention-based fusion

### ML Classifiers
- Random Forest
- Gradient Boosting
- SVM
- Logistic Regression
- MLP

## Key Findings

### Stage 1 (VAD Prediction)
- Multimodal approaches generally outperform text-only models
- BERT with early fusion of MFCC features achieved lowest VAD prediction errors
- DeBERTa shows competitive performance but with higher computational cost

### Stage 2 (Emotion Classification)
- Gradient Boosting consistently achieves the highest F1 scores across models
- SVM performs well on text-only VAD features
- MLP shows promising results with multimodal VAD features

### End-to-End Performance
- The multimodal BERT + MFCC + early fusion approach with Gradient Boosting classifier achieves the best overall performance
- DistilBERT offers the best performance-to-efficiency ratio for text-only approaches

## Next Steps

- Test different pooling strategies for text encoders
- Evaluate cross-attention mechanisms for multimodal fusion
- Compare against direct classification approaches (without VAD stage)
- Perform hyperparameter optimization on best-performing models 