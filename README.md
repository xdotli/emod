# Emotion Recognition using Two-Step Approach

This project implements a multimodal emotion recognition system using the IEMOCAP dataset. The approach uses a two-step process:

1. Convert audio and text modalities to VAD (valence-arousal-dominance) tuples
2. Map VAD tuples to emotion categories

## Project Structure

- `main.py`: Main script that implements the full pipeline
- `process_vad.py`: Contains the rule-based VAD-to-emotion mapping function
- `vad_to_emotion_model.py`: Alternative ML approach for VAD-to-emotion mapping
- `test_vad_model.py`: Script to analyze and visualize the VAD-to-emotion mapping
- `model_tracker.txt`: Keeps track of latest model paths and performance metrics

## Models

The system uses three neural networks:

1. **AudioVADModel**: Converts audio features to VAD values
2. **TextVADModel**: Converts text (using BERT) to VAD values
3. **FusionVADModel**: Uses attention mechanism to combine audio and text VAD predictions

## Performance

Current performance metrics:

- Text to VAD: MSE = 0.1845
- Audio to VAD: MSE = 0.2850
- Fusion to VAD: MSE = 0.1905
- VAD to Emotion: Accuracy = 0.0612 (6.12%)
- End-to-End Accuracy: 0.0612 (6.12%)

The VAD-to-emotion mapping is currently the bottleneck of the system.

## How to Run

```bash
# Train and evaluate the model
python main.py

# Test the VAD-to-emotion mapping
python test_vad_model.py

# Train a machine learning model for VAD-to-emotion mapping
python vad_to_emotion_model.py
```

## Improvement Directions

1. Replace the rule-based VAD-to-emotion mapping with the machine learning approach in `vad_to_emotion_model.py`
2. Improve the fusion mechanism with more sophisticated techniques
3. Fine-tune the audio and text encoders on the IEMOCAP dataset
4. Explore different VAD space divisions for emotion mapping
