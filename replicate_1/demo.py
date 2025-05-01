"""
Demo script for the emotion recognition system.
"""
import os
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

from models import (
    TextVADModel,
    EmotionClassifier,
    RuleBasedEmotionClassifier
)
from utils import vad_to_emotion, get_emotion_color

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Emotion Recognition Demo')
    parser.add_argument('--text', type=str, required=True,
                        help='Text to analyze')
    parser.add_argument('--model_dir', type=str, default='output',
                        help='Directory containing trained models')
    parser.add_argument('--use_rule_based', action='store_true',
                        help='Use rule-based emotion classification instead of trained classifier')

    return parser.parse_args()

def load_models(model_dir):
    """
    Load trained models.

    Args:
        model_dir: Directory containing trained models

    Returns:
        Loaded models
    """
    # Load text VAD model
    text_vad_model = TextVADModel(model_name='roberta-base')
    text_vad_model.load_state_dict(torch.load(os.path.join(model_dir, 'text_vad_model.pt')))
    text_vad_model.eval()

    # Load emotion classifier if not using rule-based
    emotion_classifier = None
    if os.path.exists(os.path.join(model_dir, 'text_emotion_classifier.pt')):
        emotion_classifier = EmotionClassifier(num_classes=4)
        emotion_classifier.load_state_dict(torch.load(os.path.join(model_dir, 'text_emotion_classifier.pt')))
        emotion_classifier.eval()

    return text_vad_model, emotion_classifier

def predict_emotion(text, text_vad_model, emotion_classifier=None, use_rule_based=False):
    """
    Predict emotion from text.

    Args:
        text: Input text
        text_vad_model: Trained text VAD model
        emotion_classifier: Trained emotion classifier (optional)
        use_rule_based: Whether to use rule-based classification

    Returns:
        Predicted VAD values and emotion
    """
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    # Tokenize text
    encoding = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move model and inputs to device
    text_vad_model = text_vad_model.to(device)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Predict VAD values
    with torch.no_grad():
        vad_pred = text_vad_model(input_ids, attention_mask)

    # Convert to numpy
    vad_values = vad_pred.cpu().numpy()[0]

    # Predict emotion
    if use_rule_based or emotion_classifier is None:
        # Use rule-based classification
        rule_based_classifier = RuleBasedEmotionClassifier()
        emotion = rule_based_classifier.predict(vad_values.reshape(1, -1))[0]
    else:
        # Use trained classifier
        emotion_classifier = emotion_classifier.to(device)
        with torch.no_grad():
            vad_tensor = torch.tensor(vad_values, dtype=torch.float).unsqueeze(0).to(device)
            emotion_logits = emotion_classifier(vad_tensor)
            _, predicted = torch.max(emotion_logits, 1)
            emotion_idx = predicted.item()

            # Map index to emotion name
            emotion_map = {0: 'angry', 1: 'happy', 2: 'neutral', 3: 'sad'}
            emotion = emotion_map.get(emotion_idx, 'unknown')

    return vad_values, emotion

def visualize_results(text, vad_values, emotion):
    """
    Visualize prediction results.

    Args:
        text: Input text
        vad_values: Predicted VAD values
        emotion: Predicted emotion
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot VAD values
    vad_labels = ['Valence', 'Arousal', 'Dominance']
    colors = ['#FF9999', '#99FF99', '#9999FF']

    ax1.bar(vad_labels, vad_values, color=colors)
    ax1.set_ylim(1, 5)
    ax1.set_title('Predicted VAD Values')
    ax1.set_ylabel('Value (1-5)')

    # Add value labels
    for i, v in enumerate(vad_values):
        ax1.text(i, v + 0.1, f'{v:.2f}', ha='center')

    # Plot emotion
    emotion_color = get_emotion_color(emotion)
    ax2.pie([1], labels=[emotion.capitalize()], colors=[emotion_color], autopct='%1.1f%%',
            startangle=90, wedgeprops={'alpha': 0.7})
    ax2.set_title('Predicted Emotion')

    # Add text as figure title
    plt.suptitle(f'Text: "{text}"', fontsize=12)

    plt.tight_layout()
    plt.show()

def main():
    """
    Main function.
    """
    # Parse arguments
    args = parse_args()

    # Load models
    text_vad_model, emotion_classifier = load_models(args.model_dir)

    # Predict emotion
    vad_values, emotion = predict_emotion(
        args.text,
        text_vad_model,
        emotion_classifier,
        use_rule_based=args.use_rule_based
    )

    # Print results
    print(f"Text: {args.text}")
    print(f"VAD values: Valence={vad_values[0]:.2f}, Arousal={vad_values[1]:.2f}, Dominance={vad_values[2]:.2f}")
    print(f"Predicted emotion: {emotion}")

    # Skip visualization for now
    # visualize_results(args.text, vad_values, emotion)

if __name__ == '__main__':
    main()
