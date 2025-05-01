"""
Test the rule-based emotion classifier.
"""
import numpy as np
from models import RuleBasedEmotionClassifier

def main():
    # Initialize the rule-based classifier
    classifier = RuleBasedEmotionClassifier()
    
    # Example VAD values
    examples = [
        # Format: [valence, arousal, dominance]
        [4.5, 4.0, 4.0],  # High valence, high arousal, high dominance (happy)
        [2.0, 4.5, 4.5],  # Low valence, high arousal, high dominance (angry)
        [2.0, 2.0, 2.0],  # Low valence, low arousal, low dominance (sad)
        [3.0, 3.0, 3.0],  # Medium valence, medium arousal, medium dominance (neutral)
    ]
    
    # Predict emotions for each example
    print("Testing rule-based emotion classifier with example VAD values:\n")
    for i, vad in enumerate(examples):
        emotion = classifier.predict(np.array([vad]))[0]
        print(f"Example {i+1}:")
        print(f"VAD values: Valence={vad[0]:.2f}, Arousal={vad[1]:.2f}, Dominance={vad[2]:.2f}")
        print(f"Predicted emotion: {emotion}")
        print("-" * 80)

if __name__ == '__main__':
    main()
