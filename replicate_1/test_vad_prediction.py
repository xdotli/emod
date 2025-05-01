"""
Test VAD prediction with example texts.
"""
import numpy as np
from models import RuleBasedEmotionClassifier

def main():
    # Initialize the rule-based classifier
    classifier = RuleBasedEmotionClassifier()
    
    # Example texts and their estimated VAD values
    examples = [
        # Format: [text, [valence, arousal, dominance]]
        ["I am feeling very happy today!", [4.5, 4.0, 4.0]],
        ["I am so angry right now!", [2.0, 4.5, 4.5]],
        ["I feel sad and depressed.", [2.0, 2.0, 2.0]],
        ["I'm just feeling normal, nothing special.", [3.0, 3.0, 3.0]],
        ["That was the best day of my life!", [4.8, 4.2, 4.5]],
        ["I hate when people lie to me.", [1.8, 4.3, 4.0]],
        ["I miss my family so much.", [2.2, 2.5, 2.0]],
        ["It's just another ordinary day.", [3.2, 2.8, 3.1]]
    ]
    
    # Predict emotions for each example
    print("Testing VAD prediction and emotion classification with example texts:\n")
    for i, (text, vad) in enumerate(examples):
        emotion = classifier.predict(np.array([vad]))[0]
        print(f"Example {i+1}:")
        print(f"Text: {text}")
        print(f"Estimated VAD values: Valence={vad[0]:.2f}, Arousal={vad[1]:.2f}, Dominance={vad[2]:.2f}")
        print(f"Predicted emotion: {emotion}")
        print("-" * 80)

if __name__ == '__main__':
    main()
