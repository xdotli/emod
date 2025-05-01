"""
Test VAD to emotion mapping.
"""
from utils.vad_emotion_mapping import vad_to_emotion

def main():
    # Example VAD values
    examples = [
        # Format: [valence, arousal, dominance]
        [4.5, 4.0, 4.0],  # High valence, high arousal, high dominance (happy)
        [2.0, 4.5, 4.5],  # Low valence, high arousal, high dominance (angry)
        [2.0, 2.0, 2.0],  # Low valence, low arousal, low dominance (sad)
        [3.0, 3.0, 3.0],  # Medium valence, medium arousal, medium dominance (neutral)
        [4.8, 4.2, 4.5],  # Very high valence, high arousal, high dominance (happy)
        [1.8, 4.3, 4.0],  # Very low valence, high arousal, high dominance (angry)
        [2.2, 2.5, 2.0],  # Low valence, medium-low arousal, low dominance (sad)
        [3.2, 2.8, 3.1]   # Medium valence, medium arousal, medium dominance (neutral)
    ]
    
    # Map VAD values to emotions
    print("Testing VAD to emotion mapping:\n")
    for i, vad in enumerate(examples):
        valence, arousal, dominance = vad
        emotion = vad_to_emotion(valence, arousal, dominance)
        print(f"Example {i+1}:")
        print(f"VAD values: Valence={valence:.2f}, Arousal={arousal:.2f}, Dominance={dominance:.2f}")
        print(f"Mapped emotion: {emotion}")
        print("-" * 80)

if __name__ == '__main__':
    main()
