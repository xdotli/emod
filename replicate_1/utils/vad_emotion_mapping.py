"""
Utility functions for mapping between VAD values and emotions.
"""
import numpy as np

def vad_to_emotion(valence, arousal, dominance):
    """
    Map VAD values to emotion categories using rule-based approach.
    
    Args:
        valence: Valence value (1-5)
        arousal: Arousal value (1-5)
        dominance: Dominance value (1-5)
        
    Returns:
        Emotion category (happy, sad, angry, neutral)
    """
    # Define VAD thresholds for each emotion
    # Format: (valence_min, valence_max, arousal_min, arousal_max, dominance_min, dominance_max)
    emotion_thresholds = {
        'happy': (3.5, 5.0, 3.0, 5.0, 3.0, 5.0),  # High valence, high arousal, high dominance
        'angry': (1.0, 2.5, 3.5, 5.0, 3.5, 5.0),  # Low valence, high arousal, high dominance
        'sad': (1.0, 2.5, 1.0, 2.5, 1.0, 2.5),    # Low valence, low arousal, low dominance
        'neutral': (2.5, 3.5, 2.5, 3.5, 2.5, 3.5)  # Medium valence, medium arousal, medium dominance
    }
    
    # Calculate distance to each emotion prototype
    distances = {}
    for emotion, thresholds in emotion_thresholds.items():
        v_min, v_max, a_min, a_max, d_min, d_max = thresholds
        
        # Calculate distance to center of threshold range
        v_center = (v_min + v_max) / 2
        a_center = (a_min + a_max) / 2
        d_center = (d_min + d_max) / 2
        
        distance = np.sqrt((valence - v_center)**2 + (arousal - a_center)**2 + (dominance - d_center)**2)
        distances[emotion] = distance
    
    # Return the emotion with the smallest distance
    return min(distances, key=distances.get)

def batch_vad_to_emotion(vad_values):
    """
    Map batch of VAD values to emotion categories.
    
    Args:
        vad_values: Array of VAD values (valence, arousal, dominance)
        
    Returns:
        Array of emotion categories
    """
    emotions = []
    for vad in vad_values:
        valence, arousal, dominance = vad
        emotion = vad_to_emotion(valence, arousal, dominance)
        emotions.append(emotion)
    
    return np.array(emotions)

def emotion_to_vad(emotion):
    """
    Map emotion category to typical VAD values.
    
    Args:
        emotion: Emotion category (happy, sad, angry, neutral)
        
    Returns:
        Tuple of (valence, arousal, dominance)
    """
    # Define typical VAD values for each emotion
    emotion_vad = {
        'happy': (4.0, 4.0, 4.0),    # High valence, high arousal, high dominance
        'angry': (2.0, 4.0, 4.0),    # Low valence, high arousal, high dominance
        'sad': (2.0, 2.0, 2.0),      # Low valence, low arousal, low dominance
        'neutral': (3.0, 3.0, 3.0)   # Medium valence, medium arousal, medium dominance
    }
    
    return emotion_vad.get(emotion.lower(), (3.0, 3.0, 3.0))  # Default to neutral

def get_emotion_color(emotion):
    """
    Get color for visualization based on emotion.
    
    Args:
        emotion: Emotion category
        
    Returns:
        RGB color tuple
    """
    emotion_colors = {
        'happy': (0.0, 1.0, 0.0),    # Green
        'angry': (1.0, 0.0, 0.0),    # Red
        'sad': (0.0, 0.0, 1.0),      # Blue
        'neutral': (0.5, 0.5, 0.5)   # Gray
    }
    
    return emotion_colors.get(emotion.lower(), (0.5, 0.5, 0.5))  # Default to gray
