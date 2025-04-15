"""
State-of-the-art models for emotion detection.
"""

from sota_models.transcription import TranscriptionModel
from sota_models.emotion_detection import (
    TextEmotionDetector,
    AudioEmotionDetector,
    MultimodalEmotionDetector
)
from sota_models.integration import SotaEmotionAnalyzer

__all__ = [
    'TranscriptionModel',
    'TextEmotionDetector',
    'AudioEmotionDetector',
    'MultimodalEmotionDetector',
    'SotaEmotionAnalyzer'
]
