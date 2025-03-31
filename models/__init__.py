"""
Models package for IEMOCAP Emotion Recognition

This package contains the models used for the IEMOCAP emotion recognition system.
"""

from models.audio_model import AudioVADModel, get_audio_feature_dim
from models.text_model import TextVADModel, MeanPooling
from models.fusion_model import FusionVADModel, CrossModalAttention

__all__ = [
    'AudioVADModel',
    'TextVADModel', 
    'FusionVADModel',
    'get_audio_feature_dim',
    'MeanPooling',
    'CrossModalAttention'
] 