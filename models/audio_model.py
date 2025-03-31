#!/usr/bin/env python3
"""
Audio VAD Model - Improved architecture for audio to VAD conversion

This module contains an improved model for converting audio features to
VAD (valence-arousal-dominance) values with better performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model
from pathlib import Path

AUDIO_FEATURE_DIM = 68  # Default dimension for traditional acoustic features

class AudioVADModel(nn.Module):
    """
    Enhanced model to convert audio features to VAD values.
    
    Architecture improvements:
    1. Pre-normalization of input
    2. Deeper network with skip connections
    3. Proper weight initialization
    4. Layer normalization for internal activations
    """
    def __init__(self, input_dim=AUDIO_FEATURE_DIM, use_wav2vec=False):
        super(AudioVADModel, self).__init__()
        self.use_wav2vec = use_wav2vec
        
        if use_wav2vec:
            try:
                # Use Wav2Vec 2.0 for audio encoding
                self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
                input_dim = 768  # Output dimension from Wav2Vec 2.0
                
                # Freeze early layers to prevent overfitting
                for param in self.wav2vec.feature_extractor.parameters():
                    param.requires_grad = False
                    
                # Only fine-tune the last few layers
                for i, layer in enumerate(self.wav2vec.encoder.layers):
                    if i < 8:  # Freeze the first 8 layers (out of 12)
                        for param in layer.parameters():
                            param.requires_grad = False
            except Exception as e:
                print(f"Warning: Could not load Wav2Vec2 model: {e}")
                print("Falling back to traditional audio features")
                self.use_wav2vec = False
                input_dim = AUDIO_FEATURE_DIM
                self.wav2vec = None
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_dim)
        
        # Encoder blocks
        self.encoder = nn.Sequential(
            # Block 1
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            
            # Block 2
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(256) for _ in range(3)
        ])
        
        # Valence head
        self.valence_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Tanh()  # Scale to [-1, 1]
        )
        
        # Arousal head
        self.arousal_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Tanh()  # Scale to [-1, 1]
        )
        
        # Dominance head
        self.dominance_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Tanh()  # Scale to [-1, 1]
        )
        
        # Initialize weights with careful scaling
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for better convergence"""
        if isinstance(module, nn.Linear):
            # Use Kaiming initialization for linear layers
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input features [batch_size, feature_dim] or 
               waveform [batch_size, time] if using Wav2Vec
               
        Returns:
            vad_values: Predicted VAD values [batch_size, 3]
        """
        if self.use_wav2vec and hasattr(self, 'wav2vec') and self.wav2vec is not None:
            try:
                # Process raw audio with Wav2Vec
                outputs = self.wav2vec(x)
                x = outputs.last_hidden_state.mean(dim=1)  # Average over time dimension
            except Exception as e:
                print(f"Error processing audio with Wav2Vec: {e}")
                # If there's an error with Wav2Vec, just use the input as is
                # This allows graceful fallback to spectral features
        
        # Normalize input
        x = self.input_norm(x)
        
        # Encode features
        x = self.encoder(x)
        
        # Apply residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Get individual VAD predictions from specialized heads
        valence = self.valence_head(x)
        arousal = self.arousal_head(x)
        dominance = self.dominance_head(x)
        
        # Combine predictions
        vad_values = torch.cat([valence, arousal, dominance], dim=1)
        
        return vad_values


class ResidualBlock(nn.Module):
    """Residual block with pre-activation design"""
    def __init__(self, dim, hidden_dim=None):
        super(ResidualBlock, self).__init__()
        if hidden_dim is None:
            hidden_dim = dim
            
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        identity = x
        
        # Pre-activation residual path
        out = self.norm1(x)
        out = F.gelu(out)
        out = self.linear1(out)
        
        out = self.norm2(out)
        out = F.gelu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        
        # Add residual connection
        out = out + identity
        
        return out


# Function to get audio feature dimension
def get_audio_feature_dim():
    return AUDIO_FEATURE_DIM 