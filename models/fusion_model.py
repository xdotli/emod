#!/usr/bin/env python3
"""
Fusion VAD Model - Improved architecture for multimodal fusion

This module contains an improved model for fusing audio and text modalities
to predict VAD (valence-arousal-dominance) values with better performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FusionVADModel(nn.Module):
    """
    Enhanced model to fuse audio and text features for VAD prediction.
    
    Architecture improvements:
    1. Better cross-modal attention mechanism
    2. Gating mechanisms to control information flow
    3. Separate paths for each VAD component
    4. Regularization and normalization improvements
    """
    def __init__(self, audio_dim=3, text_dim=3, hidden_dim=128):
        super(FusionVADModel, self).__init__()
        
        # Project each modality to higher dimension for better representation
        self.audio_projection = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # Cross-modal attention - each modality attends to the other
        self.audio_to_text_attention = CrossModalAttention(hidden_dim)
        self.text_to_audio_attention = CrossModalAttention(hidden_dim)
        
        # Gating mechanism for dynamic fusion
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, 2),  # 4 = audio, text, audio_attended, text_attended
            nn.Softmax(dim=1)
        )
        
        # Integration network
        self.integration = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Separate regression heads for each VAD dimension
        self.valence_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Tanh()  # Scale to [-1, 1]
        )
        
        self.arousal_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Tanh()  # Scale to [-1, 1]
        )
        
        self.dominance_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Tanh()  # Scale to [-1, 1]
        )
        
        # Initialize weights for better convergence
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training"""
        for module in [self.audio_projection, self.text_projection, 
                      self.integration, self.valence_head, 
                      self.arousal_head, self.dominance_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(self, audio_vad, text_vad):
        """
        Forward pass through the model.
        
        Args:
            audio_vad: Audio-based VAD predictions [batch_size, 3]
            text_vad: Text-based VAD predictions [batch_size, 3]
            
        Returns:
            fused_vad: Fused VAD values [batch_size, 3]
            weights: Attention weights for audio and text
        """
        # Project to higher dimension
        audio_features = self.audio_projection(audio_vad)
        text_features = self.text_projection(text_vad)
        
        # Apply cross-modal attention
        audio_attended_text, audio_attn_weights = self.audio_to_text_attention(
            audio_features, text_features
        )
        text_attended_audio, text_attn_weights = self.text_to_audio_attention(
            text_features, audio_features
        )
        
        # Calculate gating weights based on all features
        gate_input = torch.cat([
            audio_features, text_features, 
            audio_attended_text, text_attended_audio
        ], dim=1)
        gates = self.fusion_gate(gate_input)
        
        # Apply dynamic weighting
        audio_weight = gates[:, 0].unsqueeze(1)
        text_weight = gates[:, 1].unsqueeze(1)
        
        weighted_audio = (audio_features + audio_attended_text) * audio_weight
        weighted_text = (text_features + text_attended_audio) * text_weight
        
        # Combine modalities
        combined = torch.cat([weighted_audio, weighted_text], dim=1)
        integrated = self.integration(combined)
        
        # Get VAD predictions from specialized heads
        valence = self.valence_head(integrated)
        arousal = self.arousal_head(integrated)
        dominance = self.dominance_head(integrated)
        
        # Combine predictions
        fused_vad = torch.cat([valence, arousal, dominance], dim=1)
        
        return fused_vad, gates


class CrossModalAttention(nn.Module):
    """
    Improved cross-modal attention mechanism.
    
    This implements a multi-head attention where one modality attends
    to the other, using scaled dot-product attention.
    """
    def __init__(self, dim, num_heads=4):
        super(CrossModalAttention, self).__init__()
        assert dim % num_heads == 0, "Dimension must be divisible by number of heads"
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5  # Scaling factor
        
        # Linear projections for query, key, value
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        
        # Output projection
        self.output_proj = nn.Linear(dim, dim)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, query_input, key_value_input):
        """
        Forward pass for cross-modal attention.
        
        Args:
            query_input: Features that will attend to the other modality [batch_size, dim]
            key_value_input: Features to be attended to [batch_size, dim]
            
        Returns:
            output: Attended features [batch_size, dim]
            attention_weights: Attention weights [batch_size, num_heads]
        """
        batch_size = query_input.size(0)
        
        # Apply layer normalization
        query = self.norm1(query_input)
        key_value = self.norm2(key_value_input)
        
        # Linear projections and reshape to [batch_size, num_heads, 1, head_dim]
        # Note: We're doing single-token attention, so the sequence dimension is 1
        query = self.query_proj(query).view(batch_size, self.num_heads, 1, self.head_dim)
        key = self.key_proj(key_value).view(batch_size, self.num_heads, 1, self.head_dim)
        value = self.value_proj(key_value).view(batch_size, self.num_heads, 1, self.head_dim)
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, value)
        
        # Reshape and apply output projection
        context = context.view(batch_size, -1)  # [batch_size, dim]
        output = self.output_proj(context)
        
        # Residual connection
        output = output + query_input
        
        # Return attended features and attention weights
        return output, attention_weights.squeeze(-2).mean(dim=1)  # Average attention weights across heads 