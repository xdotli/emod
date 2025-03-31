#!/usr/bin/env python3
"""
Text VAD Model - Improved architecture for text to VAD conversion

This module contains an improved model for converting text input to
VAD (valence-arousal-dominance) values with better performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, AutoModel

class TextVADModel(nn.Module):
    """
    Enhanced model to convert text embeddings to VAD values.
    
    Architecture improvements:
    1. Better language model (RoBERTa base as default)
    2. Better fine-tuning strategy with layer-wise learning rates
    3. More sophisticated output network with separate VAD heads
    4. Better regularization
    """
    def __init__(self, model_name="roberta-base", finetune=True):
        super(TextVADModel, self).__init__()
        
        # Use robust language model that generalizes well
        if "roberta" in model_name:
            self.encoder = RobertaModel.from_pretrained(model_name)
        else:
            self.encoder = AutoModel.from_pretrained(model_name)
        
        hidden_size = self.encoder.config.hidden_size

        # To fine-tune or not to fine-tune
        if not finetune:
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            # Freeze embeddings and early layers which capture general linguistic patterns
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = False
            
            # Gradually unfreeze layers (for RoBERTa with 12 layers)
            # First layers capture more general features, later layers more task-specific
            try:
                num_layers = len(self.encoder.encoder.layer)
                for i, layer in enumerate(self.encoder.encoder.layer):
                    # Only fine-tune the last few layers
                    if i < num_layers - 4:  # Fine-tune just the last 4 layers
                        for param in layer.parameters():
                            param.requires_grad = False
            except AttributeError:
                # Different architecture, fallback approach
                print("Warning: Layer freezing not applied for this model architecture")
        
        # Pooling layer for sentence representation
        self.pooler = MeanPooling()
        
        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        # Separate heads for each VAD dimension for better specialization
        self.valence_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(), 
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Tanh()  # Scale to [-1, 1]
        )
        
        self.arousal_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Tanh()  # Scale to [-1, 1]
        )
        
        self.dominance_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Tanh()  # Scale to [-1, 1]
        )
        
        # Initialize weights for better convergence
        self._init_weights()
        
    def _init_weights(self):
        """Initialize output head weights for better training"""
        for module in [self.trunk, self.valence_head, self.arousal_head, self.dominance_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            vad_values: Predicted VAD values [batch_size, 3]
        """
        # Encode text with language model
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get sentence representation
        embeddings = self.pooler(outputs.last_hidden_state, attention_mask)
        
        # Process through trunk
        features = self.trunk(embeddings)
        
        # Get VAD predictions from specialized heads
        valence = self.valence_head(features)
        arousal = self.arousal_head(features)
        dominance = self.dominance_head(features)
        
        # Combine predictions
        vad_values = torch.cat([valence, arousal, dominance], dim=1)
        
        return vad_values


class MeanPooling(nn.Module):
    """Mean pooling layer that respects attention mask"""
    def forward(self, token_embeddings, attention_mask):
        # Expand attention_mask to the same dimensions as token_embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum all token embeddings using attention mask
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        
        # Count non-padding tokens in each sequence
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Calculate mean
        mean_embeddings = sum_embeddings / sum_mask
        
        return mean_embeddings 