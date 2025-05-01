"""
VAD Predictor Models

This module contains classes and functions for predicting VAD (Valence-Arousal-Dominance)
values from text and audio inputs.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import numpy as np

# Constants
MAX_SEQ_LENGTH = 128
MODEL_NAME = "roberta-base"

class TextVADDataset(Dataset):
    """Dataset for text to VAD prediction."""
    def __init__(self, texts, vad_values, tokenizer, max_len=MAX_SEQ_LENGTH):
        self.texts = texts
        self.vad_values = vad_values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        vad = self.vad_values[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'vad_values': torch.tensor(vad, dtype=torch.float)
        }

class MultimodalVADDataset(Dataset):
    """Dataset for multimodal (text + audio) VAD prediction."""
    def __init__(self, texts, audio_features, vad_values, tokenizer, max_len=MAX_SEQ_LENGTH):
        self.texts = texts
        self.audio_features = audio_features
        self.vad_values = vad_values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        audio_feat = self.audio_features[idx]
        vad = self.vad_values[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'audio_features': torch.tensor(audio_feat, dtype=torch.float),
            'vad_values': torch.tensor(vad, dtype=torch.float)
        }

class TextVADModel(nn.Module):
    """Model for predicting VAD values from text."""
    def __init__(self, model_name=MODEL_NAME, dropout=0.1):
        super(TextVADModel, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = self.transformer.config.hidden_size

        self.shared_layer = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.Dropout(dropout),
            nn.GELU(),
        )

        self.valence_branch = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        self.arousal_branch = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        self.dominance_branch = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        shared = self.shared_layer(pooled)
        valence = self.valence_branch(shared)
        arousal = self.arousal_branch(shared)
        dominance = self.dominance_branch(shared)
        return torch.cat([valence, arousal, dominance], dim=1)

class MultimodalVADModel(nn.Module):
    """Model for predicting VAD values from text and audio."""
    def __init__(self, model_name=MODEL_NAME, audio_feat_dim=88, dropout=0.1, fusion_type='early'):
        super(MultimodalVADModel, self).__init__()
        self.fusion_type = fusion_type
        
        # Text processing
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = self.transformer.config.hidden_size
        
        # Audio processing
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_feat_dim, 256),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.Dropout(dropout),
            nn.GELU(),
        )
        
        # Early fusion architecture
        if fusion_type == 'early':
            # Combine text and audio at feature level
            self.text_projector = nn.Linear(hidden_size, 512)
            self.fusion = nn.Sequential(
                nn.Linear(1024, 512),  # 512 (text) + 512 (audio)
                nn.LayerNorm(512),
                nn.Dropout(dropout),
                nn.GELU(),
            )
            
            self.valence_branch = nn.Sequential(
                nn.Linear(512, 128),
                nn.LayerNorm(128),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(128, 1)
            )
            self.arousal_branch = nn.Sequential(
                nn.Linear(512, 128),
                nn.LayerNorm(128),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(128, 1)
            )
            self.dominance_branch = nn.Sequential(
                nn.Linear(512, 128),
                nn.LayerNorm(128),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(128, 1)
            )
        
        # Late fusion architecture
        else:  # fusion_type == 'late'
            # Process text and audio separately, then combine predictions
            self.text_valence = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.LayerNorm(256),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(256, 1)
            )
            self.text_arousal = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.LayerNorm(256),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(256, 1)
            )
            self.text_dominance = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.LayerNorm(256),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(256, 1)
            )
            
            self.audio_valence = nn.Sequential(
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(256, 1)
            )
            self.audio_arousal = nn.Sequential(
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(256, 1)
            )
            self.audio_dominance = nn.Sequential(
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(256, 1)
            )
            
            # Fusion weights (learnable parameters)
            self.fusion_weights = nn.Parameter(torch.ones(3, 2))  # 3 VAD dimensions, 2 modalities
            self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, audio_features):
        # Process text
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        text_hidden = outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding
        
        # Process audio
        audio_hidden = self.audio_encoder(audio_features)
        
        # Early fusion
        if self.fusion_type == 'early':
            text_proj = self.text_projector(text_hidden)
            multimodal_hidden = torch.cat([text_proj, audio_hidden], dim=1)
            fused = self.fusion(multimodal_hidden)
            
            valence = self.valence_branch(fused)
            arousal = self.arousal_branch(fused)
            dominance = self.dominance_branch(fused)
        
        # Late fusion
        else:  # fusion_type == 'late'
            # Text predictions
            text_valence = self.text_valence(text_hidden)
            text_arousal = self.text_arousal(text_hidden)
            text_dominance = self.text_dominance(text_hidden)
            
            # Audio predictions
            audio_valence = self.audio_valence(audio_hidden)
            audio_arousal = self.audio_arousal(audio_hidden)
            audio_dominance = self.audio_dominance(audio_hidden)
            
            # Get fusion weights (softmax to ensure they sum to 1)
            weights = self.softmax(self.fusion_weights)
            
            # Weighted fusion
            valence = weights[0, 0] * text_valence + weights[0, 1] * audio_valence
            arousal = weights[1, 0] * text_arousal + weights[1, 1] * audio_arousal
            dominance = weights[2, 0] * text_dominance + weights[2, 1] * audio_dominance
        
        return torch.cat([valence, arousal, dominance], dim=1)

def train_vad_model(model, train_loader, val_loader, num_epochs=10, learning_rate=2e-5, 
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Train a VAD prediction model.
    
    Args:
        model (nn.Module): The model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        num_epochs (int): Number of training epochs
        learning_rate (float): Learning rate
        device (torch.device): Device to use for training
        
    Returns:
        nn.Module: The trained model
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]") as t:
            for batch in t:
                # Get batch inputs
                if isinstance(model, TextVADModel):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    targets = batch['vad_values'].to(device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    outputs = model(input_ids, attention_mask)
                else:  # MultimodalVADModel
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    audio_features = batch['audio_features'].to(device)
                    targets = batch['vad_values'].to(device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    outputs = model(input_ids, attention_mask, audio_features)
                
                # Compute loss and update
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                # Update metrics
                train_loss += loss.item()
                t.set_postfix(loss=loss.item())
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]") as t:
                for batch in t:
                    # Get batch inputs
                    if isinstance(model, TextVADModel):
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        targets = batch['vad_values'].to(device)
                        
                        # Forward pass
                        outputs = model(input_ids, attention_mask)
                    else:  # MultimodalVADModel
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        audio_features = batch['audio_features'].to(device)
                        targets = batch['vad_values'].to(device)
                        
                        # Forward pass
                        outputs = model(input_ids, attention_mask, audio_features)
                    
                    # Compute loss
                    loss = criterion(outputs, targets)
                    
                    # Update metrics
                    val_loss += loss.item()
                    t.set_postfix(loss=loss.item())
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}")
    
    return model

def evaluate_vad_model(model, test_loader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Evaluate a VAD prediction model.
    
    Args:
        model (nn.Module): The model to evaluate
        test_loader (DataLoader): Test data loader
        device (torch.device): Device to use for evaluation
        
    Returns:
        tuple: (predictions, targets, metrics)
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Get batch inputs
            if isinstance(model, TextVADModel):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['vad_values'].to(device)
                
                # Forward pass
                outputs = model(input_ids, attention_mask)
            else:  # MultimodalVADModel
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                audio_features = batch['audio_features'].to(device)
                targets = batch['vad_values'].to(device)
                
                # Forward pass
                outputs = model(input_ids, attention_mask, audio_features)
            
            # Collect predictions and targets
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Stack all predictions and targets
    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    
    # Compute metrics
    mse = mean_squared_error(targets, preds, multioutput='raw_values')
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, preds, multioutput='raw_values')
    r2 = r2_score(targets, preds, multioutput='raw_values')
    
    # Print metrics
    vad_labels = ['Valence', 'Arousal', 'Dominance']
    for i in range(3):
        print(f"{vad_labels[i]} - MSE: {mse[i]:.4f}, RMSE: {rmse[i]:.4f}, MAE: {mae[i]:.4f}, R²: {r2[i]:.4f}")
    
    # Return metrics dictionary
    metrics = {
        'mse': mse.tolist(),
        'rmse': rmse.tolist(),
        'mae': mae.tolist(),
        'r2': r2.tolist()
    }
    
    return preds, targets, metrics

def predict_vad(model, texts, tokenizer, audio_features=None, batch_size=16,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Predict VAD values for new text inputs.
    
    Args:
        model (nn.Module): Trained VAD prediction model
        texts (list): List of text inputs
        tokenizer: Tokenizer for text processing
        audio_features (ndarray, optional): Audio features if using multimodal model
        batch_size (int): Batch size for prediction
        device (torch.device): Device to use for prediction
        
    Returns:
        ndarray: Predicted VAD values (shape: [n_samples, 3])
    """
    model.eval()
    all_preds = []
    
    # Create dataset and dataloader
    dummy_labels = np.zeros((len(texts), 3))  # Dummy VAD values
    
    if audio_features is not None:
        dataset = MultimodalVADDataset(texts, audio_features, dummy_labels, tokenizer)
    else:
        dataset = TextVADDataset(texts, dummy_labels, tokenizer)
    
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    # Make predictions
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            if audio_features is not None:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                audio_feats = batch['audio_features'].to(device)
                
                outputs = model(input_ids, attention_mask, audio_feats)
            else:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model(input_ids, attention_mask)
            
            all_preds.append(outputs.cpu().numpy())
    
    # Stack all predictions
    preds = np.vstack(all_preds)
    
    return preds 