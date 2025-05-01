"""
Multimodal VAD prediction model.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

class MultimodalVADDataset(Dataset):
    """
    Dataset for multimodal VAD prediction.
    """
    def __init__(self, text_features, audio_features, vad_values):
        """
        Initialize the dataset.
        
        Args:
            text_features: Text features (input_ids, attention_mask)
            audio_features: Audio features (mel spectrograms)
            vad_values: Array of VAD values (valence, arousal, dominance)
        """
        self.text_input_ids = text_features['input_ids']
        self.text_attention_mask = text_features['attention_mask']
        self.audio_features = audio_features
        self.vad_values = vad_values
    
    def __len__(self):
        return len(self.vad_values)
    
    def __getitem__(self, idx):
        return {
            'text_input_ids': self.text_input_ids[idx],
            'text_attention_mask': self.text_attention_mask[idx],
            'audio_features': self.audio_features[idx],
            'vad_values': torch.tensor(self.vad_values[idx], dtype=torch.float)
        }

class MultimodalVADModel(nn.Module):
    """
    Multimodal VAD prediction model.
    """
    def __init__(self, text_model, audio_model, fusion_dropout=0.3):
        """
        Initialize the model.
        
        Args:
            text_model: Pretrained text VAD model
            audio_model: Pretrained audio VAD model
            fusion_dropout: Dropout rate for fusion layers
        """
        super(MultimodalVADModel, self).__init__()
        
        # Load pretrained models
        self.text_model = text_model
        self.audio_model = audio_model
        
        # Freeze the pretrained models
        for param in self.text_model.parameters():
            param.requires_grad = False
        for param in self.audio_model.parameters():
            param.requires_grad = False
        
        # Fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(6, 128),  # 3 (text VAD) + 3 (audio VAD) = 6
            nn.LayerNorm(128),
            nn.Dropout(fusion_dropout),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.Dropout(fusion_dropout),
            nn.ReLU(),
            nn.Linear(64, 3)  # Output: VAD
        )
    
    def forward(self, text_input_ids, text_attention_mask, audio_features):
        """
        Forward pass.
        
        Args:
            text_input_ids: Text input IDs
            text_attention_mask: Text attention mask
            audio_features: Audio features
            
        Returns:
            VAD predictions (valence, arousal, dominance)
        """
        # Get text VAD predictions
        with torch.no_grad():
            text_vad = self.text_model(text_input_ids, text_attention_mask)
        
        # Get audio VAD predictions
        with torch.no_grad():
            audio_vad = self.audio_model(audio_features)
        
        # Concatenate text and audio VAD predictions
        multimodal_features = torch.cat([text_vad, audio_vad], dim=1)
        
        # Fusion
        vad_predictions = self.fusion_layer(multimodal_features)
        
        return vad_predictions

class MultimodalVADTrainer:
    """
    Trainer for the multimodal VAD prediction model.
    """
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the trainer.
        
        Args:
            model: MultimodalVADModel instance
            device: Device to use for training
        """
        self.model = model
        self.device = device
        self.model.to(device)
    
    def train(self, train_loader, test_loader, num_epochs=10, learning_rate=1e-4):
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training data
            test_loader: DataLoader for testing data
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            
        Returns:
            Trained model
        """
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                text_input_ids = batch['text_input_ids'].to(self.device)
                text_attention_mask = batch['text_attention_mask'].to(self.device)
                audio_features = batch['audio_features'].to(self.device)
                targets = batch['vad_values'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(text_input_ids, text_attention_mask, audio_features)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(train_loader):.4f}")
            
            # Evaluation
            if (epoch + 1) % 2 == 0 or epoch == num_epochs - 1:
                self.evaluate(test_loader)
        
        return self.model
    
    def evaluate(self, test_loader):
        """
        Evaluate the model.
        
        Args:
            test_loader: DataLoader for testing data
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for batch in test_loader:
                text_input_ids = batch['text_input_ids'].to(self.device)
                text_attention_mask = batch['text_attention_mask'].to(self.device)
                audio_features = batch['audio_features'].to(self.device)
                targets = batch['vad_values'].to(self.device)
                
                outputs = self.model(text_input_ids, text_attention_mask, audio_features)
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # Stack all predictions and targets
        preds = np.vstack(all_preds)
        targets = np.vstack(all_targets)
        
        # Compute metrics per VAD dimension
        mse = mean_squared_error(targets, preds, multioutput='raw_values')
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, preds, multioutput='raw_values')
        r2 = r2_score(targets, preds, multioutput='raw_values')
        
        # Print metrics
        vad_labels = ['Valence', 'Arousal', 'Dominance']
        for i in range(3):
            print(f"{vad_labels[i]} - MSE: {mse[i]:.4f}, RMSE: {rmse[i]:.4f}, MAE: {mae[i]:.4f}, R²: {r2[i]:.4f}")
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def predict(self, text_input_ids, text_attention_mask, audio_features):
        """
        Predict VAD values for new inputs.
        
        Args:
            text_input_ids: Text input IDs
            text_attention_mask: Text attention mask
            audio_features: Audio features
            
        Returns:
            VAD predictions (valence, arousal, dominance)
        """
        self.model.eval()
        
        # Move inputs to device
        text_input_ids = text_input_ids.to(self.device)
        text_attention_mask = text_attention_mask.to(self.device)
        audio_features = audio_features.to(self.device)
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(text_input_ids, text_attention_mask, audio_features)
        
        return outputs.cpu().numpy()
