"""
Text-based VAD prediction model.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

class TextVADDataset(Dataset):
    """
    Dataset for text-based VAD prediction.
    """
    def __init__(self, texts, vad_values, tokenizer, max_length=128):
        """
        Initialize the dataset.
        
        Args:
            texts: List of text transcripts
            vad_values: Array of VAD values (valence, arousal, dominance)
            tokenizer: Tokenizer for text encoding
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.vad_values = vad_values
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        vad = self.vad_values[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'vad_values': torch.tensor(vad, dtype=torch.float)
        }

class TextVADModel(nn.Module):
    """
    Text-based VAD prediction model.
    """
    def __init__(self, model_name='roberta-base', dropout=0.1):
        """
        Initialize the model.
        
        Args:
            model_name: Name of the pretrained transformer model
            dropout: Dropout rate
        """
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
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            VAD predictions (valence, arousal, dominance)
        """
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        shared = self.shared_layer(pooled)
        valence = self.valence_branch(shared)
        arousal = self.arousal_branch(shared)
        dominance = self.dominance_branch(shared)
        return torch.cat([valence, arousal, dominance], dim=1)

class TextVADTrainer:
    """
    Trainer for the text-based VAD prediction model.
    """
    def __init__(self, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the trainer.
        
        Args:
            model: TextVADModel instance
            tokenizer: Tokenizer for text encoding
            device: Device to use for training
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
    
    def train(self, train_loader, test_loader, num_epochs=20, learning_rate=2e-5):
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
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=learning_rate, 
            steps_per_epoch=len(train_loader), 
            epochs=num_epochs
        )
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                targets = batch['vad_values'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(train_loader):.4f}")
            
            # Evaluation
            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
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
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                targets = batch['vad_values'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
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
    
    def predict(self, texts):
        """
        Predict VAD values for new texts.
        
        Args:
            texts: List of text transcripts
            
        Returns:
            VAD predictions (valence, arousal, dominance)
        """
        self.model.eval()
        
        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
        
        return outputs.cpu().numpy()
