"""
Fine-tuning of pre-trained language models for VAD prediction.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
import logging
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)

class VADDataset(Dataset):
    """
    Dataset for VAD prediction.
    """
    def __init__(self, texts, vad_values, tokenizer, max_length=128):
        """
        Initialize the dataset.
        
        Args:
            texts (list): List of text strings
            vad_values (np.ndarray): Array of VAD values
            tokenizer: Tokenizer for encoding texts
            max_length (int): Maximum sequence length
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
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Remove batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # Add VAD values
        encoding['vad'] = torch.tensor(vad, dtype=torch.float)
        
        return encoding

class VADRegressor(nn.Module):
    """
    Regressor for VAD prediction.
    """
    def __init__(self, model_name, num_labels=3):
        """
        Initialize the regressor.
        
        Args:
            model_name (str): Name of the pre-trained model
            num_labels (int): Number of output labels (3 for VAD)
        """
        super(VADRegressor, self).__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Load pre-trained model
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Add regression head
        self.dropout = nn.Dropout(0.1)
        self.regressor = nn.Linear(self.encoder.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Forward pass.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (torch.Tensor): Attention mask
            token_type_ids (torch.Tensor): Token type IDs
            
        Returns:
            torch.Tensor: Predicted VAD values
        """
        # Get encoder outputs
        if token_type_ids is not None:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        else:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Get the [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout and regression head
        pooled_output = self.dropout(pooled_output)
        vad_values = self.regressor(pooled_output)
        
        return vad_values

class VADFineTuner:
    """
    Fine-tuner for VAD prediction.
    """
    def __init__(self, model_name="roberta-base", device=None, learning_rate=2e-5, batch_size=16, max_length=128):
        """
        Initialize the fine-tuner.
        
        Args:
            model_name (str): Name of the pre-trained model
            device (str): Device to use for training ('cuda' or 'cpu')
            learning_rate (float): Learning rate for optimization
            batch_size (int): Batch size for training
            max_length (int): Maximum sequence length
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_length = max_length
        
        logger.info(f"Initializing VADFineTuner with {model_name} on {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize model
        self.model = VADRegressor(model_name)
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Initialize loss function
        self.criterion = nn.MSELoss()
        
        # Initialize training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_mse': [],
            'val_rmse': [],
            'val_mae': []
        }
        
    def train(self, train_texts, train_vad, val_texts, val_vad, epochs=10, output_dir=None):
        """
        Train the model.
        
        Args:
            train_texts (list): List of training text strings
            train_vad (np.ndarray): Array of training VAD values
            val_texts (list): List of validation text strings
            val_vad (np.ndarray): Array of validation VAD values
            epochs (int): Number of training epochs
            output_dir (str): Directory to save the model and results
            
        Returns:
            dict: Training history
        """
        # Create datasets
        train_dataset = VADDataset(train_texts, train_vad, self.tokenizer, self.max_length)
        val_dataset = VADDataset(val_texts, val_vad, self.tokenizer, self.max_length)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # Create output directory
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save configuration
            config = {
                'model_name': self.model_name,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'max_length': self.max_length,
                'epochs': epochs
            }
            
            with open(os.path.join(output_dir, 'config.json'), 'w') as f:
                json.dump(config, f, indent=2)
        
        # Create log file
        if output_dir:
            log_path = os.path.join(output_dir, 'training_log.txt')
            log_file = open(log_path, 'w')
            log_file.write("epoch,train_loss,val_loss,val_mse,val_rmse,val_mae\n")
        
        # Training loop
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Training)"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                vad = batch['vad'].to(self.device)
                
                # Forward pass
                if 'token_type_ids' in batch:
                    token_type_ids = batch['token_type_ids'].to(self.device)
                    outputs = self.model(input_ids, attention_mask, token_type_ids)
                else:
                    outputs = self.model(input_ids, attention_mask)
                
                # Calculate loss
                loss = self.criterion(outputs, vad)
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update statistics
                train_loss += loss.item() * input_ids.size(0)
            
            # Calculate average training loss
            train_loss /= len(train_loader.dataset)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (Validation)"):
                    # Move batch to device
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    vad = batch['vad'].to(self.device)
                    
                    # Forward pass
                    if 'token_type_ids' in batch:
                        token_type_ids = batch['token_type_ids'].to(self.device)
                        outputs = self.model(input_ids, attention_mask, token_type_ids)
                    else:
                        outputs = self.model(input_ids, attention_mask)
                    
                    # Calculate loss
                    loss = self.criterion(outputs, vad)
                    
                    # Update statistics
                    val_loss += loss.item() * input_ids.size(0)
                    
                    # Collect predictions and targets
                    all_preds.append(outputs.cpu().numpy())
                    all_targets.append(vad.cpu().numpy())
            
            # Calculate average validation loss
            val_loss /= len(val_loader.dataset)
            
            # Calculate validation metrics
            all_preds = np.vstack(all_preds)
            all_targets = np.vstack(all_targets)
            
            val_mse = mean_squared_error(all_targets, all_preds)
            val_rmse = np.sqrt(val_mse)
            val_mae = mean_absolute_error(all_targets, all_preds)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_mse'].append(val_mse)
            self.history['val_rmse'].append(val_rmse)
            self.history['val_mae'].append(val_mae)
            
            # Print progress
            logger.info(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_mse={val_mse:.4f}, val_rmse={val_rmse:.4f}, val_mae={val_mae:.4f}")
            
            # Write to log file
            if output_dir:
                log_file.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f},{val_mse:.6f},{val_rmse:.6f},{val_mae:.6f}\n")
                log_file.flush()
            
            # Save model checkpoint
            if output_dir:
                checkpoint_dir = os.path.join(output_dir, 'checkpoints')
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }, checkpoint_path)
        
        # Close log file
        if output_dir:
            log_file.close()
        
        # Save final model
        if output_dir:
            model_path = os.path.join(output_dir, 'model.pt')
            torch.save(self.model.state_dict(), model_path)
            
            # Save history
            history_path = os.path.join(output_dir, 'history.json')
            with open(history_path, 'w') as f:
                json.dump(self.history, f, indent=2)
            
            # Plot training curves
            self._plot_training_curves(output_dir)
        
        return self.history
    
    def _plot_training_curves(self, output_dir):
        """
        Plot training curves.
        
        Args:
            output_dir (str): Directory to save the plots
        """
        # Plot loss curves
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'loss_curves.png'))
        plt.close()
        
        # Plot validation metrics
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['val_mse'], label='MSE')
        plt.plot(self.history['val_rmse'], label='RMSE')
        plt.plot(self.history['val_mae'], label='MAE')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title('Validation Metrics')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'validation_metrics.png'))
        plt.close()
    
    def predict(self, texts, batch_size=16):
        """
        Predict VAD values for a list of texts.
        
        Args:
            texts (list): List of text strings
            batch_size (int): Batch size for processing
            
        Returns:
            np.ndarray: Array of predicted VAD values
        """
        # Create dataset
        dataset = VADDataset(texts, np.zeros((len(texts), 3)), self.tokenizer, self.max_length)
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # Prediction
        self.model.eval()
        all_preds = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting VAD"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                if 'token_type_ids' in batch:
                    token_type_ids = batch['token_type_ids'].to(self.device)
                    outputs = self.model(input_ids, attention_mask, token_type_ids)
                else:
                    outputs = self.model(input_ids, attention_mask)
                
                # Collect predictions
                all_preds.append(outputs.cpu().numpy())
        
        return np.vstack(all_preds)
    
    def save(self, output_dir):
        """
        Save the model and tokenizer.
        
        Args:
            output_dir (str): Directory to save the model and tokenizer
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(output_dir, 'model.pt')
        torch.save(self.model.state_dict(), model_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # Save configuration
        config = {
            'model_name': self.model_name,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'max_length': self.max_length
        }
        
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model saved to {output_dir}")
    
    @classmethod
    def load(cls, model_dir, device=None):
        """
        Load a model from disk.
        
        Args:
            model_dir (str): Directory containing the saved model
            device (str): Device to use for inference ('cuda' or 'cpu')
            
        Returns:
            VADFineTuner: The loaded model
        """
        # Load configuration
        with open(os.path.join(model_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        
        # Initialize fine-tuner
        fine_tuner = cls(
            model_name=config['model_name'],
            device=device,
            learning_rate=config['learning_rate'],
            batch_size=config['batch_size'],
            max_length=config['max_length']
        )
        
        # Load tokenizer
        fine_tuner.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Load model
        model_path = os.path.join(model_dir, 'model.pt')
        fine_tuner.model.load_state_dict(torch.load(model_path, map_location=fine_tuner.device))
        
        logger.info(f"Model loaded from {model_dir}")
        
        return fine_tuner
