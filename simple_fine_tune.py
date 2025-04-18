#!/usr/bin/env python3
"""
Simplified script for fine-tuning a pre-trained language model for VAD prediction.
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VADDataset(Dataset):
    """Dataset for VAD prediction."""
    def __init__(self, texts, vad_values, tokenizer, max_length=128):
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
    """Regressor for VAD prediction."""
    def __init__(self, model_name, num_labels=3):
        super(VADRegressor, self).__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Load pre-trained model
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Add regression head
        self.dropout = nn.Dropout(0.1)
        self.regressor = nn.Linear(self.encoder.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
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

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Fine-tune a pre-trained language model for VAD prediction')
    
    parser.add_argument('--data_path', type=str, default='IEMOCAP_Final.csv',
                        help='Path to IEMOCAP_Final.csv')
    parser.add_argument('--output_dir', type=str, default='results/simple_fine_tuning',
                        help='Directory to save results')
    parser.add_argument('--model_name', type=str, default='facebook/bart-large-mnli',
                        help='Name of the pre-trained model')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate for optimization')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data for testing')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Proportion of data for validation')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

def extract_vad_values(dimension_str):
    """Extract VAD values from the dimension string."""
    try:
        # Convert string representation to list of dictionaries
        dimension_list = eval(dimension_str)
        
        # Extract the first dictionary (assuming there's only one)
        if isinstance(dimension_list, list) and len(dimension_list) > 0:
            vad_dict = dimension_list[0]
            return vad_dict
        return None
    except:
        return None

def preprocess_data(df):
    """Preprocess IEMOCAP data."""
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Extract VAD values from the 'dimension' column
    df['vad_values'] = df['dimension'].apply(extract_vad_values)
    
    # Create separate columns for valence, arousal, and dominance
    df['valence'] = df['vad_values'].apply(lambda x: x['valence'] if x else None)
    df['arousal'] = df['vad_values'].apply(lambda x: x['arousal'] if x else None)
    df['dominance'] = df['vad_values'].apply(lambda x: x['dominance'] if x else None)
    
    # Drop rows with missing VAD values
    df = df.dropna(subset=['valence', 'arousal', 'dominance'])
    
    # Use Major_emotion as the emotion label
    df['emotion'] = df['Major_emotion'].str.strip()
    
    # Keep only necessary columns
    cols_to_keep = ['Transcript', 'valence', 'arousal', 'dominance', 'emotion']
    df = df[cols_to_keep]
    
    return df

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data")
    df = pd.read_csv(args.data_path)
    df = preprocess_data(df)
    
    # Split data into train, validation, and test sets
    train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=args.random_state)
    train_df, val_df = train_test_split(train_df, test_size=args.val_size/(1-args.test_size), random_state=args.random_state)
    
    logger.info(f"Data split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    # Scale VAD values
    logger.info("Scaling VAD values")
    scaler = StandardScaler()
    train_vad = train_df[['valence', 'arousal', 'dominance']].values
    val_vad = val_df[['valence', 'arousal', 'dominance']].values
    test_vad = test_df[['valence', 'arousal', 'dominance']].values
    
    train_vad_scaled = scaler.fit_transform(train_vad)
    val_vad_scaled = scaler.transform(val_vad)
    test_vad_scaled = scaler.transform(test_vad)
    
    # Save scaler
    scaler_path = os.path.join(args.output_dir, 'vad_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        import pickle
        pickle.dump(scaler, f)
    
    # Extract texts
    train_texts = train_df['Transcript'].tolist()
    val_texts = val_df['Transcript'].tolist()
    test_texts = test_df['Transcript'].tolist()
    
    # Initialize tokenizer and model
    logger.info(f"Initializing tokenizer and model with {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = VADRegressor(args.model_name)
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Create datasets and dataloaders
    train_dataset = VADDataset(train_texts, train_vad_scaled, tokenizer, args.max_length)
    val_dataset = VADDataset(val_texts, val_vad_scaled, tokenizer, args.max_length)
    test_dataset = VADDataset(test_texts, test_vad_scaled, tokenizer, args.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    
    # Create log file
    log_path = os.path.join(args.output_dir, 'training_log.txt')
    with open(log_path, 'w') as f:
        f.write("epoch,train_loss,val_loss,val_mse,val_rmse,val_mae\n")
    
    # Training loop
    logger.info("Starting training")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Training)"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            vad = batch['vad'].to(device)
            
            # Forward pass
            if 'token_type_ids' in batch:
                token_type_ids = batch['token_type_ids'].to(device)
                outputs = model(input_ids, attention_mask, token_type_ids)
            else:
                outputs = model(input_ids, attention_mask)
            
            # Calculate loss
            loss = criterion(outputs, vad)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item() * input_ids.size(0)
        
        # Calculate average training loss
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Validation)"):
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                vad = batch['vad'].to(device)
                
                # Forward pass
                if 'token_type_ids' in batch:
                    token_type_ids = batch['token_type_ids'].to(device)
                    outputs = model(input_ids, attention_mask, token_type_ids)
                else:
                    outputs = model(input_ids, attention_mask)
                
                # Calculate loss
                loss = criterion(outputs, vad)
                
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
        
        # Print progress
        logger.info(f"Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_mse={val_mse:.4f}, val_rmse={val_rmse:.4f}, val_mae={val_mae:.4f}")
        
        # Write to log file
        with open(log_path, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f},{val_mse:.6f},{val_rmse:.6f},{val_mae:.6f}\n")
        
        # Save model if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved best model with validation loss {val_loss:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pt')))
    
    # Evaluate on test set
    logger.info("Evaluating on test set")
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            vad = batch['vad'].to(device)
            
            # Forward pass
            if 'token_type_ids' in batch:
                token_type_ids = batch['token_type_ids'].to(device)
                outputs = model(input_ids, attention_mask, token_type_ids)
            else:
                outputs = model(input_ids, attention_mask)
            
            # Calculate loss
            loss = criterion(outputs, vad)
            
            # Update statistics
            test_loss += loss.item() * input_ids.size(0)
            
            # Collect predictions and targets
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(vad.cpu().numpy())
    
    # Calculate average test loss
    test_loss /= len(test_loader.dataset)
    
    # Calculate test metrics
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    test_mse = mean_squared_error(all_targets, all_preds)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(all_targets, all_preds)
    
    # Calculate metrics for each dimension
    dim_metrics = {}
    for i, dim in enumerate(['valence', 'arousal', 'dominance']):
        dim_mse = mean_squared_error(all_targets[:, i], all_preds[:, i])
        dim_rmse = np.sqrt(dim_mse)
        dim_mae = mean_absolute_error(all_targets[:, i], all_preds[:, i])
        
        dim_metrics[dim] = {
            'mse': float(dim_mse),
            'rmse': float(dim_rmse),
            'mae': float(dim_mae)
        }
    
    # Save test metrics
    test_metrics = {
        'loss': float(test_loss),
        'mse': float(test_mse),
        'rmse': float(test_rmse),
        'mae': float(test_mae),
        'dim_metrics': dim_metrics
    }
    
    with open(os.path.join(args.output_dir, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    # Print test metrics
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test MSE: {test_mse:.4f}")
    logger.info(f"Test RMSE: {test_rmse:.4f}")
    logger.info(f"Test MAE: {test_mae:.4f}")
    
    for dim, metrics in dim_metrics.items():
        logger.info(f"{dim.capitalize()} MSE: {metrics['mse']:.4f}")
        logger.info(f"{dim.capitalize()} RMSE: {metrics['rmse']:.4f}")
        logger.info(f"{dim.capitalize()} MAE: {metrics['mae']:.4f}")
    
    # Save predictions
    test_df['pred_valence'] = scaler.inverse_transform(all_preds)[:, 0]
    test_df['pred_arousal'] = scaler.inverse_transform(all_preds)[:, 1]
    test_df['pred_dominance'] = scaler.inverse_transform(all_preds)[:, 2]
    
    test_df.to_csv(os.path.join(args.output_dir, 'test_predictions.csv'), index=False)
    
    logger.info(f"All results saved to {args.output_dir}")

if __name__ == '__main__':
    main()
