#!/usr/bin/env python3
"""
Text-based VAD (Valence-Arousal-Dominance) Prediction

This module implements a text-based VAD prediction model using a transformer model like BERT
or RoBERTa to predict the continuous VAD values from text input.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
import time

# Constants
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
MAX_SEQ_LENGTH = 128
MODEL_NAME = "roberta-base"  # Could also use sentence-transformers models
RANDOM_SEED = 42

# Paths
DEFAULT_DATA_DIR = "data"
DEFAULT_MODEL_DIR = "checkpoints"
DEFAULT_LOG_DIR = "logs"

# Create directories
os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)
os.makedirs(DEFAULT_MODEL_DIR, exist_ok=True)
os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else 
                     "mps" if torch.backends.mps.is_available() else 
                     "cpu")

print(f"Using device: {device}")

# Set random seeds
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


class TextVADDataset(Dataset):
    """Dataset for text-based VAD prediction"""
    
    def __init__(self, texts, vad_values, tokenizer, max_len=MAX_SEQ_LENGTH):
        self.texts = texts
        self.vad_values = vad_values  # Numpy array with shape (n_samples, 3)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        vad = self.vad_values[idx]
        
        # Tokenize text
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
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'vad_values': torch.tensor(vad, dtype=torch.float)
        }


class TextVADModel(nn.Module):
    """Text-based VAD prediction model using transformer"""
    
    def __init__(self, model_name=MODEL_NAME, dropout=0.1):
        super(TextVADModel, self).__init__()
        
        # Load pre-trained model
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = self.transformer.config.hidden_size
        
        # Freeze embeddings for more stable training
        if hasattr(self.transformer, 'embeddings'):
            for param in self.transformer.embeddings.parameters():
                param.requires_grad = False
        
        # Separate branches for each VAD dimension to allow specialization
        self.shared_layer = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.Dropout(dropout),
            nn.GELU(),
        )
        
        # Valence prediction branch
        self.valence_branch = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
        # Arousal prediction branch
        self.arousal_branch = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
        # Dominance prediction branch
        self.dominance_branch = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, input_ids, attention_mask):
        """Forward pass"""
        # Get transformer output
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use the [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Shared features
        shared_features = self.shared_layer(pooled_output)
        
        # Predict VAD dimensions
        valence = self.valence_branch(shared_features)
        arousal = self.arousal_branch(shared_features)
        dominance = self.dominance_branch(shared_features)
        
        # Combine predictions
        vad = torch.cat([valence, arousal, dominance], dim=1)
        
        return vad


def load_iemocap_data(data_path=None):
    """
    Load IEMOCAP data with VAD values
    
    Args:
        data_path: Path to data file
        
    Returns:
        DataFrame with texts, emotions, and VAD values
    """
    if data_path and os.path.exists(data_path):
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        return df
    
    # Check if we have processed data already
    processed_path = os.path.join(DEFAULT_DATA_DIR, "iemocap_vad.csv")
    if os.path.exists(processed_path):
        print(f"Loading existing data from {processed_path}")
        df = pd.read_csv(processed_path)
        return df
    
    print("No data found. Please provide a path to the IEMOCAP data.")
    print("Creating a small mock dataset for testing purposes.")
    
    # Create a small mock dataset
    mock_data = {
        'text': [
            "I'm so excited about the weekend!",
            "I miss you so much.",
            "I can't believe you did that!",
            "I'm going to the store.",
            "That was the best movie I've seen all year.",
            "I didn't get the promotion I was hoping for.",
            "This is completely unacceptable.",
            "The meeting is at 2pm."
        ],
        'emotion': ['happy', 'sad', 'angry', 'neutral', 'happy', 'sad', 'angry', 'neutral'],
        'valence': [0.8, -0.7, -0.5, 0.0, 0.9, -0.6, -0.8, 0.0],
        'arousal': [0.6, -0.3, 0.8, 0.1, 0.7, -0.2, 0.9, 0.0],
        'dominance': [0.5, -0.4, 0.7, 0.0, 0.6, -0.5, 0.8, 0.0]
    }
    
    df = pd.DataFrame(mock_data)
    
    # Save processed data
    df.to_csv(processed_path, index=False)
    print(f"Created mock dataset with {len(df)} samples and saved to {processed_path}")
    
    return df


def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, num_epochs=NUM_EPOCHS):
    """
    Train the VAD prediction model
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        criterion: Loss function
        num_epochs: Number of epochs to train
        
    Returns:
        Trained model and training history
    """
    # Training history
    history = {
        'train_loss': [],
        'train_mse': [],
        'val_loss': [],
        'val_mse': [],
        'val_r2': []
    }
    
    # Track best model
    best_val_mse = float('inf')
    best_model_path = os.path.join(DEFAULT_MODEL_DIR, "text_vad_best.pt")
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Training phase
        model.train()
        train_loss = 0
        train_mse = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Get batch data
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            vad_values = batch['vad_values'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            
            # Calculate loss
            loss = criterion(outputs, vad_values)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update stats
            train_loss += loss.item()
            train_mse += mean_squared_error(
                vad_values.cpu().detach().numpy(),
                outputs.cpu().detach().numpy()
            )
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        # Adjust learning rate
        scheduler.step()
        
        # Calculate epoch statistics
        train_loss /= len(train_loader)
        train_mse /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Get batch data
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                vad_values = batch['vad_values'].to(device)
                
                # Forward pass
                outputs = model(input_ids, attention_mask)
                
                # Calculate loss
                loss = criterion(outputs, vad_values)
                
                # Update stats
                val_loss += loss.item()
                
                # Save predictions and targets for metrics
                val_preds.append(outputs.cpu().numpy())
                val_targets.append(vad_values.cpu().numpy())
        
        # Calculate validation statistics
        val_loss /= len(val_loader)
        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)
        val_mse = mean_squared_error(val_targets, val_preds)
        val_r2 = r2_score(val_targets, val_preds)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_mse'].append(train_mse)
        history['val_loss'].append(val_loss)
        history['val_mse'].append(val_mse)
        history['val_r2'].append(val_r2)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs} - Time: {time.time() - start_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f}, MSE: {train_mse:.4f}")
        print(f"Val Loss: {val_loss:.4f}, MSE: {val_mse:.4f}, R²: {val_r2:.4f}")
        
        # Save best model
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'mse': val_mse,
                'r2': val_r2
            }, best_model_path)
            print(f"New best model saved to {best_model_path}")
    
    # Save final model
    final_model_path = os.path.join(DEFAULT_MODEL_DIR, "text_vad_final.pt")
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': val_loss,
        'mse': val_mse,
        'r2': val_r2
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    return model, history


def evaluate_model(model, test_loader):
    """
    Evaluate the VAD prediction model
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        
    Returns:
        Evaluation metrics and predictions
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    test_loss = 0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for batch in test_loader:
            # Get batch data
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            vad_values = batch['vad_values'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            
            # Calculate loss
            loss = criterion(outputs, vad_values)
            
            # Update stats
            test_loss += loss.item()
            
            # Save predictions and targets for metrics
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(vad_values.cpu().numpy())
    
    # Calculate evaluation statistics
    test_loss /= len(test_loader)
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Calculate metrics per dimension
    mse_per_dim = np.mean((all_preds - all_targets) ** 2, axis=0)
    r2_per_dim = [r2_score(all_targets[:, i], all_preds[:, i]) for i in range(3)]
    
    # Overall metrics
    overall_mse = mean_squared_error(all_targets, all_preds)
    overall_r2 = r2_score(all_targets.flatten(), all_preds.flatten())
    
    # Create metrics dict
    metrics = {
        'loss': test_loss,
        'mse': overall_mse,
        'r2': overall_r2,
        'valence_mse': mse_per_dim[0],
        'arousal_mse': mse_per_dim[1],
        'dominance_mse': mse_per_dim[2],
        'valence_r2': r2_per_dim[0],
        'arousal_r2': r2_per_dim[1],
        'dominance_r2': r2_per_dim[2]
    }
    
    # Print metrics
    print("\nEvaluation Results:")
    print("-" * 40)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Overall MSE: {overall_mse:.4f}")
    print(f"Overall R²: {overall_r2:.4f}")
    print("\nPer-Dimension Metrics:")
    print(f"Valence - MSE: {mse_per_dim[0]:.4f}, R²: {r2_per_dim[0]:.4f}")
    print(f"Arousal - MSE: {mse_per_dim[1]:.4f}, R²: {r2_per_dim[1]:.4f}")
    print(f"Dominance - MSE: {mse_per_dim[2]:.4f}, R²: {r2_per_dim[2]:.4f}")
    
    return metrics, all_preds, all_targets


def plot_training_history(history):
    """
    Plot training history
    
    Args:
        history: Training history
    """
    # Create figure and axes
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training and validation loss
    axs[0].plot(history['train_loss'], label='Train Loss')
    axs[0].plot(history['val_loss'], label='Validation Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss (MSE)')
    axs[0].set_title('Training and Validation Loss')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot validation R²
    axs[1].plot(history['val_r2'], label='Validation R²')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('R²')
    axs[1].set_title('Validation R²')
    axs[1].legend()
    axs[1].grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(DEFAULT_LOG_DIR, 'training_history.png'))
    plt.show()


def vad_to_emotion(valence, arousal, dominance):
    """
    Convert VAD values to emotion category using rule-based approach
    
    Args:
        valence: Valence value (-1 to 1)
        arousal: Arousal value (-1 to 1)
        dominance: Dominance value (-1 to 1)
        
    Returns:
        Emotion category
    """
    # Simple rule-based mapping based on VAD space
    if valence >= 0.2:
        if arousal >= 0.2:
            if dominance >= 0.2:
                return "happy"
            else:
                return "excited"
        else:
            if dominance >= 0.2:
                return "satisfied"
            else:
                return "relaxed"
    else:
        if arousal >= 0.2:
            if dominance >= 0.2:
                return "angry"
            else:
                return "afraid"
        else:
            if dominance >= 0.2:
                return "sad"
            else:
                return "depressed"


# Helper function to convert numpy types to native Python types for JSON serialization
def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    return obj


def main(args):
    """Main function to run the training and evaluation"""
    # Step 1: Load and prepare data
    print("Loading dataset...")
    df = load_iemocap_data(args.data_path)
    
    # Ensure the VAD values are in the [-1, 1] range
    if df['valence'].max() > 1 or df['arousal'].max() > 1 or df['dominance'].max() > 1:
        print("Normalizing VAD values to [-1, 1] range...")
        if df['valence'].max() > 5:  # Assume 1-5 scale
            # Convert from 1-5 scale to [-1, 1] scale
            df['valence'] = (df['valence'] - 3) / 2
            df['arousal'] = (df['arousal'] - 3) / 2
            df['dominance'] = (df['dominance'] - 3) / 2
    
    # Split data
    texts = df['text'].values
    vad_values = df[['valence', 'arousal', 'dominance']].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        texts, vad_values, test_size=0.2, random_state=RANDOM_SEED
    )
    
    # Further split training data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=RANDOM_SEED
    )
    
    # Step 2: Create datasets and dataloaders
    print("Creating datasets and dataloaders...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    train_dataset = TextVADDataset(X_train, y_train, tokenizer)
    val_dataset = TextVADDataset(X_val, y_val, tokenizer)
    test_dataset = TextVADDataset(X_test, y_test, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Step 3: Initialize model
    print(f"Initializing model with {args.model_name}...")
    model = TextVADModel(model_name=args.model_name).to(device)
    
    # Step 4: Set up training
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    # Learning rate scheduler
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.1 * total_steps,
        num_training_steps=total_steps
    )
    
    # Step 5: Train model
    if not args.eval_only:
        print("Training model...")
        model, history = train_model(
            model, train_loader, val_loader, optimizer, scheduler, criterion, args.num_epochs
        )
        
        # Plot training history
        plot_training_history(history)
    else:
        # Load pre-trained model
        print(f"Loading model from {args.model_path}...")
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Step 6: Evaluate model
    print("Evaluating model...")
    metrics, predictions, targets = evaluate_model(model, test_loader)
    
    # Save metrics
    metrics_path = os.path.join(DEFAULT_LOG_DIR, 'text_vad_metrics.json')
    # Convert metrics dictionary before saving
    metrics_serializable = convert_numpy(metrics) 
    with open(metrics_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=4)
    print(f"Metrics saved to {metrics_path}")
    
    # Step 7: Convert VAD predictions to emotions
    test_texts = X_test
    pred_emotions = []
    true_emotions = []
    
    for i in range(len(predictions)):
        v_pred, a_pred, d_pred = predictions[i]
        v_true, a_true, d_true = targets[i]
        
        pred_emotion = vad_to_emotion(v_pred, a_pred, d_pred)
        true_emotion = vad_to_emotion(v_true, a_true, d_true)
        
        pred_emotions.append(pred_emotion)
        true_emotions.append(true_emotion)
    
    # Calculate emotion classification accuracy
    correct = sum(p == t for p, t in zip(pred_emotions, true_emotions))
    accuracy = correct / len(pred_emotions) * 100
    
    print(f"\nEmotion Classification Accuracy: {accuracy:.2f}%")
    
    # Save emotion predictions
    results = []
    for i in range(min(10, len(test_texts))):  # Show first 10 examples
        results.append({
            'text': test_texts[i],
            'true_vad': targets[i].tolist(),
            'pred_vad': predictions[i].tolist(),
            'true_emotion': true_emotions[i],
            'pred_emotion': pred_emotions[i]
        })
    
    results_path = os.path.join(DEFAULT_LOG_DIR, 'text_vad_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Sample results saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text-based VAD Prediction")
    parser.add_argument('--data_path', type=str, default=None, help='Path to data file')
    parser.add_argument('--model_name', type=str, default=MODEL_NAME, help='Pre-trained model name')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('--eval_only', action='store_true', help='Only evaluate model')
    parser.add_argument('--model_path', type=str, default=None, help='Path to pre-trained model')
    
    args = parser.parse_args()
    
    main(args) 