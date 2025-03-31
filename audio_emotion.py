#!/usr/bin/env python3
"""
Audio-based Emotion Recognition

This module implements an audio-based emotion recognition model using spectral features
with direct classification to emotion categories.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from tabulate import tabulate
import argparse
import json
import time
from pathlib import Path
import librosa

# Constants
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
NUM_EPOCHS = 5
RANDOM_SEED = 42
EMOTIONS = ["angry", "happy", "neutral", "sad"]
AUDIO_FEATURE_DIM = 68  # Spectral features dimension

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


class AudioEmotionDataset(Dataset):
    """Dataset for audio-based emotion recognition"""
    
    def __init__(self, audio_features, emotions):
        self.audio_features = audio_features
        self.emotions = emotions
        self.emotion_to_id = {emotion: i for i, emotion in enumerate(EMOTIONS)}
        
    def __len__(self):
        return len(self.audio_features)
    
    def __getitem__(self, idx):
        audio_feature = self.audio_features[idx]
        emotion = self.emotions[idx]
        
        # Convert emotion to label id
        label = self.emotion_to_id.get(emotion, 0)  # Default to first class if unknown
        
        # Convert features to tensor if not already
        if not isinstance(audio_feature, torch.Tensor):
            audio_feature = torch.tensor(audio_feature, dtype=torch.float)
        
        return {
            'audio_features': audio_feature,
            'label': torch.tensor(label, dtype=torch.long)
        }


class AudioEmotionClassifier(nn.Module):
    """Audio emotion classifier with residual connections"""
    
    def __init__(self, input_dim=AUDIO_FEATURE_DIM, num_classes=len(EMOTIONS)):
        super(AudioEmotionClassifier, self).__init__()
        
        # Batch normalization for input
        self.input_norm = nn.BatchNorm1d(input_dim)
        
        # First block
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        # Residual block 1
        self.res1 = ResidualBlock(256)
        
        # Residual block 2
        self.res2 = ResidualBlock(256)
        
        # Output layer
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for better convergence"""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Forward pass"""
        # Normalize input
        x = self.input_norm(x)
        
        # First block
        x = self.block1(x)
        
        # Residual blocks
        x = self.res1(x)
        x = self.res2(x)
        
        # Classification head
        logits = self.classifier(x)
        
        return logits


class ResidualBlock(nn.Module):
    """Residual block with batch normalization"""
    
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
    
    def forward(self, x):
        identity = x
        out = self.block(x)
        out = out + identity  # Residual connection
        out = F.leaky_relu(out, 0.2)
        return out


def extract_audio_features(audio_path):
    """
    Extract spectral features from audio file
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        numpy array of features
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        
        # Extract features - MFCC, spectral features, etc.
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).mean(axis=1)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
        zcr = librosa.feature.zero_crossing_rate(y).mean()
        
        # Chromagram
        chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
        
        # Create feature vector
        features = np.hstack([
            mfcc,
            spectral_centroid,
            spectral_bandwidth,
            spectral_rolloff,
            zcr,
            chroma
        ])
        
        # Ensure consistent size
        if len(features) < AUDIO_FEATURE_DIM:
            features = np.pad(features, (0, AUDIO_FEATURE_DIM - len(features)))
        elif len(features) > AUDIO_FEATURE_DIM:
            features = features[:AUDIO_FEATURE_DIM]
        
        return features
    
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return np.zeros(AUDIO_FEATURE_DIM)


def load_data(data_path=None):
    """
    Load dataset with audio features or create mock data
    
    Args:
        data_path: Path to data file
        
    Returns:
        DataFrame with audio_features and emotions
    """
    if data_path and os.path.exists(data_path):
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        return df
    
    # Check if we have processed data
    processed_path = os.path.join(DEFAULT_DATA_DIR, "processed_data.csv")
    if os.path.exists(processed_path):
        print(f"Loading existing data from {processed_path}")
        df = pd.read_csv(processed_path)
        
        # Process audio features if needed
        if 'audio_features' in df.columns and isinstance(df['audio_features'].iloc[0], str):
            print("Converting string audio features to arrays...")
            
            # Parse string representation of audio features to numpy arrays
            def parse_array(x):
                if not isinstance(x, str):
                    return x
                try:
                    x = x.strip('[]')
                    x = x.replace('\n', ' ')
                    values = [float(val) for val in x.split()]
                    return np.array(values)
                except Exception as e:
                    print(f"Error parsing array: {e}")
                    return np.zeros(AUDIO_FEATURE_DIM)
            
            df['audio_features'] = df['audio_features'].apply(parse_array)
        
        return df
    
    print("No data found, creating mock dataset")
    
    # Create mock data with random features
    mock_data = []
    for emotion in EMOTIONS:
        # Create 250 samples for each emotion
        for i in range(250):
            mock_data.append({
                'audio_features': np.random.randn(AUDIO_FEATURE_DIM),
                'emotion': emotion
            })
    
    df = pd.DataFrame(mock_data)
    
    # Save processed data
    # Convert numpy arrays to strings for CSV
    df_save = df.copy()
    df_save['audio_features'] = df_save['audio_features'].apply(lambda x: ' '.join(map(str, x)))
    df_save.to_csv(processed_path, index=False)
    
    print(f"Created mock dataset with {len(df)} samples and saved to {processed_path}")
    
    return df


def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, num_epochs=NUM_EPOCHS):
    """
    Train the model
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        criterion: Loss function
        num_epochs: Number of epochs to train for
        
    Returns:
        Trained model and training history
    """
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Best validation accuracy
    best_val_acc = 0.0
    best_model_path = os.path.join(DEFAULT_MODEL_DIR, "audio_emotion_best.pt")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Training
        model.train()
        train_loss = 0.0
        train_corrects = 0
        train_total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            audio_features = batch['audio_features'].to(device)
            labels = batch['label'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(audio_features)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Track statistics
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_corrects += torch.sum(preds == labels).item()
            train_total += labels.size(0)
            
            # Print progress
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        # Calculate training metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = train_corrects / train_total * 100
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                audio_features = batch['audio_features'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                outputs = model(audio_features)
                loss = criterion(outputs, labels)
                
                # Track statistics
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels).item()
                val_total += labels.size(0)
        
        # Calculate validation metrics
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = val_corrects / val_total * 100
        
        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step(epoch_val_loss)
        
        # Save history
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        # Print epoch statistics
        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")
        print(f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
        
        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with accuracy {best_val_acc:.2f}%")
    
    # Save final model
    final_model_path = os.path.join(DEFAULT_MODEL_DIR, "audio_emotion_final.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")
    
    return model, history


def evaluate_model(model, test_loader):
    """
    Evaluate the model on the test set
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Move data to device
            audio_features = batch['audio_features'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(audio_features)
            _, preds = torch.max(outputs, 1)
            
            # Track predictions
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds) * 100
    
    # Get classification report
    report = classification_report(all_labels, all_preds, 
                                  target_names=EMOTIONS, 
                                  output_dict=True)
    
    # Get confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm
    }


def plot_confusion_matrix(cm, classes=EMOTIONS):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        classes: Class names
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save plot
    plt.savefig(os.path.join(DEFAULT_LOG_DIR, 'audio_confusion_matrix.png'))
    plt.close()


def plot_training_history(history):
    """
    Plot training history
    
    Args:
        history: Training history dictionary
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(DEFAULT_LOG_DIR, 'audio_training_history.png'))
    plt.close()


def print_classification_report(report):
    """
    Print classification report in a nice format
    
    Args:
        report: Classification report dictionary
    """
    print("\nClassification Report:")
    print("-" * 60)
    
    # Print per-class metrics
    rows = []
    for cls in EMOTIONS:
        if cls in report:
            rows.append([
                cls,
                f"{report[cls]['precision']:.4f}",
                f"{report[cls]['recall']:.4f}",
                f"{report[cls]['f1-score']:.4f}",
                f"{report[cls]['support']}"
            ])
    
    print(tabulate(rows, headers=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'], tablefmt='grid'))
    
    # Print summary metrics
    summary_rows = []
    for avg_type in ['accuracy', 'macro avg', 'weighted avg']:
        if avg_type == 'accuracy':
            summary_rows.append([avg_type, '', '', f"{report[avg_type]:.4f}", ''])
        elif avg_type in report:
            metrics = report[avg_type]
            summary_rows.append([
                avg_type,
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['f1-score']:.4f}",
                f"{metrics['support']}"
            ])
    
    print(tabulate(summary_rows, headers=['Metric', 'Precision', 'Recall', 'F1-Score', 'Support'], tablefmt='grid'))
    print("-" * 60)


def main(args):
    """Main function"""
    print("\nAudio-based Emotion Recognition")
    print("=" * 60)
    
    # Load data
    df = load_data(args.data_path)
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(df)}")
    emotion_counts = df['emotion'].value_counts()
    for emotion, count in emotion_counts.items():
        print(f"  {emotion}: {count} ({count/len(df)*100:.2f}%)")
    
    # Split data
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=RANDOM_SEED, stratify=df['emotion'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=RANDOM_SEED, stratify=temp_df['emotion'])
    
    print(f"\nSplit sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Create datasets
    train_dataset = AudioEmotionDataset(train_df['audio_features'].values, train_df['emotion'].values)
    val_dataset = AudioEmotionDataset(val_df['audio_features'].values, val_df['emotion'].values)
    test_dataset = AudioEmotionDataset(test_df['audio_features'].values, test_df['emotion'].values)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = AudioEmotionClassifier().to(device)
    
    # Load pre-trained model if exists and not training
    best_model_path = os.path.join(DEFAULT_MODEL_DIR, "audio_emotion_best.pt")
    if os.path.exists(best_model_path) and not args.train:
        print(f"\nLoading pre-trained model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    elif args.evaluate and not os.path.exists(best_model_path):
        print("\nError: No pre-trained model found for evaluation.")
        print(f"Expected model at {best_model_path}")
        print("Please train a model first.")
        return
    
    # Training
    if args.train or not os.path.exists(best_model_path):
        print("\nTraining model...")
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # Initialize scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        
        # Initialize criterion
        criterion = nn.CrossEntropyLoss()
        
        # Train model
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            num_epochs=args.epochs if args.epochs else NUM_EPOCHS
        )
        
        # Plot training history
        plot_training_history(history)
    
    # Evaluation
    if args.evaluate or args.train:
        print("\nEvaluating model on test set...")
        metrics = evaluate_model(model, test_loader)
        
        # Print results
        print(f"\nTest Accuracy: {metrics['accuracy']:.2f}%")
        print_classification_report(metrics['report'])
        
        # Plot confusion matrix
        plot_confusion_matrix(metrics['confusion_matrix'])
        print(f"Confusion matrix saved to {os.path.join(DEFAULT_LOG_DIR, 'audio_confusion_matrix.png')}")
    
    print("\nDone!")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Audio-based Emotion Recognition")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--data-path", type=str, help="Path to data file")
    parser.add_argument("--epochs", type=int, help="Number of epochs to train for")
    args = parser.parse_args()
    
    # If no action specified, run both training and evaluation
    if not args.train and not args.evaluate:
        args.train = True
        args.evaluate = True
    
    main(args) 