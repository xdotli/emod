#!/usr/bin/env python3
"""
Multimodal Emotion Recognition

This module implements a multimodal emotion recognition system by combining
text and audio modalities with an attention-based fusion approach.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from tabulate import tabulate
import argparse
import json
import time
from pathlib import Path

# Import models from individual modality files
from text_emotion import TextEmotionClassifier
from audio_emotion import AudioEmotionClassifier

# Constants
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
MAX_SEQ_LENGTH = 128
MODEL_NAME = "roberta-base"
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


class MultimodalEmotionDataset(Dataset):
    """Dataset for multimodal emotion recognition"""
    
    def __init__(self, df, tokenizer, max_len=MAX_SEQ_LENGTH):
        self.df = df
        self.texts = df['text'].values
        self.audio_features = df['audio_features'].values
        self.emotions = df['emotion'].values
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.emotion_to_id = {emotion: i for i, emotion in enumerate(EMOTIONS)}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        audio_feature = self.audio_features[idx]
        emotion = self.emotions[idx]
        
        # Convert emotion to label id
        label = self.emotion_to_id.get(emotion, 0)  # Default to first class if unknown
        
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
        
        # Convert audio features to tensor if not already
        if not isinstance(audio_feature, torch.Tensor):
            audio_feature = torch.tensor(audio_feature, dtype=torch.float)
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'audio_features': audio_feature,
            'label': torch.tensor(label, dtype=torch.long)
        }


class MultimodalFusionClassifier(nn.Module):
    """
    Multimodal fusion model for emotion recognition.
    
    Combines a pre-trained text model and audio model with 
    a cross-attention fusion mechanism.
    """
    def __init__(self, num_classes=len(EMOTIONS), text_embedding_dim=768, audio_embedding_dim=256, hidden_dim=256):
        super(MultimodalFusionClassifier, self).__init__()
        
        # Load pre-trained models
        self.text_model = RobertaModel.from_pretrained(MODEL_NAME)
        self.audio_model = AudioEmotionClassifier()
        
        # Freeze the pre-trained models
        for param in self.text_model.parameters():
            param.requires_grad = False
        
        for param in self.audio_model.parameters():
            param.requires_grad = False
        
        # Audio encoder to get embeddings from audio model
        self.audio_encoder = nn.Sequential(
            nn.Linear(num_classes, audio_embedding_dim),
            nn.LayerNorm(audio_embedding_dim),
            nn.GELU()
        )
        
        # Text encoder to get embeddings from text model
        self.text_encoder = nn.Sequential(
            nn.Linear(text_embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Cross-attention for fusion
        self.cross_attention = CrossAttention(hidden_dim, audio_embedding_dim)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + audio_embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for better convergence"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
                
    def load_pretrained_models(self, text_path, audio_path):
        """
        Load pre-trained weights for text and audio models
        
        Args:
            text_path: Path to text model weights
            audio_path: Path to audio model weights
        """
        if os.path.exists(text_path):
            # For text, we need to create a TextEmotionClassifier and transfer only the RoBERTa weights
            text_classifier = TextEmotionClassifier()
            text_classifier.load_state_dict(torch.load(text_path, map_location=device))
            
            # Transfer RoBERTa weights
            self.text_model.load_state_dict(text_classifier.roberta.state_dict())
            print(f"Loaded pre-trained text model from {text_path}")
        else:
            print(f"Warning: Text model weights not found at {text_path}")
        
        if os.path.exists(audio_path):
            self.audio_model.load_state_dict(torch.load(audio_path, map_location=device))
            print(f"Loaded pre-trained audio model from {audio_path}")
        else:
            print(f"Warning: Audio model weights not found at {audio_path}")
    
    def forward(self, input_ids, attention_mask, audio_features):
        """
        Forward pass through the model
        
        Args:
            input_ids: Token IDs for text [batch_size, seq_len]
            attention_mask: Attention mask for text [batch_size, seq_len]
            audio_features: Audio features [batch_size, feature_dim]
            
        Returns:
            logits: Predicted logits for each class [batch_size, num_classes]
            fusion_weights: Attention weights from fusion [batch_size, 2]
        """
        # Process text through RoBERTa
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_cls = text_outputs.last_hidden_state[:, 0, :]  # CLS token for text representation
        text_embedding = self.text_encoder(text_cls)
        
        # Process audio through audio model
        with torch.no_grad():
            audio_logits = self.audio_model(audio_features)
        audio_embedding = self.audio_encoder(audio_logits)
        
        # Fusion through cross-attention
        fused_embedding, fusion_weights = self.cross_attention(text_embedding, audio_embedding)
        
        # Concatenate the fused embedding with both modalities
        combined_embedding = torch.cat([fused_embedding, audio_embedding], dim=1)
        
        # Classifier head
        logits = self.classifier(combined_embedding)
        
        return logits, fusion_weights


class CrossAttention(nn.Module):
    """
    Cross-attention mechanism for multimodal fusion
    """
    def __init__(self, text_dim, audio_dim):
        super(CrossAttention, self).__init__()
        
        # Attention projections
        self.query_text = nn.Linear(text_dim, text_dim)
        self.key_audio = nn.Linear(audio_dim, text_dim)
        self.value_audio = nn.Linear(audio_dim, text_dim)
        
        # Output projection
        self.output = nn.Linear(text_dim, text_dim)
        
        # Scaling factor
        self.scale = text_dim ** -0.5
        
        # Weights for fusion
        self.fusion_weights = nn.Linear(text_dim + audio_dim, 2)
    
    def forward(self, text_embedding, audio_embedding):
        """
        Forward pass through cross-attention
        
        Args:
            text_embedding: Text embedding [batch_size, text_dim]
            audio_embedding: Audio embedding [batch_size, audio_dim]
            
        Returns:
            fused_embedding: Fused embedding [batch_size, text_dim]
            weights: Fusion weights [batch_size, 2]
        """
        # Project text to query
        query = self.query_text(text_embedding).unsqueeze(1)  # [batch_size, 1, text_dim]
        
        # Project audio to key and value
        key = self.key_audio(audio_embedding).unsqueeze(1)    # [batch_size, 1, text_dim]
        value = self.value_audio(audio_embedding).unsqueeze(1)  # [batch_size, 1, text_dim]
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to value
        context = torch.matmul(attention_weights, value).squeeze(1)  # [batch_size, text_dim]
        
        # Project to output
        attended_audio = self.output(context)
        
        # Compute fusion weights
        combined = torch.cat([text_embedding, audio_embedding], dim=1)
        weights = F.softmax(self.fusion_weights(combined), dim=1)
        
        # Weighted combination
        text_weight = weights[:, 0].unsqueeze(1)
        audio_weight = weights[:, 1].unsqueeze(1)
        
        fused_embedding = text_weight * text_embedding + audio_weight * attended_audio
        
        return fused_embedding, weights


def load_data(data_path=None):
    """
    Load dataset with both text and audio features
    
    Args:
        data_path: Path to data file
        
    Returns:
        DataFrame with text, audio_features and emotions
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
    
    print("No data found, creating mock dataset not supported for multimodal model.")
    print("Please generate data first using the text_emotion.py or audio_emotion.py scripts.")
    return None


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
        'val_acc': [],
        'text_weight': [],
        'audio_weight': []
    }
    
    # Best validation accuracy
    best_val_acc = 0.0
    best_model_path = os.path.join(DEFAULT_MODEL_DIR, "multimodal_emotion_best.pt")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Training
        model.train()
        train_loss = 0.0
        train_corrects = 0
        train_total = 0
        text_weight_sum = 0.0
        audio_weight_sum = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio_features = batch['audio_features'].to(device)
            labels = batch['label'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs, fusion_weights = model(input_ids, attention_mask, audio_features)
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
            
            # Track fusion weights
            text_weight_sum += fusion_weights[:, 0].mean().item()
            audio_weight_sum += fusion_weights[:, 1].mean().item()
            
            # Print progress
            if (batch_idx + 1) % 20 == 0:
                print(f"Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        # Calculate training metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = train_corrects / train_total * 100
        avg_text_weight = text_weight_sum / len(train_loader)
        avg_audio_weight = audio_weight_sum / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0
        val_text_weight_sum = 0.0
        val_audio_weight_sum = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                audio_features = batch['audio_features'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                outputs, fusion_weights = model(input_ids, attention_mask, audio_features)
                loss = criterion(outputs, labels)
                
                # Track statistics
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels).item()
                val_total += labels.size(0)
                
                # Track fusion weights
                val_text_weight_sum += fusion_weights[:, 0].mean().item()
                val_audio_weight_sum += fusion_weights[:, 1].mean().item()
        
        # Calculate validation metrics
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = val_corrects / val_total * 100
        val_avg_text_weight = val_text_weight_sum / len(val_loader)
        val_avg_audio_weight = val_audio_weight_sum / len(val_loader)
        
        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step(epoch_val_loss)
        
        # Save history
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['text_weight'].append(val_avg_text_weight)
        history['audio_weight'].append(val_avg_audio_weight)
        
        # Print epoch statistics
        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")
        print(f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
        print(f"Fusion Weights - Text: {val_avg_text_weight:.4f}, Audio: {val_avg_audio_weight:.4f}")
        
        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with accuracy {best_val_acc:.2f}%")
    
    # Save final model
    final_model_path = os.path.join(DEFAULT_MODEL_DIR, "multimodal_emotion_final.pt")
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
    all_text_weights = []
    all_audio_weights = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Move data to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio_features = batch['audio_features'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs, fusion_weights = model(input_ids, attention_mask, audio_features)
            _, preds = torch.max(outputs, 1)
            
            # Track predictions
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_text_weights.extend(fusion_weights[:, 0].cpu().numpy())
            all_audio_weights.extend(fusion_weights[:, 1].cpu().numpy())
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds) * 100
    
    # Get classification report
    report = classification_report(all_labels, all_preds, 
                                  target_names=EMOTIONS, 
                                  output_dict=True)
    
    # Get confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Calculate average fusion weights
    avg_text_weight = np.mean(all_text_weights)
    avg_audio_weight = np.mean(all_audio_weights)
    
    return {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm,
        'text_weight': avg_text_weight,
        'audio_weight': avg_audio_weight
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
    plt.savefig(os.path.join(DEFAULT_LOG_DIR, 'multimodal_confusion_matrix.png'))
    plt.close()


def plot_training_history(history):
    """
    Plot training history
    
    Args:
        history: Training history dictionary
    """
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # Plot fusion weights
    plt.subplot(1, 3, 3)
    plt.plot(history['text_weight'], label='Text')
    plt.plot(history['audio_weight'], label='Audio')
    plt.title('Fusion Weights')
    plt.xlabel('Epoch')
    plt.ylabel('Weight')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(DEFAULT_LOG_DIR, 'multimodal_training_history.png'))
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
    print("\nMultimodal Emotion Recognition")
    print("=" * 60)
    
    # Load data
    df = load_data(args.data_path)
    if df is None:
        return
    
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
    
    # Initialize tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    
    # Create datasets
    train_dataset = MultimodalEmotionDataset(train_df, tokenizer)
    val_dataset = MultimodalEmotionDataset(val_df, tokenizer)
    test_dataset = MultimodalEmotionDataset(test_df, tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = MultimodalFusionClassifier().to(device)
    
    # Load pre-trained individual models
    text_model_path = os.path.join(DEFAULT_MODEL_DIR, "text_emotion_best.pt")
    audio_model_path = os.path.join(DEFAULT_MODEL_DIR, "audio_emotion_best.pt")
    model.load_pretrained_models(text_model_path, audio_model_path)
    
    # Load pre-trained fusion model if exists and not training
    best_model_path = os.path.join(DEFAULT_MODEL_DIR, "multimodal_emotion_best.pt")
    if os.path.exists(best_model_path) and not args.train:
        print(f"\nLoading pre-trained fusion model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    elif args.evaluate and not os.path.exists(best_model_path):
        print("\nError: No pre-trained fusion model found for evaluation.")
        print(f"Expected model at {best_model_path}")
        print("Please train a model first.")
        return
    
    # Training
    if args.train or not os.path.exists(best_model_path):
        print("\nTraining model...")
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        
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
        print(f"Fusion Weights - Text: {metrics['text_weight']:.4f}, Audio: {metrics['audio_weight']:.4f}")
        print_classification_report(metrics['report'])
        
        # Plot confusion matrix
        plot_confusion_matrix(metrics['confusion_matrix'])
        print(f"Confusion matrix saved to {os.path.join(DEFAULT_LOG_DIR, 'multimodal_confusion_matrix.png')}")
    
    print("\nDone!")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Multimodal Emotion Recognition")
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