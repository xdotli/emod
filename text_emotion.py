#!/usr/bin/env python3
"""
Text-based Emotion Recognition

This module implements a text-based emotion recognition model using RoBERTa,
with direct classification of text to emotion categories.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from tabulate import tabulate
import argparse
import json
import time
from pathlib import Path

# Constants
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
MAX_SEQ_LENGTH = 128
MODEL_NAME = "roberta-base"
RANDOM_SEED = 42
EMOTIONS = ["angry", "happy", "neutral", "sad"]

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


class TextEmotionDataset(Dataset):
    """Dataset for text-based emotion recognition"""
    
    def __init__(self, texts, emotions, tokenizer, max_len=MAX_SEQ_LENGTH):
        self.texts = texts
        self.emotions = emotions
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.emotion_to_id = {emotion: i for i, emotion in enumerate(EMOTIONS)}
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
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
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class TextEmotionClassifier(nn.Module):
    """Text emotion classifier using RoBERTa"""
    
    def __init__(self, num_classes=len(EMOTIONS), dropout=0.1):
        super(TextEmotionClassifier, self).__init__()
        
        # Load RoBERTa
        self.roberta = RobertaModel.from_pretrained(MODEL_NAME)
        hidden_size = self.roberta.config.hidden_size
        
        # Freeze embeddings for faster training and to prevent overfitting
        for param in self.roberta.embeddings.parameters():
            param.requires_grad = False
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        """Forward pass"""
        # Get RoBERTa output
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use the [CLS] token representation for classification
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Pass through classifier
        logits = self.classifier(pooled_output)
        
        return logits


def load_data(data_path=None):
    """
    Load dataset. If no path provided, create mock data.
    
    Args:
        data_path: Path to CSV file with data
        
    Returns:
        DataFrame with texts and emotions
    """
    if data_path and os.path.exists(data_path):
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        return df
    
    # Check if we have processed data already
    processed_path = os.path.join(DEFAULT_DATA_DIR, "processed_data.csv")
    if os.path.exists(processed_path):
        print(f"Loading existing data from {processed_path}")
        df = pd.read_csv(processed_path)
        return df
    
    print("No data found, creating mock dataset")
    
    # Create mock data
    emotion_texts = {
        'happy': [
            "I'm so excited about the weekend!",
            "That was the best movie I've seen all year.",
            "I got the job! I'm so happy!",
            "This is wonderful news!",
            "I can't wait to see you again!"
        ],
        'sad': [
            "I miss you so much.",
            "I didn't get the promotion I was hoping for.",
            "I'm feeling really down today.",
            "It's been a tough week.",
            "I'm sorry for your loss."
        ],
        'angry': [
            "I can't believe you did that!",
            "This is completely unacceptable.",
            "I'm furious about what happened.",
            "You always do this!",
            "I'm tired of being treated this way."
        ],
        'neutral': [
            "I'm going to the store.",
            "The meeting is at 2pm.",
            "Please pass me that document.",
            "I'll call you later.",
            "The weather is cloudy today."
        ]
    }
    
    data = []
    for emotion, texts in emotion_texts.items():
        # Create 250 samples for each emotion by repeating the texts
        for _ in range(50):
            for text in texts:
                data.append({
                    'text': text,
                    'emotion': emotion
                })
    
    df = pd.DataFrame(data)
    
    # Save processed data
    df.to_csv(processed_path, index=False)
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
    best_model_path = os.path.join(DEFAULT_MODEL_DIR, "text_emotion_best.pt")
    
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
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            
            # Track statistics
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_corrects += torch.sum(preds == labels).item()
            train_total += labels.size(0)
            
            # Print progress
            if (batch_idx + 1) % 20 == 0:
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
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                # Track statistics
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels).item()
                val_total += labels.size(0)
        
        # Calculate validation metrics
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = val_corrects / val_total * 100
        
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
    final_model_path = os.path.join(DEFAULT_MODEL_DIR, "text_emotion_final.pt")
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
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
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
    plt.savefig(os.path.join(DEFAULT_LOG_DIR, 'confusion_matrix.png'))
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
    plt.savefig(os.path.join(DEFAULT_LOG_DIR, 'training_history.png'))
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
    print("\nText-based Emotion Recognition")
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
    
    # Initialize tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    
    # Create datasets
    train_dataset = TextEmotionDataset(train_df['text'].values, train_df['emotion'].values, tokenizer)
    val_dataset = TextEmotionDataset(val_df['text'].values, val_df['emotion'].values, tokenizer)
    test_dataset = TextEmotionDataset(test_df['text'].values, test_df['emotion'].values, tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = TextEmotionClassifier().to(device)
    
    # Load pre-trained model if exists and not training
    best_model_path = os.path.join(DEFAULT_MODEL_DIR, "text_emotion_best.pt")
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
        
        # Initialize optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
        
        # Calculate total training steps
        num_training_steps = len(train_loader) * NUM_EPOCHS
        
        # Create scheduler with warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(0.1 * num_training_steps),
            num_training_steps=num_training_steps
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
        print(f"Confusion matrix saved to {os.path.join(DEFAULT_LOG_DIR, 'confusion_matrix.png')}")
    
    print("\nDone!")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Text-based Emotion Recognition")
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