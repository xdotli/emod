#!/usr/bin/env python3
"""
VAD to Emotion Pipeline

This script implements the full pipeline for emotion recognition from text:
1. Text -> VAD values using a fine-tuned transformer model
2. VAD values -> Emotion category using either rule-based mapping or ML classifier
"""

import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
import argparse
from pathlib import Path
from text_vad import TextVADModel, TextVADDataset, vad_to_emotion
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import time

# Constants
BATCH_SIZE = 16
RANDOM_SEED = 42
EMOTIONS = ["angry", "happy", "neutral", "sad"]
MODEL_NAME = "roberta-base"

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


class VADtoEmotionClassifier:
    """Classifier for mapping VAD values to emotion categories"""
    
    def __init__(self, classifier_type="rf"):
        """
        Initialize the classifier
        
        Args:
            classifier_type: Type of classifier to use ("rf" for RandomForest, 
                            "svm" for SupportVectorMachine)
        """
        self.classifier_type = classifier_type
        
        if classifier_type == "rf":
            self.classifier = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=RANDOM_SEED
            )
        elif classifier_type == "svm":
            self.classifier = SVC(
                kernel='rbf', 
                probability=True,
                random_state=RANDOM_SEED
            )
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")
    
    def fit(self, vad_values, emotion_labels):
        """
        Train the classifier on VAD values and emotion labels
        
        Args:
            vad_values: VAD values as numpy array of shape (n_samples, 3)
            emotion_labels: Emotion labels as numpy array of shape (n_samples,)
        """
        print(f"Training {self.classifier_type.upper()} classifier...")
        self.classifier.fit(vad_values, emotion_labels)
    
    def predict(self, vad_values):
        """
        Predict emotion categories from VAD values
        
        Args:
            vad_values: VAD values as numpy array of shape (n_samples, 3)
            
        Returns:
            Predicted emotion labels
        """
        return self.classifier.predict(vad_values)
    
    def predict_proba(self, vad_values):
        """
        Predict emotion probabilities from VAD values
        
        Args:
            vad_values: VAD values as numpy array of shape (n_samples, 3)
            
        Returns:
            Predicted emotion probabilities
        """
        return self.classifier.predict_proba(vad_values)
    
    def save(self, path):
        """
        Save the classifier to disk
        
        Args:
            path: Path to save the classifier
        """
        with open(path, 'wb') as f:
            pickle.dump(self.classifier, f)
        print(f"Saved classifier to {path}")
    
    def load(self, path):
        """
        Load the classifier from disk
        
        Args:
            path: Path to load the classifier from
        """
        with open(path, 'rb') as f:
            self.classifier = pickle.load(f)
        print(f"Loaded classifier from {path}")


def load_data(data_path=None):
    """
    Load IEMOCAP data with emotions and VAD values
    
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
    processed_path = os.path.join(DEFAULT_DATA_DIR, "iemocap_processed.csv")
    if os.path.exists(processed_path):
        print(f"Loading existing data from {processed_path}")
        df = pd.read_csv(processed_path)
        return df
    
    print("No data found. Please provide a path to the IEMOCAP data.")
    # Create a small mock dataset for testing
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
    
    return df


def evaluate_vad_predictions(true_vad, pred_vad):
    """
    Evaluate VAD predictions
    
    Args:
        true_vad: True VAD values
        pred_vad: Predicted VAD values
        
    Returns:
        Dictionary of evaluation metrics
    """
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    
    # Overall metrics
    mse = mean_squared_error(true_vad, pred_vad)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_vad, pred_vad)
    r2 = r2_score(true_vad, pred_vad)
    
    # Per-dimension metrics
    val_mse = mean_squared_error(true_vad[:, 0], pred_vad[:, 0])
    aro_mse = mean_squared_error(true_vad[:, 1], pred_vad[:, 1])
    dom_mse = mean_squared_error(true_vad[:, 2], pred_vad[:, 2])
    
    val_r2 = r2_score(true_vad[:, 0], pred_vad[:, 0])
    aro_r2 = r2_score(true_vad[:, 1], pred_vad[:, 1])
    dom_r2 = r2_score(true_vad[:, 2], pred_vad[:, 2])
    
    # Print metrics
    print("\nVAD Prediction Metrics:")
    print("-" * 40)
    print(f"Overall MSE: {mse:.4f}")
    print(f"Overall RMSE: {rmse:.4f}")
    print(f"Overall MAE: {mae:.4f}")
    print(f"Overall R²: {r2:.4f}")
    
    print("\nPer-Dimension Metrics:")
    print(f"Valence - MSE: {val_mse:.4f}, R²: {val_r2:.4f}")
    print(f"Arousal - MSE: {aro_mse:.4f}, R²: {aro_r2:.4f}")
    print(f"Dominance - MSE: {dom_mse:.4f}, R²: {dom_r2:.4f}")
    
    # Return metrics dictionary
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'valence_mse': val_mse,
        'arousal_mse': aro_mse,
        'dominance_mse': dom_mse,
        'valence_r2': val_r2,
        'arousal_r2': aro_r2,
        'dominance_r2': dom_r2
    }


def evaluate_emotion_predictions(true_emotions, pred_emotions):
    """
    Evaluate emotion predictions
    
    Args:
        true_emotions: True emotion labels
        pred_emotions: Predicted emotion labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Calculate accuracy
    accuracy = accuracy_score(true_emotions, pred_emotions)
    
    # Generate classification report
    class_report = classification_report(
        true_emotions, 
        pred_emotions, 
        output_dict=True
    )
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(true_emotions, pred_emotions)
    
    # Print metrics
    print("\nEmotion Classification Metrics:")
    print("-" * 40)
    print(f"Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(true_emotions, pred_emotions))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=sorted(set(true_emotions)),
        yticklabels=sorted(set(true_emotions))
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(DEFAULT_LOG_DIR, 'confusion_matrix.png'))
    
    # Return metrics dictionary
    return {
        'accuracy': accuracy,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix.tolist()
    }


def predict_vad_from_text(model, texts, tokenizer, batch_size=BATCH_SIZE):
    """
    Predict VAD values from text using the pre-trained model
    
    Args:
        model: Trained TextVADModel
        texts: List of text strings
        tokenizer: Tokenizer for the model
        batch_size: Batch size for predictions
        
    Returns:
        Numpy array of VAD predictions
    """
    # Create a simple dataset without labels
    class SimpleTextDataset(torch.utils.data.Dataset):
        def __init__(self, texts, tokenizer, max_len=128):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_len = max_len
            
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = str(self.texts[idx])
            
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
                'attention_mask': encoding['attention_mask'].flatten()
            }
    
    # Create dataset and dataloader
    dataset = SimpleTextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    # Set model to evaluation mode
    model.eval()
    
    # Predict in batches
    all_predictions = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Get batch data
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            
            # Save predictions
            all_predictions.append(outputs.cpu().numpy())
    
    # Concatenate predictions
    all_predictions = np.vstack(all_predictions)
    
    return all_predictions


def end_to_end_pipeline(args):
    """
    Run the full text -> VAD -> emotion pipeline
    
    Args:
        args: Command-line arguments
    """
    print("\n=== Text to Emotion Pipeline via VAD ===\n")
    
    # Step 1: Load the data
    print("Loading data...")
    df = load_data(args.data_path)
    
    # Filter to keep only the target emotions
    if 'emotion' in df.columns:
        df = df[df['emotion'].isin(EMOTIONS)]
    
    # Ensure VAD values are in the [-1, 1] range
    if df['valence'].max() > 1 or df['arousal'].max() > 1 or df['dominance'].max() > 1:
        print("Normalizing VAD values to [-1, 1] range...")
        if df['valence'].max() > 5:  # Assume 1-5 scale
            # Convert from 1-5 scale to [-1, 1] scale
            df['valence'] = (df['valence'] - 3) / 2
            df['arousal'] = (df['arousal'] - 3) / 2
            df['dominance'] = (df['dominance'] - 3) / 2
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
    
    # Step 2: Load or train the text-to-VAD model
    print("\nPreparing text-to-VAD model...")
    tokenizer = AutoTokenizer.from_pretrained(args.vad_model_name)
    vad_model = TextVADModel(model_name=args.vad_model_name).to(device)
    
    if args.vad_model_path and os.path.exists(args.vad_model_path):
        print(f"Loading pre-trained VAD model from {args.vad_model_path}")
        checkpoint = torch.load(args.vad_model_path, map_location=device)
        vad_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("No pre-trained VAD model provided. You should train one first using text_vad.py")
        return
    
    # Step 3: Train VAD-to-emotion classifier if using ML approach
    if args.use_ml_classifier:
        print("\nPreparing VAD-to-emotion classifier...")
        vad_to_emotion_classifier = VADtoEmotionClassifier(classifier_type=args.classifier_type)
        
        if args.classifier_path and os.path.exists(args.classifier_path):
            print(f"Loading pre-trained classifier from {args.classifier_path}")
            vad_to_emotion_classifier.load(args.classifier_path)
        else:
            print("Training VAD-to-emotion classifier...")
            train_vad = train_df[['valence', 'arousal', 'dominance']].values
            train_emotions = train_df['emotion'].values
            
            vad_to_emotion_classifier.fit(train_vad, train_emotions)
            
            # Save the trained classifier
            classifier_path = os.path.join(DEFAULT_MODEL_DIR, f"vad_to_emotion_{args.classifier_type}.pkl")
            vad_to_emotion_classifier.save(classifier_path)
    
    # Step 4: Evaluate on test set
    print("\nEvaluating the pipeline on test set...")
    test_texts = test_df['text'].values
    test_vad = test_df[['valence', 'arousal', 'dominance']].values
    if 'emotion' in test_df.columns:
        test_emotions = test_df['emotion'].values
    
    # Predict VAD values from text
    print("Predicting VAD values from text...")
    pred_vad = predict_vad_from_text(vad_model, test_texts, tokenizer)
    
    # Evaluate VAD predictions
    print("\nEvaluating VAD predictions...")
    vad_metrics = evaluate_vad_predictions(test_vad, pred_vad)
    
    # Predict emotions from VAD values
    print("\nPredicting emotions from VAD values...")
    if args.use_ml_classifier:
        pred_emotions = vad_to_emotion_classifier.predict(pred_vad)
    else:
        # Use rule-based approach
        pred_emotions = []
        for i in range(len(pred_vad)):
            v, a, d = pred_vad[i]
            emotion = vad_to_emotion(v, a, d)
            pred_emotions.append(emotion)
    
    # Evaluate emotion predictions if we have ground truth
    if 'emotion' in test_df.columns:
        print("\nEvaluating emotion predictions...")
        emotion_metrics = evaluate_emotion_predictions(test_emotions, pred_emotions)
    
    # Save results
    print("\nSaving results...")
    results = {
        'vad_metrics': vad_metrics,
        'emotion_metrics': emotion_metrics if 'emotion' in test_df.columns else None,
        'examples': []
    }
    
    # Add some examples to results
    for i in range(min(10, len(test_texts))):
        example = {
            'text': test_texts[i],
            'true_vad': test_vad[i].tolist(),
            'pred_vad': pred_vad[i].tolist(),
            'pred_emotion': pred_emotions[i]
        }
        if 'emotion' in test_df.columns:
            example['true_emotion'] = test_emotions[i]
        
        results['examples'].append(example)
    
    # Save results to file
    results_path = os.path.join(DEFAULT_LOG_DIR, 'pipeline_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_path}")
    
    print("\n=== Pipeline evaluation complete ===")


def main():
    parser = argparse.ArgumentParser(description="Text to Emotion Pipeline via VAD")
    parser.add_argument('--data_path', type=str, default=None, help='Path to data file')
    parser.add_argument('--vad_model_name', type=str, default=MODEL_NAME, help='Pre-trained model name')
    parser.add_argument('--vad_model_path', type=str, default=None, help='Path to trained VAD model')
    parser.add_argument('--use_ml_classifier', action='store_true', help='Use ML classifier for VAD to emotion')
    parser.add_argument('--classifier_type', type=str, default='rf', choices=['rf', 'svm'], help='Type of classifier')
    parser.add_argument('--classifier_path', type=str, default=None, help='Path to trained classifier')
    
    args = parser.parse_args()
    
    # Run the pipeline
    end_to_end_pipeline(args)


if __name__ == "__main__":
    main() 