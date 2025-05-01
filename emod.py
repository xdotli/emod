#!/usr/bin/env python3
"""
EMOD - Emotion Recognition System

This script implements a two-stage emotion recognition system:
1. Convert text to Valence-Arousal-Dominance (VAD) using a fine-tuned RoBERTa model
2. Classify emotions based on VAD values

Based on the approach from work.ipynb
"""

import os
import argparse
import ast
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB

from transformers import AutoModel, AutoTokenizer

# Constants
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
MAX_SEQ_LENGTH = 128
MODEL_NAME = "roberta-base"
RANDOM_SEED = 42
EMOTION_MAPPING = {
    'neutral': 'neutral',
    'frustration': 'angry',
    'anger': 'angry',
    'surprise': None,
    'disgust': None,
    'other': None,
    'sadness': 'sad',
    'fear': None,
    'happiness': 'happy',
    'excited': 'happy'
}

# Set random seeds
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Add a custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

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

def preprocess_data(csv_path):
    """Preprocess the IEMOCAP dataset."""
    print(f"Loading data from {csv_path}...")
    labels_df = pd.read_csv(csv_path)
    
    # Extract relevant columns
    df = labels_df[['Speaker_id', 'Transcript', 'dimension', 'category']].copy()
    df['category'] = df['category'].astype(str).str.lower()
    df = df.drop_duplicates(subset='Transcript')
    
    # Get majority emotion label
    def get_majority_label(label_str):
        try:
            labels = ast.literal_eval(label_str)
            count = Counter(labels)
            most_common = count.most_common(1)[0]
            if most_common[1] >= 2:
                return most_common[0]
            else:
                return None
        except:
            return None
    
    # Create emotion column and filter rows
    df['Emotion'] = df['category'].apply(get_majority_label)
    df_filtered = df.dropna(subset=['Emotion'])
    
    # Map to simplified emotion categories
    df_filtered['Mapped_Emotion'] = df_filtered['Emotion'].map(EMOTION_MAPPING)
    df_final = df_filtered.dropna(subset=['Mapped_Emotion'])
    
    # Parse dimension column
    def smart_parse(x):
        try:
            if pd.isna(x):
                return None
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
                return parsed[0]
            elif isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
    
    df_final['parsed_dim'] = df_final['dimension'].apply(smart_parse)
    
    # Flatten VAD scores and prepare final dataframe
    vad_df = pd.json_normalize(df_final['parsed_dim'])
    df_final = df_final.reset_index(drop=True)
    vad_df = vad_df.reset_index(drop=True)
    
    # Combine with transcript and return data
    df_model = pd.concat([df_final[['Transcript', 'Mapped_Emotion']], vad_df], axis=1)
    
    print(f"Processed {len(df_model)} samples with valid labels")
    return df_model

def train_vad_model(X_train, y_train, X_val, y_val, tokenizer, num_epochs=NUM_EPOCHS):
    """Train the Text to VAD model."""
    print("Setting up datasets and dataloaders...")
    train_dataset = TextVADDataset(X_train, y_train, tokenizer)
    val_dataset = TextVADDataset(X_val, y_val, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = TextVADModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['vad_values'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['vad_values'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
    
    return model

def evaluate_vad_model(model, X_test, y_test, tokenizer):
    """Evaluate the VAD prediction model."""
    print("Evaluating VAD prediction model...")
    test_dataset = TextVADDataset(X_test, y_test, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['vad_values'].to(device)
            
            outputs = model(input_ids, attention_mask)
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
    
    return preds, targets, {
        'mse': mse.tolist(),
        'rmse': rmse.tolist(),
        'mae': mae.tolist(),
        'r2': r2.tolist()
    }

def train_emotion_classifier(X_train, y_train):
    """Train the VAD to emotion classifier."""
    print("Training ensemble classifier (VAD to emotion)...")
    # Initialize models for ensemble
    nb = GaussianNB()
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    
    # Voting classifier (soft = average class probabilities)
    ensemble = VotingClassifier(
        estimators=[('nb', nb), ('rf', rf)],
        voting='soft'
    )
    
    # Train ensemble
    ensemble.fit(X_train, y_train)
    return ensemble

def evaluate_emotion_classifier(model, X_test, y_test):
    """Evaluate the emotion classifier."""
    print("Evaluating emotion classifier...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Weighted F1 Score: {f1_weighted:.4f}")
    print(f"Macro F1 Score: {f1_macro:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    return y_pred, {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'classification_report': classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    }

def save_results(output_dir, vad_metrics, emotion_metrics, data_info=None):
    """Save results and metrics to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    metrics = {
        'vad_metrics': vad_metrics,
        'emotion_metrics': emotion_metrics,
        'data_info': data_info
    }
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)
    
    print(f"Results saved to {output_dir}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="EMOD - Two-Stage Emotion Recognition")
    parser.add_argument('--data_path', type=str, default='IEMOCAP_Final.csv',
                        help='Path to the IEMOCAP dataset CSV file')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs for training the VAD model')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_model', action='store_true',
                        help='Save the trained models')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Load and preprocess data
    df_model = preprocess_data(args.data_path)
    
    # Step 2: Prepare data for VAD prediction
    X = df_model['Transcript'].values
    y_vad = df_model[['valence', 'arousal', 'dominance']].values
    y_emotion = df_model['Mapped_Emotion'].values
    
    # Split data into train, validation, and test sets
    X_train, X_temp, y_vad_train, y_vad_temp, y_emotion_train, y_emotion_temp = train_test_split(
        X, y_vad, y_emotion, test_size=0.3, random_state=args.seed)
    
    X_val, X_test, y_vad_val, y_vad_test, y_emotion_val, y_emotion_test = train_test_split(
        X_temp, y_vad_temp, y_emotion_temp, test_size=0.5, random_state=args.seed)
    
    # Step 3: Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Step 4: Train VAD prediction model
    vad_model = train_vad_model(X_train, y_vad_train, X_val, y_vad_val, tokenizer, num_epochs=args.epochs)
    
    # Step 5: Evaluate VAD prediction model
    vad_preds, vad_targets, vad_metrics = evaluate_vad_model(vad_model, X_test, y_vad_test, tokenizer)
    
    # Step 6: Train emotion classifier using predicted VAD values
    vad_train_preds = []
    with torch.no_grad():
        train_dataset = TextVADDataset(X_train, y_vad_train, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
        
        for batch in tqdm(train_loader, desc="Generating VAD predictions for train set"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = vad_model(input_ids, attention_mask)
            vad_train_preds.append(outputs.cpu().numpy())
    
    vad_train_preds = np.vstack(vad_train_preds)
    
    # Step 7: Train emotion classifier
    emotion_model = train_emotion_classifier(vad_train_preds, y_emotion_train)
    
    # Step 8: Evaluate emotion classifier
    emotion_preds, emotion_metrics = evaluate_emotion_classifier(emotion_model, vad_preds, y_emotion_test)
    
    # Step 9: Save results
    data_info = {
        'total_samples': len(df_model),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'emotion_distribution': {emotion: count for emotion, count in zip(*np.unique(y_emotion, return_counts=True))},
    }
    
    save_results(args.output_dir, vad_metrics, emotion_metrics, data_info)
    
    # Step 10: Save models if requested
    if args.save_model:
        os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
        torch.save(vad_model.state_dict(), os.path.join(args.output_dir, 'models', 'vad_model.pt'))
        torch.save(emotion_model, os.path.join(args.output_dir, 'models', 'emotion_model.pt'))
        print("Models saved successfully")

if __name__ == '__main__':
    main() 