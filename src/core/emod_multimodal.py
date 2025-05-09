#!/usr/bin/env python3
"""
EMOD Multimodal - Emotion Recognition System with Audio and Text

This script implements a two-stage multimodal emotion recognition system:
1. Convert text and audio to Valence-Arousal-Dominance (VAD) using fusion
2. Classify emotions based on VAD values

Based on the approach from work.ipynb with additional audio processing capability
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

# Audio processing imports
import librosa
import librosa.display
from scipy.stats import skew, kurtosis

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

# Audio feature extraction parameters
SAMPLE_RATE = 16000  # Hz
FRAME_LENGTH = 0.025  # seconds
FRAME_STEP = 0.01  # seconds
NUM_MFCC = 13  # Number of MFCC coefficients to extract

# Set random seeds
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class MultimodalVADDataset(Dataset):
    """Dataset for multimodal (text + audio) VAD prediction."""
    def __init__(self, texts, audio_features, vad_values, tokenizer, max_len=MAX_SEQ_LENGTH):
        self.texts = texts
        self.audio_features = audio_features
        self.vad_values = vad_values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        audio_feat = self.audio_features[idx]
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
            'audio_features': torch.tensor(audio_feat, dtype=torch.float),
            'vad_values': torch.tensor(vad, dtype=torch.float)
        }

class MultimodalVADModel(nn.Module):
    """Model for predicting VAD values from text and audio."""
    def __init__(self, model_name=MODEL_NAME, audio_feat_dim=88, dropout=0.1, fusion_type='early'):
        super(MultimodalVADModel, self).__init__()
        self.fusion_type = fusion_type
        
        # Text processing
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = self.transformer.config.hidden_size
        
        # Audio processing
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_feat_dim, 256),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.Dropout(dropout),
            nn.GELU(),
        )
        
        # Early fusion architecture
        if fusion_type == 'early':
            # Combine text and audio at feature level
            self.text_projector = nn.Linear(hidden_size, 512)
            self.fusion = nn.Sequential(
                nn.Linear(1024, 512),  # 512 (text) + 512 (audio)
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
        
        # Late fusion architecture
        else:  # fusion_type == 'late'
            # Process text and audio separately, then combine predictions
            self.text_valence = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.LayerNorm(256),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(256, 1)
            )
            self.text_arousal = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.LayerNorm(256),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(256, 1)
            )
            self.text_dominance = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.LayerNorm(256),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(256, 1)
            )
            
            self.audio_valence = nn.Sequential(
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(256, 1)
            )
            self.audio_arousal = nn.Sequential(
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(256, 1)
            )
            self.audio_dominance = nn.Sequential(
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(256, 1)
            )
            
            # Fusion weights (learnable parameters)
            self.fusion_weights = nn.Parameter(torch.ones(3, 2))  # 3 VAD dimensions, 2 modalities
            self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, audio_features):
        # Process text
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        text_hidden = outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding
        
        # Process audio
        audio_hidden = self.audio_encoder(audio_features)
        
        # Early fusion
        if self.fusion_type == 'early':
            text_proj = self.text_projector(text_hidden)
            multimodal_hidden = torch.cat([text_proj, audio_hidden], dim=1)
            fused = self.fusion(multimodal_hidden)
            
            valence = self.valence_branch(fused)
            arousal = self.arousal_branch(fused)
            dominance = self.dominance_branch(fused)
        
        # Late fusion
        else:  # fusion_type == 'late'
            # Text predictions
            text_valence = self.text_valence(text_hidden)
            text_arousal = self.text_arousal(text_hidden)
            text_dominance = self.text_dominance(text_hidden)
            
            # Audio predictions
            audio_valence = self.audio_valence(audio_hidden)
            audio_arousal = self.audio_arousal(audio_hidden)
            audio_dominance = self.audio_dominance(audio_hidden)
            
            # Get fusion weights (softmax to ensure they sum to 1)
            weights = self.softmax(self.fusion_weights)
            
            # Weighted fusion
            valence = weights[0, 0] * text_valence + weights[0, 1] * audio_valence
            arousal = weights[1, 0] * text_arousal + weights[1, 1] * audio_arousal
            dominance = weights[2, 0] * text_dominance + weights[2, 1] * audio_dominance
        
        return torch.cat([valence, arousal, dominance], dim=1)

def extract_audio_features(audio_path, sr=SAMPLE_RATE):
    """Extract acoustic features from audio file."""
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=sr)
        
        # Feature extraction
        # 1. Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=NUM_MFCC)
        mfcc_means = np.mean(mfccs, axis=1)
        mfcc_vars = np.var(mfccs, axis=1)
        
        # 2. Spectral features
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)[0]
        
        # 3. Temporal features
        zero_cross = librosa.feature.zero_crossing_rate(y)[0]
        rms = librosa.feature.rms(y=y)[0]
        
        # 4. Statistics (mean, std, skewness, kurtosis)
        feature_list = [
            np.mean(spec_centroid), np.std(spec_centroid), skew(spec_centroid), kurtosis(spec_centroid),
            np.mean(spec_rolloff), np.std(spec_rolloff), skew(spec_rolloff), kurtosis(spec_rolloff),
            np.mean(spec_contrast), np.std(spec_contrast), skew(spec_contrast), kurtosis(spec_contrast),
            np.mean(zero_cross), np.std(zero_cross), skew(zero_cross), kurtosis(zero_cross),
            np.mean(rms), np.std(rms), skew(rms), kurtosis(rms)
        ]
        
        # 5. Pitch-related features
        pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
        mean_pitches = np.mean(pitches, axis=1)
        mean_magnitudes = np.mean(magnitudes, axis=1)
        pitch_features = [
            np.mean(mean_pitches[mean_pitches > 0]) if np.any(mean_pitches > 0) else 0,
            np.std(mean_pitches[mean_pitches > 0]) if np.any(mean_pitches > 0) else 0,
            np.mean(mean_magnitudes)
        ]
        
        # Combine all features
        all_features = np.concatenate([
            mfcc_means, mfcc_vars,
            feature_list,
            pitch_features
        ])
        
        return all_features
    
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        # Return zero vector as fallback
        return np.zeros(NUM_MFCC * 2 + 20 + 3)  # Match the expected dimension

def preprocess_data(csv_path, audio_base_path=None, use_audio=True):
    """Preprocess the IEMOCAP dataset including audio if specified."""
    print(f"Loading data from {csv_path}...")
    labels_df = pd.read_csv(csv_path)
    
    # Extract relevant columns
    df = labels_df[['Speaker_id', 'Transcript', 'dimension', 'category', 'Audio_Uttrance_Path']].copy()
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
    
    # Combine with transcript
    df_model = pd.concat([df_final[['Transcript', 'Mapped_Emotion', 'Audio_Uttrance_Path']], vad_df], axis=1)
    
    # Extract audio features if needed
    if use_audio and audio_base_path is not None:
        print("Extracting audio features...")
        audio_features = []
        
        for i, path in enumerate(tqdm(df_model['Audio_Uttrance_Path'])):
            # Adjust path if needed
            if not os.path.isabs(path):
                full_path = os.path.join(audio_base_path, path)
            else:
                full_path = path
            
            # Extract features or use fallback
            if os.path.exists(full_path):
                features = extract_audio_features(full_path)
            else:
                print(f"Warning: Audio file not found: {full_path}")
                features = np.zeros(NUM_MFCC * 2 + 20 + 3)  # Match the expected dimension
            
            audio_features.append(features)
        
        # Convert to numpy array
        audio_features = np.vstack(audio_features)
        
        print(f"Processed {len(df_model)} samples with valid labels")
        return df_model, audio_features
    else:
        print(f"Processed {len(df_model)} samples with valid labels (without audio)")
        return df_model, None

def train_multimodal_vad_model(X_train, audio_train, y_train, X_val, audio_val, y_val, 
                              tokenizer, fusion_type='early', num_epochs=NUM_EPOCHS):
    """Train the multimodal (text + audio) VAD model."""
    print(f"Setting up datasets and dataloaders for {fusion_type} fusion...")
    train_dataset = MultimodalVADDataset(X_train, audio_train, y_train, tokenizer)
    val_dataset = MultimodalVADDataset(X_val, audio_val, y_val, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    audio_feat_dim = audio_train.shape[1]
    model = MultimodalVADModel(MODEL_NAME, audio_feat_dim, fusion_type=fusion_type).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio_features = batch['audio_features'].to(device)
            targets = batch['vad_values'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, audio_features)
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
                audio_features = batch['audio_features'].to(device)
                targets = batch['vad_values'].to(device)
                
                outputs = model(input_ids, attention_mask, audio_features)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
    
    return model

def evaluate_multimodal_vad_model(model, X_test, audio_test, y_test, tokenizer):
    """Evaluate the multimodal VAD prediction model."""
    print("Evaluating multimodal VAD prediction model...")
    test_dataset = MultimodalVADDataset(X_test, audio_test, y_test, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio_features = batch['audio_features'].to(device)
            targets = batch['vad_values'].to(device)
            
            outputs = model(input_ids, attention_mask, audio_features)
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
    parser = argparse.ArgumentParser(description="EMOD Multimodal - Two-Stage Emotion Recognition")
    parser.add_argument('--data_path', type=str, default='IEMOCAP_Final.csv',
                        help='Path to the IEMOCAP dataset CSV file')
    parser.add_argument('--audio_base_path', type=str, default=None,
                        help='Base path to audio files (if not in absolute path)')
    parser.add_argument('--output_dir', type=str, default='results_multimodal',
                        help='Directory to save results')
    parser.add_argument('--fusion_type', type=str, default='early', choices=['early', 'late'],
                        help='Type of fusion for multimodal model')
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
    
    # Step 1: Load and preprocess data (including audio)
    df_model, audio_features = preprocess_data(args.data_path, args.audio_base_path)
    
    if audio_features is None:
        print("No audio features extracted. Please provide a valid audio_base_path.")
        return
    
    # Step 2: Prepare data for VAD prediction
    X = df_model['Transcript'].values
    y_vad = df_model[['valence', 'arousal', 'dominance']].values
    y_emotion = df_model['Mapped_Emotion'].values
    
    # Step 3: Split data
    indices = np.arange(len(X))
    train_indices, temp_indices = train_test_split(indices, test_size=0.3, random_state=args.seed)
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=args.seed)
    
    # Text data
    X_train, X_val, X_test = X[train_indices], X[val_indices], X[test_indices]
    
    # Audio features
    audio_train = audio_features[train_indices]
    audio_val = audio_features[val_indices]
    audio_test = audio_features[test_indices]
    
    # VAD and emotion labels
    y_vad_train, y_vad_val, y_vad_test = y_vad[train_indices], y_vad[val_indices], y_vad[test_indices]
    y_emotion_train, y_emotion_val, y_emotion_test = y_emotion[train_indices], y_emotion[val_indices], y_emotion[test_indices]
    
    # Step 4: Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Step 5: Train multimodal VAD prediction model
    vad_model = train_multimodal_vad_model(
        X_train, audio_train, y_vad_train,
        X_val, audio_val, y_vad_val,
        tokenizer, fusion_type=args.fusion_type, num_epochs=args.epochs
    )
    
    # Step 6: Evaluate multimodal VAD prediction model
    vad_preds, vad_targets, vad_metrics = evaluate_multimodal_vad_model(
        vad_model, X_test, audio_test, y_vad_test, tokenizer
    )
    
    # Step 7: Generate VAD predictions for training emotion classifier
    vad_train_preds = []
    with torch.no_grad():
        train_dataset = MultimodalVADDataset(X_train, audio_train, y_vad_train, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
        
        for batch in tqdm(train_loader, desc="Generating VAD predictions for train set"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio_features = batch['audio_features'].to(device)
            outputs = vad_model(input_ids, attention_mask, audio_features)
            vad_train_preds.append(outputs.cpu().numpy())
    
    vad_train_preds = np.vstack(vad_train_preds)
    
    # Step 8: Train emotion classifier
    emotion_model = train_emotion_classifier(vad_train_preds, y_emotion_train)
    
    # Step 9: Evaluate emotion classifier
    emotion_preds, emotion_metrics = evaluate_emotion_classifier(emotion_model, vad_preds, y_emotion_test)
    
    # Step 10: Save results
    data_info = {
        'total_samples': len(df_model),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'emotion_distribution': {emotion: count for emotion, count in zip(*np.unique(y_emotion, return_counts=True))},
        'fusion_type': args.fusion_type
    }
    
    save_results(args.output_dir, vad_metrics, emotion_metrics, data_info)
    
    # Step 11: Save models if requested
    if args.save_model:
        os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
        torch.save(vad_model.state_dict(), os.path.join(args.output_dir, 'models', 'multimodal_vad_model.pt'))
        torch.save(emotion_model, os.path.join(args.output_dir, 'models', 'emotion_model.pt'))
        print("Models saved successfully")

if __name__ == '__main__':
    main() 