#!/usr/bin/env python3
"""
IEMOCAP Emotion Recognition with Two-Step Approach (Modality -> VAD -> Emotion)

This script implements a multimodal emotion recognition system using the IEMOCAP dataset.
The approach follows two steps:
1. Convert audio and text modalities to VAD (valence-arousal-dominance) tuples
2. Map VAD tuples to emotion categories

Author: AI Assistant
Date: March 2024
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tabulate import tabulate
from process_vad import vad_to_emotion
import random
import time
import platform
import json
from pathlib import Path

# Import models from the models package
from models import AudioVADModel, TextVADModel, FusionVADModel, get_audio_feature_dim

# Constants and configuration
IEMOCAP_PATH = "Datasets/IEMOCAP_full_release"
TEXT_MODEL = "roberta-base"  # Default text model
BATCH_SIZE = 32
LEARNING_RATE = 5e-5  # Lower learning rate for better stability
RANDOM_SEED = 42
AUDIO_FEATURE_DIM = get_audio_feature_dim()

# Default paths
DEFAULT_CHECKPOINT_DIR = "checkpoints"
DEFAULT_DATA_DIR = "data"
DEFAULT_LOGS_DIR = "logs"

# Create necessary directories
os.makedirs(DEFAULT_CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)
os.makedirs(DEFAULT_LOGS_DIR, exist_ok=True)

# Setup device - use MPS if available on Mac (M1/M2), otherwise use CUDA if available, otherwise CPU
if torch.backends.mps.is_available() and platform.system() == "Darwin" and platform.processor() == "arm":
    DEVICE = torch.device("mps")
    print("Using MPS (Metal Performance Shaders) for GPU acceleration on Apple Silicon")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA for GPU acceleration")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU for computation (no GPU acceleration available)")

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Model configuration
CONFIG = {
    "num_epochs": 10,  # Increased for better convergence
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "audio_model": {
        "use_wav2vec": False,  # Set to True to use Wav2Vec 2.0 (requires raw audio)
    },
    "text_model": {
        "model_name": TEXT_MODEL,
        "finetune": True,
    },
    "vad_mapping": {
        "use_ml": False,  # Will be set by run.py if available
    }
}

# Check if vad_to_emotion_model.py exists and import get_vad_to_emotion_predictor if it does
try:
    from vad_to_emotion_model import get_vad_to_emotion_predictor
    use_ml_vad_to_emotion = True
except ImportError:
    use_ml_vad_to_emotion = False

# Model tracking file
MODEL_TRACKER_FILE = "model_tracker.txt"

def generate_run_id():
    """Generate a unique run ID for model checkpoints"""
    timestamp = int(time.time())
    random_suffix = random.randint(1000, 9999)
    return f"{timestamp}_{random_suffix}"

def get_model_paths(checkpoint_dir=DEFAULT_CHECKPOINT_DIR, run_id=None):
    """
    Get the paths for model checkpoints.
    
    Args:
        checkpoint_dir: Directory to save/load checkpoints
        run_id: Optional run ID for new checkpoints
        
    Returns:
        Dictionary of model paths
    """
    # If no run_id provided, use latest or create new one
    if run_id is None:
        run_id = generate_run_id()
    
    return {
        'audio_model_path': os.path.join(checkpoint_dir, f"audio_vad_model_{run_id}.pt"),
        'text_model_path': os.path.join(checkpoint_dir, f"text_vad_model_{run_id}.pt"),
        'fusion_model_path': os.path.join(checkpoint_dir, f"fusion_vad_model_{run_id}.pt"),
        'vad_to_emotion_model_path': os.path.join(checkpoint_dir, "vad_to_emotion_model.pkl")
    }

def get_latest_model_paths(checkpoint_dir=DEFAULT_CHECKPOINT_DIR):
    """
    Find the latest model checkpoints in the checkpoint directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Dictionary of model paths or None if no checkpoints found
    """
    model_paths = {
        'audio_model_path': None,
        'text_model_path': None,
        'fusion_model_path': None,
        'vad_to_emotion_model_path': None
    }
    
    # Check for vad_to_emotion_model
    vad_model_path = os.path.join(checkpoint_dir, "vad_to_emotion_model.pkl")
    if os.path.exists(vad_model_path):
        model_paths['vad_to_emotion_model_path'] = vad_model_path
    
    # Find the latest run ID by looking at audio model files
    audio_models = list(Path(checkpoint_dir).glob("audio_vad_model_*.pt"))
    if not audio_models:
        # No checkpoints found with run IDs, check for default names
        default_audio = os.path.join(checkpoint_dir, "audio_vad_model.pt")
        default_text = os.path.join(checkpoint_dir, "text_vad_model.pt")
        default_fusion = os.path.join(checkpoint_dir, "fusion_vad_model.pt")
        
        if all(os.path.exists(p) for p in [default_audio, default_text, default_fusion]):
            model_paths['audio_model_path'] = default_audio
            model_paths['text_model_path'] = default_text
            model_paths['fusion_model_path'] = default_fusion
            return model_paths
        return None
    
    # Sort by modification time (newest first)
    latest_audio = sorted(audio_models, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    run_id = latest_audio.stem.split('_')[-1]  # Extract run ID from filename
    
    # Construct paths for all models
    model_paths['audio_model_path'] = str(latest_audio)
    model_paths['text_model_path'] = os.path.join(checkpoint_dir, f"text_vad_model_{run_id}.pt")
    model_paths['fusion_model_path'] = os.path.join(checkpoint_dir, f"fusion_vad_model_{run_id}.pt")
    
    # Verify that all models exist
    if not all(os.path.exists(model_paths[key]) for key in ['audio_model_path', 'text_model_path', 'fusion_model_path']):
        return None
    
    return model_paths

def save_model_info(model_paths, performance_metrics, config, checkpoint_dir=DEFAULT_CHECKPOINT_DIR):
    """
    Save model information and metrics to a JSON file.
    
    Args:
        model_paths: Dictionary of model paths
        performance_metrics: Dictionary of performance metrics
        config: Model configuration dictionary
        checkpoint_dir: Directory to save the info file
    """
    # Create info object
    info = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_paths": {k: os.path.basename(v) if v else None for k, v in model_paths.items()},
        "performance": {},
        "config": config
    }
    
    # Convert numpy types and tensors to Python native types
    for k, v in performance_metrics.items():
        if isinstance(v, (np.integer, np.floating, np.bool_)):
            info["performance"][k] = v.item()  # Convert numpy scalars to Python scalars
        elif isinstance(v, np.ndarray):
            info["performance"][k] = v.tolist()  # Convert numpy arrays to lists
        elif hasattr(v, 'item'):  # For PyTorch tensors
            info["performance"][k] = v.item()
        else:
            info["performance"][k] = v
    
    # Extract run_id from audio model path
    if model_paths['audio_model_path']:
        run_id = os.path.basename(model_paths['audio_model_path']).split('_')[-1].split('.')[0]
        info_path = os.path.join(checkpoint_dir, f"model_info_{run_id}.json")
    else:
        info_path = os.path.join(checkpoint_dir, f"model_info_{int(time.time())}.json")
    
    # Save as JSON
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)
    
    print(f"Model info saved to {info_path}")
    
    # Also update a "latest.json" file
    latest_path = os.path.join(checkpoint_dir, "latest.json")
    with open(latest_path, 'w') as f:
        json.dump(info, f, indent=4)
    
    print(f"Latest model info updated in {latest_path}")

class IEMOCAPDataset(Dataset):
    """
    IEMOCAP dataset for multimodal emotion recognition.
    
    Prepares data for both audio and text modalities with improved preprocessing.
    """
    def __init__(self, data, tokenizer=None, max_len=128, use_wav2vec=False):
        self.data = data
        self.use_wav2vec = use_wav2vec
        
        # Initialize tokenizer if not provided
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
        else:
            self.tokenizer = tokenizer
            
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['text']
        
        # Prepare audio features
        if self.use_wav2vec:
            # For Wav2Vec, this would be the raw audio waveform
            audio_features = torch.tensor(row['audio_waveform'], dtype=torch.float)
        else:
            # Traditional audio features
            audio_features = torch.tensor(row['audio_features'], dtype=torch.float)
        
        # Prepare text features
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Get VAD values
        valence = row['valence']
        arousal = row['arousal']
        dominance = row['dominance']
        vad_values = torch.tensor([valence, arousal, dominance], dtype=torch.float)
        
        # Get emotion label
        emotion = row['emotion']
        
        return {
            'audio_features': audio_features,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'vad_values': vad_values,
            'emotion': emotion,
            'text': text,
            'utterance_id': row['utterance_id']
        }

def create_mock_dataset(n_samples=500):
    """
    Create a mock dataset for demonstration purposes.
    
    Parameters:
    - n_samples: Number of samples to generate
    
    Returns:
    - DataFrame with mock data
    """
    print("Creating mock dataset for demonstration purposes...")
    
    # Define emotion labels and their corresponding VAD values
    emotion_to_vad = {
        'happy': [0.8, 0.7, 0.6],   # High valence, arousal, dominance
        'sad': [-0.8, -0.5, -0.6],  # Low valence, arousal, dominance
        'angry': [-0.6, 0.8, 0.7],  # Low valence, high arousal, dominance
        'neutral': [0.0, 0.0, 0.0]  # Neutral
    }
    
    # Sample texts for each emotion
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
    emotions = list(emotion_to_vad.keys())
    
    for i in range(n_samples):
        # Randomly select an emotion
        emotion = np.random.choice(emotions)
        
        # Get VAD values with some noise
        base_vad = emotion_to_vad[emotion]
        noise = np.random.normal(0, 0.1, 3)  # Small random noise
        vad_values = np.clip(base_vad + noise, -1, 1)
        
        # Randomly select a text for this emotion
        text = np.random.choice(emotion_texts[emotion])
        
        # Create mock audio features (random for demonstration)
        audio_features = np.random.randn(AUDIO_FEATURE_DIM).tolist()
        
        data.append({
            'utterance_id': f"mock_{i}",
            'text': text,
            'audio_features': audio_features,
            'valence': vad_values[0],
            'arousal': vad_values[1],
            'dominance': vad_values[2],
            'emotion': emotion
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    return df

def parse_iemocap_emotions():
    """
    Parse emotion labels from IEMOCAP dataset if available.
    
    Returns:
    - DataFrame with parsed data, or None if parsing fails
    """
    try:
        all_data = []
        
        for session in range(1, 6):  # Sessions 1-5
            session_path = os.path.join(IEMOCAP_PATH, f"Session{session}")
            eval_path = os.path.join(session_path, "dialog", "EmoEvaluation")
            
            if not os.path.exists(eval_path):
                print(f"Path {eval_path} does not exist, skipping...")
                continue
                
            # Process each dialog evaluation file
            for filename in os.listdir(eval_path):
                if not filename.endswith(".txt") or not filename.startswith("Ses"):
                    continue
                    
                file_path = os.path.join(eval_path, filename)
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    if line.startswith("[") and "\t" in line:
                        parts = line.strip().split("\t")
                        if len(parts) >= 3:
                            utterance_id = parts[0].strip("[]").split()[0]
                            emotion = parts[2]
                            
                            # Extract VAD values if available
                            vad_values = [0, 0, 0]  # Default
                            if len(parts) >= 4 and "[" in parts[3] and "]" in parts[3]:
                                try:
                                    vad_str = parts[3].strip("[]")
                                    vad_values = [float(x.strip()) for x in vad_str.split(",")]
                                    # Normalize to [-1, 1] from [1, 5] scale
                                    vad_values = [(v - 3) / 2 for v in vad_values]
                                except:
                                    pass
                            
                            # Map emotion codes to standardized emotions
                            if emotion in ["neu", "neutral"]:
                                std_emotion = "neutral"
                            elif emotion in ["ang", "angry"]:
                                std_emotion = "angry"
                            elif emotion in ["hap", "happy", "exc", "excited"]:
                                std_emotion = "happy"
                            elif emotion in ["sad"]:
                                std_emotion = "sad"
                            else:
                                # Skip other emotions for this demo
                                continue
                            
                            all_data.append({
                                'utterance_id': utterance_id,
                                'valence': vad_values[0],
                                'arousal': vad_values[1],
                                'dominance': vad_values[2],
                                'emotion': std_emotion,
                                # Mock data for features that would normally be extracted
                                'audio_features': np.random.randn(AUDIO_FEATURE_DIM).tolist(),
                                'text': f"Mock text for {utterance_id}"
                            })
        
        if all_data:
            return pd.DataFrame(all_data)
        return None
    except Exception as e:
        print(f"Error parsing IEMOCAP: {e}")
        return None

def calculate_vad_accuracy(y_true, y_pred):
    """
    Calculate accuracy metrics for VAD prediction.
    
    Returns:
    - mse: Mean Squared Error
    - r2: R-squared score
    - custom_accuracy: Percentage of predictions within a specific error threshold
    """
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Custom accuracy: percentage of predictions within ±0.2 of true values
    # This gives a more intuitive "accuracy" measure for regression tasks
    abs_diff = np.abs(y_true - y_pred)
    within_threshold = (abs_diff <= 0.2).all(axis=1)
    custom_accuracy = within_threshold.mean() * 100
    
    # Component-wise accuracy
    v_accuracy = (abs_diff[:, 0] <= 0.2).mean() * 100
    a_accuracy = (abs_diff[:, 1] <= 0.2).mean() * 100
    d_accuracy = (abs_diff[:, 2] <= 0.2).mean() * 100
    
    return {
        'mse': mse,
        'r2': r2,
        'accuracy': custom_accuracy,
        'valence_accuracy': v_accuracy,
        'arousal_accuracy': a_accuracy,
        'dominance_accuracy': d_accuracy
    }

def train(audio_model, text_model, fusion_model, train_loader, optimizer, criterion, device):
    """
    Train the models for one epoch.
    
    Returns:
    - Dictionary of training metrics
    """
    audio_model.train()
    text_model.train()
    fusion_model.train()
    
    total_loss = 0
    audio_vad_preds = []
    text_vad_preds = []
    fusion_vad_preds = []
    true_vad_values = []
    attention_weights = []
    step_count = 0
    
    print("Training...")
    for batch_idx, batch in enumerate(train_loader):
        audio_features = batch['audio_features'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        vad_values = batch['vad_values'].to(device)
        
        # Forward pass
        audio_vad = audio_model(audio_features)
        text_vad = text_model(input_ids, attention_mask)
        fused_vad, weights = fusion_model(audio_vad, text_vad)
        
        # Calculate loss
        loss = criterion(fused_vad, vad_values)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Print progress
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}")
        
        # Store predictions and targets for metrics calculation
        audio_vad_preds.append(audio_vad.cpu().detach().numpy())
        text_vad_preds.append(text_vad.cpu().detach().numpy())
        fusion_vad_preds.append(fused_vad.cpu().detach().numpy())
        true_vad_values.append(vad_values.cpu().numpy())
        attention_weights.append(weights.cpu().detach().numpy())
        step_count += 1
    
    # Concatenate batches
    audio_vad_preds = np.vstack(audio_vad_preds)
    text_vad_preds = np.vstack(text_vad_preds)
    fusion_vad_preds = np.vstack(fusion_vad_preds)
    true_vad_values = np.vstack(true_vad_values)
    attention_weights = np.vstack(attention_weights)
    
    # Calculate VAD metrics using the new function
    fusion_metrics = calculate_vad_accuracy(true_vad_values, fusion_vad_preds)
    audio_metrics = calculate_vad_accuracy(true_vad_values, audio_vad_preds)
    text_metrics = calculate_vad_accuracy(true_vad_values, text_vad_preds)
    
    # Calculate average attention weights
    avg_audio_weight = np.mean(attention_weights[:, 0])
    avg_text_weight = np.mean(attention_weights[:, 1])
    
    # Calculate emotion predictions (step 2)
    pred_emotions = []
    for vad in fusion_vad_preds:
        emotion = vad_to_emotion(vad[0], vad[1], vad[2])
        pred_emotions.append(emotion)
    
    # Get actual emotions
    true_emotions = []
    for batch in train_loader:
        true_emotions.extend(batch['emotion'])
    
    # Calculate accuracy
    correct = sum(1 for pred, true in zip(pred_emotions, true_emotions) if pred == true)
    accuracy = correct / len(true_emotions)
    
    metrics = {
        'loss': total_loss / step_count,
        
        # Fusion VAD metrics
        'fusion_mse': fusion_metrics['mse'],
        'fusion_r2': fusion_metrics['r2'],
        'fusion_accuracy': fusion_metrics['accuracy'],
        'fusion_valence_acc': fusion_metrics['valence_accuracy'],
        'fusion_arousal_acc': fusion_metrics['arousal_accuracy'],
        'fusion_dominance_acc': fusion_metrics['dominance_accuracy'],
        
        # Audio VAD metrics
        'audio_mse': audio_metrics['mse'],
        'audio_r2': audio_metrics['r2'],
        'audio_accuracy': audio_metrics['accuracy'],
        
        # Text VAD metrics
        'text_mse': text_metrics['mse'],
        'text_r2': text_metrics['r2'],
        'text_accuracy': text_metrics['accuracy'],
        
        # Emotion metrics
        'accuracy': accuracy * 100,  # Convert to percentage
        
        # Attention weights
        'avg_audio_weight': avg_audio_weight,
        'avg_text_weight': avg_text_weight
    }
    
    return metrics

def evaluate(audio_model, text_model, fusion_model, val_loader, criterion, device):
    """
    Evaluate the models on the validation set.
    
    Returns:
    - Dictionary of evaluation metrics
    - Predicted emotions
    - True emotions
    - Detailed results dataframe
    """
    audio_model.eval()
    text_model.eval()
    fusion_model.eval()
    
    total_loss = 0
    all_fusion_vad_preds = []
    all_audio_vad_preds = []
    all_text_vad_preds = []
    all_true_vad_values = []
    all_attention_weights = []
    all_pred_emotions = []
    all_true_emotions = []
    all_utterance_ids = []
    all_texts = []
    step_count = 0
    
    print("Evaluating...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            audio_features = batch['audio_features'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            vad_values = batch['vad_values'].to(device)
            emotions = batch['emotion']
            utterance_ids = batch['utterance_id']
            texts = batch['text']
            
            # Forward pass
            audio_vad = audio_model(audio_features)
            text_vad = text_model(input_ids, attention_mask)
            fused_vad, weights = fusion_model(audio_vad, text_vad)
            
            # Calculate loss
            loss = criterion(fused_vad, vad_values)
            total_loss += loss.item()
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(val_loader)} Loss: {loss.item():.4f}")
            
            # Store predictions and ground truth
            all_fusion_vad_preds.append(fused_vad.cpu().numpy())
            all_audio_vad_preds.append(audio_vad.cpu().numpy())
            all_text_vad_preds.append(text_vad.cpu().numpy())
            all_true_vad_values.append(vad_values.cpu().numpy())
            all_attention_weights.append(weights.cpu().numpy())
            all_utterance_ids.extend(utterance_ids)
            all_texts.extend(texts)
            
            # Convert VAD values to emotion categories
            for i in range(fused_vad.size(0)):
                v, a, d = fused_vad[i].cpu().numpy()
                emotion = vad_to_emotion(v, a, d)
                all_pred_emotions.append(emotion)
                all_true_emotions.append(emotions[i])
            
            step_count += 1
    
    # Concatenate results from all batches
    all_fusion_vad_preds = np.vstack(all_fusion_vad_preds)
    all_audio_vad_preds = np.vstack(all_audio_vad_preds)
    all_text_vad_preds = np.vstack(all_text_vad_preds)
    all_true_vad_values = np.vstack(all_true_vad_values)
    all_attention_weights = np.vstack(all_attention_weights)
    
    # Calculate VAD prediction metrics using the new function
    fusion_metrics = calculate_vad_accuracy(all_true_vad_values, all_fusion_vad_preds)
    audio_metrics = calculate_vad_accuracy(all_true_vad_values, all_audio_vad_preds)
    text_metrics = calculate_vad_accuracy(all_true_vad_values, all_text_vad_preds)
    
    # Calculate emotion classification metrics
    correct = sum(1 for p, t in zip(all_pred_emotions, all_true_emotions) if p == t)
    accuracy = correct / len(all_pred_emotions)
    
    # Calculate average attention weights
    avg_audio_weight = np.mean(all_attention_weights[:, 0])
    avg_text_weight = np.mean(all_attention_weights[:, 1])
    
    # Create results dict, combining metrics
    results = {
        'loss': total_loss / step_count,
        
        # Fusion VAD metrics
        'fusion_mse': fusion_metrics['mse'],
        'fusion_r2': fusion_metrics['r2'],
        'fusion_accuracy': fusion_metrics['accuracy'],
        'fusion_valence_acc': fusion_metrics['valence_accuracy'],
        'fusion_arousal_acc': fusion_metrics['arousal_accuracy'],
        'fusion_dominance_acc': fusion_metrics['dominance_accuracy'],
        
        # Audio VAD metrics
        'audio_mse': audio_metrics['mse'],
        'audio_r2': audio_metrics['r2'],
        'audio_accuracy': audio_metrics['accuracy'],
        
        # Text VAD metrics
        'text_mse': text_metrics['mse'],
        'text_r2': text_metrics['r2'],
        'text_accuracy': text_metrics['accuracy'],
        
        # Emotion metrics
        'accuracy': accuracy * 100,  # Convert to percentage
        
        # Attention weights
        'avg_audio_weight': avg_audio_weight,
        'avg_text_weight': avg_text_weight,
    }
    
    # Create detailed results dataframe
    details_df = pd.DataFrame({
        'utterance_id': all_utterance_ids,
        'text': all_texts,
        'true_emotion': all_true_emotions,
        'pred_emotion': all_pred_emotions,
        'true_valence': all_true_vad_values[:, 0],
        'true_arousal': all_true_vad_values[:, 1],
        'true_dominance': all_true_vad_values[:, 2],
        'pred_valence': all_fusion_vad_preds[:, 0],
        'pred_arousal': all_fusion_vad_preds[:, 1],
        'pred_dominance': all_fusion_vad_preds[:, 2],
        'audio_weight': all_attention_weights[:, 0],
        'text_weight': all_attention_weights[:, 1]
    })
    
    return results, all_pred_emotions, all_true_emotions, details_df

def print_metrics_table(metrics, title="Metrics"):
    """Print metrics in a nice table format"""
    print(f"\n{title}")
    print("-" * 80)
    
    rows = []
    for key, value in metrics.items():
        if isinstance(value, float):
            rows.append([key, f"{value:.4f}"])
        else:
            rows.append([key, value])
    
    print(tabulate(rows, headers=["Metric", "Value"], tablefmt="grid"))
    print("-" * 80)

def print_classification_report_table(true_emotions, pred_emotions, title="Classification Report"):
    """Print classification report in a nice table format"""
    report = classification_report(true_emotions, pred_emotions, output_dict=True)
    
    print(f"\n{title}")
    print("-" * 80)
    
    # Print per-class metrics
    class_rows = []
    for cls, metrics in report.items():
        if cls not in ['accuracy', 'macro avg', 'weighted avg']:
            class_rows.append([
                cls, 
                f"{metrics['precision']:.4f}", 
                f"{metrics['recall']:.4f}", 
                f"{metrics['f1-score']:.4f}", 
                f"{metrics['support']}"
            ])
    
    print(tabulate(class_rows, headers=["Class", "Precision", "Recall", "F1-Score", "Support"], tablefmt="grid"))
    
    # Print summary metrics
    summary_rows = []
    for avg_type in ['accuracy', 'macro avg', 'weighted avg']:
        if avg_type == 'accuracy':
            summary_rows.append([avg_type, "", "", f"{report[avg_type]:.4f}", f"{report['macro avg']['support']}"])
        else:
            metrics = report[avg_type]
            summary_rows.append([
                avg_type, 
                f"{metrics['precision']:.4f}", 
                f"{metrics['recall']:.4f}", 
                f"{metrics['f1-score']:.4f}", 
                f"{metrics['support']}"
            ])
    
    print(tabulate(summary_rows, headers=["Metric", "Precision", "Recall", "F1-Score", "Support"], tablefmt="grid"))
    print("-" * 80)

def plot_confusion_matrix(true_emotions, pred_emotions, title="Confusion Matrix"):
    """Plot confusion matrix for emotion classification"""
    # Get unique emotions
    emotions = sorted(list(set(true_emotions + pred_emotions)))
    
    # Create confusion matrix
    cm = confusion_matrix(true_emotions, pred_emotions, labels=emotions)
    
    # Plot simple text-based confusion matrix
    print("\nConfusion Matrix:")
    print("-" * 80)
    print(tabulate(cm, headers=emotions, showindex=emotions, tablefmt="grid"))
    print("-" * 80)
    
    # If matplotlib is working, also create a visual version
    try:
        # Plot
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(emotions))
        plt.xticks(tick_marks, emotions, rotation=45)
        plt.yticks(tick_marks, emotions)
        
        # Add text annotations
        for i in range(len(emotions)):
            for j in range(len(emotions)):
                plt.text(j, i, str(cm[i, j]), 
                        horizontalalignment="center", 
                        color="white" if cm[i, j] > np.max(cm)/2 else "black")
        
        plt.ylabel('True Emotion')
        plt.xlabel('Predicted Emotion')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        print(f"Confusion matrix saved as 'confusion_matrix.png'")
    except Exception as e:
        print(f"Could not create visual confusion matrix plot: {e}")

def visualize_vad_emotion_space():
    """Visualize the VAD space and emotion categories (text-based)"""
    print("\nVAD Space Emotion Mapping:")
    
    # Generate a sample of points in the VAD space
    vad_points = []
    emotions = []
    
    # Create a grid of VAD values (coarser grid for simplicity)
    for v in np.linspace(-1, 1, 5):
        for a in np.linspace(-1, 1, 5):
            for d in np.linspace(-1, 1, 5):
                vad_points.append([v, a, d])
                emotions.append(vad_to_emotion(v, a, d))
    
    # Count emotions
    emotion_counts = {}
    for emotion in emotions:
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    # Print emotion distribution
    print("\nEmotion distribution in VAD space:")
    print(tabulate(
        [[emotion, count] for emotion, count in emotion_counts.items()],
        headers=["Emotion", "Count"],
        tablefmt="grid"
    ))
    
    # Try to create a visual plot if matplotlib is working
    try:
        # Create a DataFrame
        vad_df = pd.DataFrame({
            'valence': [p[0] for p in vad_points],
            'arousal': [p[1] for p in vad_points],
            'dominance': [p[2] for p in vad_points],
            'emotion': emotions
        })
        
        # Visualize
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Define colors for each emotion
        emotion_colors = {
            'happy': 'yellow',
            'excited': 'orange',
            'content': 'green',
            'relaxed': 'lightgreen',
            'angry': 'red',
            'fearful': 'purple',
            'disgusted': 'brown',
            'sad': 'blue'
        }
        
        # Plot each emotion category
        for emotion in vad_df['emotion'].unique():
            subset = vad_df[vad_df['emotion'] == emotion]
            ax.scatter(
                subset['valence'], 
                subset['arousal'], 
                subset['dominance'],
                label=emotion,
                alpha=0.7
            )
        
        ax.set_xlabel('Valence')
        ax.set_ylabel('Arousal')
        ax.set_zlabel('Dominance')
        ax.set_title('Emotion Categories in VAD Space')
        ax.legend()
        
        # Set limits
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        
        plt.savefig('vad_emotion_space.png')
        print("Visualization saved as 'vad_emotion_space.png'")
    except Exception as e:
        print(f"Could not create visual VAD space plot: {e}")

def load_dataset(data_path=None):
    """
    Load the IEMOCAP dataset or mock dataset.
    
    Args:
        data_path: Optional path to custom data file
        
    Returns:
        DataFrame with dataset
    """
    # If data path is provided, load it directly
    if data_path and os.path.exists(data_path):
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
    # Check if data already exists in data directory
    elif os.path.exists(os.path.join(DEFAULT_DATA_DIR, "processed_data.csv")):
        print(f"Loading existing processed data from {os.path.join(DEFAULT_DATA_DIR, 'processed_data.csv')}")
        df = pd.read_csv(os.path.join(DEFAULT_DATA_DIR, "processed_data.csv"))
    # Try to parse IEMOCAP data
    else:
        print("Attempting to parse IEMOCAP dataset...")
        df = parse_iemocap_emotions()
        
        # If no data or error, create mock data
        if df is None or len(df) == 0:
            print("Could not parse IEMOCAP data, creating mock dataset instead.")
            df = create_mock_dataset(n_samples=1000)
    
    # Process string-formatted audio features if needed
    if 'audio_features' in df.columns and isinstance(df['audio_features'].iloc[0], str):
        print("Converting string audio features to arrays...")
        # Use a safer approach to convert string representation to numpy arrays
        def parse_array(x):
            if not isinstance(x, str):
                return x
            try:
                # Try to safely convert the string to a list
                x = x.strip('[]')  # Remove brackets
                x = x.replace('\n', ' ')  # Replace newlines with spaces
                values = [float(val) for val in x.split()]  # Split by whitespace and convert to float
                return np.array(values)
            except Exception as e:
                print(f"Error parsing array: {e}")
                # Return a zero array as fallback
                return np.zeros(AUDIO_FEATURE_DIM)
                
        # Apply the parsing function
        df['audio_features'] = df['audio_features'].apply(parse_array)
    
    # Add waveform column for Wav2Vec model if needed
    if 'audio_waveform' not in df.columns:
        # Just add a placeholder - in a real implementation this would be the actual waveform
        df['audio_waveform'] = df['audio_features'].apply(
            lambda x: np.zeros(16000) if isinstance(x, np.ndarray) else np.zeros(16000)
        )
    
    return df

def print_dataset_stats(df):
    """
    Print statistics about the dataset.
    
    Args:
        df: DataFrame with dataset
    """
    print(f"Dataset created with {len(df)} samples")
    
    # Print emotion distribution
    print("\nEmotion distribution:")
    emotion_counts = df['emotion'].value_counts()
    print(tabulate(
        [[emotion, count, f"{count/len(df)*100:.2f}%"] for emotion, count in emotion_counts.items()],
        headers=["Emotion", "Count", "Percentage"],
        tablefmt="grid"
    ))
    
    # Print VAD statistics
    print("\nVAD statistics:")
    vad_stats = df[['valence', 'arousal', 'dominance']].describe().round(4)
    print(tabulate(
        vad_stats,
        headers="keys",
        tablefmt="grid"
    ))
    
    # Check for NaN values
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        print(f"\nWarning: Dataset contains {nan_count} NaN values")
    
    # Check class balance
    min_class = emotion_counts.min()
    max_class = emotion_counts.max()
    imbalance = max_class / min_class if min_class > 0 else float('inf')
    
    if imbalance > 2:
        print(f"\nWarning: Dataset is imbalanced (ratio max/min: {imbalance:.2f})")
    else:
        print("\nClass distribution is reasonably balanced")

def main(force_train=False, evaluate_only=False, visualize=False, 
         checkpoint_dir=DEFAULT_CHECKPOINT_DIR, data_path=None, 
         vad_to_emotion_func=None):
    """
    Main function to run the emotion recognition pipeline.
    
    Args:
        force_train: Whether to force training even if models exist
        evaluate_only: Whether to only evaluate existing models
        visualize: Whether to generate visualizations
        checkpoint_dir: Directory for model checkpoints
        data_path: Path to custom data
        vad_to_emotion_func: Custom VAD to emotion mapping function
    """
    print("-" * 80)
    print("IEMOCAP EMOTION RECOGNITION WITH TWO-STEP APPROACH")
    print("Step 1: Convert modalities (audio/text) to VAD tuples")
    print("Step 2: Map VAD tuples to emotion categories")
    print("-" * 80)
    
    # Set up vad_to_emotion mapping function
    global vad_to_emotion
    if vad_to_emotion_func is not None:
        print("Using provided VAD-to-emotion mapping function")
        vad_to_emotion = vad_to_emotion_func
        CONFIG["vad_mapping"]["use_ml"] = True
    elif use_ml_vad_to_emotion:
        print("Using machine learning based VAD-to-emotion mapping")
        vad_to_emotion = get_vad_to_emotion_predictor()
        CONFIG["vad_mapping"]["use_ml"] = True
    else:
        print("Using rule-based VAD-to-emotion mapping")
    
    # Check for existing models
    model_paths = get_latest_model_paths(checkpoint_dir)
    
    # Print status of models
    print("\nModel status:")
    if model_paths:
        for key, path in model_paths.items():
            status = "Found" if path and os.path.exists(path) else "Not found"
            print(f"  {key}: {status}")
    else:
        print("  No existing models found, will train from scratch")
    
    # Generate a new run ID for this training session
    run_id = generate_run_id()
    print(f"\nRun ID for this session: {run_id}")
    
    # Load or create dataset
    print("\nPreparing dataset...")
    df = load_dataset(data_path)
    
    # Save processed data for reference
    data_file = os.path.join(DEFAULT_DATA_DIR, "processed_data.csv")
    df.to_csv(data_file, index=False)
    print(f"Dataset saved to {data_file}")
    
    # Print dataset statistics
    print_dataset_stats(df)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["text_model"]["model_name"])
    
    # Split data - stratify to ensure balanced emotion distribution
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_SEED, 
        stratify=df['emotion']
    )
    
    print(f"\nTrain set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    
    # Create datasets and dataloaders
    train_dataset = IEMOCAPDataset(
        train_df, 
        tokenizer, 
        use_wav2vec=CONFIG["audio_model"]["use_wav2vec"]
    )
    
    val_dataset = IEMOCAPDataset(
        val_df, 
        tokenizer, 
        use_wav2vec=CONFIG["audio_model"]["use_wav2vec"]
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG["batch_size"]
    )
    
    # Initialize models
    audio_model = AudioVADModel(
        input_dim=AUDIO_FEATURE_DIM,
        use_wav2vec=CONFIG["audio_model"]["use_wav2vec"]
    ).to(DEVICE)
    
    text_model = TextVADModel(
        model_name=CONFIG["text_model"]["model_name"],
        finetune=CONFIG["text_model"]["finetune"]
    ).to(DEVICE)
    
    fusion_model = FusionVADModel().to(DEVICE)
    
    # Determine if we should use existing models or train new ones
    should_train = force_train or not model_paths
    
    # Load existing models if available and not forcing training
    if not should_train:
        print("\nLoading existing models...")
        try:
            audio_model.load_state_dict(torch.load(model_paths['audio_model_path']))
            text_model.load_state_dict(torch.load(model_paths['text_model_path']))
            fusion_model.load_state_dict(torch.load(model_paths['fusion_model_path']))
            print("Models loaded successfully!")
            
            # Evaluate loaded models
            print("\nEvaluating loaded models...")
            val_metrics, val_pred_emotions, val_true_emotions, val_details_df = evaluate(
                audio_model, text_model, fusion_model, val_loader, nn.MSELoss(), DEVICE
            )
            print_metrics_table(val_metrics, title="Validation Metrics (Loaded Models)")
            print_classification_report_table(
                val_true_emotions, val_pred_emotions,
                title="Validation Classification Report (Loaded Models)"
            )
            
            # If only evaluating, skip to visualization
            if evaluate_only:
                # Save evaluation results
                eval_file = os.path.join(DEFAULT_LOGS_DIR, f"evaluation_{run_id}.csv")
                val_details_df.to_csv(eval_file, index=False)
                print(f"Evaluation results saved to {eval_file}")
                
                # Run visualizations if requested
                if visualize:
                    plot_confusion_matrix(val_pred_emotions, val_true_emotions)
                    visualize_vad_emotion_space()
                
                # Print summary
                print("\nEvaluation completed!")
                print("-" * 80)
                print("SUMMARY OF THE TWO-STEP APPROACH:")
                print("1. Modality → VAD Conversion:")
                print(f"   - Audio MSE: {val_metrics['audio_mse']:.4f}")
                print(f"   - Text MSE: {val_metrics['text_mse']:.4f}")
                print(f"   - Fusion MSE: {val_metrics['fusion_mse']:.4f}")
                print("2. VAD → Emotion Classification:")
                print(f"   - Accuracy: {val_metrics['accuracy']:.4f}")
                print(f"   - Overall Loss: {val_metrics['loss']:.4f}")
                print("-" * 80)
                
                return val_metrics
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Proceeding with training new models...")
            should_train = True
    
    # Return if we're only evaluating and there are no models to load
    if evaluate_only and not should_train:
        print("No models available for evaluation.")
        return None
    
    # If we get here, we need to train models
    if should_train:
        # Initialize optimizer with layer-wise learning rates
        # Higher learning rate for the new fusion model, lower for pretrained models
        param_groups = [
            {'params': audio_model.parameters(), 'lr': CONFIG["learning_rate"]},
            {'params': text_model.parameters(), 'lr': CONFIG["learning_rate"] * 0.5},  # Lower for pretrained LM
            {'params': fusion_model.parameters(), 'lr': CONFIG["learning_rate"] * 2.0}  # Higher for fusion
        ]
        
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        
        # Loss function
        criterion = nn.MSELoss()
        
        # Train and evaluate models
        best_val_loss = float('inf')
        train_metrics_history = []
        val_metrics_history = []
        
        print("\nStarting training...")
        for epoch in range(CONFIG["num_epochs"]):
            print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
            
            # Train
            train_metrics = train(audio_model, text_model, fusion_model, 
                                train_loader, optimizer, criterion, DEVICE)
            train_metrics_history.append(train_metrics)
            
            # Evaluate
            val_metrics, val_pred_emotions, val_true_emotions, val_details_df = evaluate(
                audio_model, text_model, fusion_model, val_loader, criterion, DEVICE
            )
            val_metrics_history.append(val_metrics)
            
            # Print metrics tables
            print_metrics_table(train_metrics, title=f"Training Metrics - Epoch {epoch+1}")
            print_metrics_table(val_metrics, title=f"Validation Metrics - Epoch {epoch+1}")
            
            # Print classification report
            print_classification_report_table(
                val_true_emotions, val_pred_emotions,
                title=f"Validation Classification Report - Epoch {epoch+1}"
            )
            
            # Update learning rate
            scheduler.step(val_metrics['loss'])
            
            # Save models if validation loss improves
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                
                # Generate model paths with run ID
                model_paths = get_model_paths(checkpoint_dir=checkpoint_dir, run_id=run_id)
                
                # Save models
                torch.save(audio_model.state_dict(), model_paths['audio_model_path'])
                torch.save(text_model.state_dict(), model_paths['text_model_path'])
                torch.save(fusion_model.state_dict(), model_paths['fusion_model_path'])
                
                # Save best validation details
                val_details_df.to_csv(os.path.join(DEFAULT_DATA_DIR, "best_validation_results.csv"), index=False)
                
                # Save model info
                save_model_info(model_paths, val_metrics, CONFIG, checkpoint_dir=checkpoint_dir)
                
                print("Models saved!")
        
        print("\nTraining completed!")
        
        # Plot confusion matrix for the final epoch
        plot_confusion_matrix(val_pred_emotions, val_true_emotions)
        
        # Create and save summary tables
        train_summary = pd.DataFrame(train_metrics_history)
        val_summary = pd.DataFrame(val_metrics_history)
        
        train_summary.to_csv(os.path.join(DEFAULT_LOGS_DIR, "train_metrics_history.csv"), index=False)
        val_summary.to_csv(os.path.join(DEFAULT_LOGS_DIR, "val_metrics_history.csv"), index=False)
        
        # Print summary of training history
        print("\nTraining History Summary:")
        print(tabulate(
            train_summary[['loss', 'audio_mse', 'text_mse', 'fusion_mse', 'accuracy']].describe().reset_index(),
            headers="keys",
            tablefmt="grid",
            floatfmt=".4f"
        ))
        
        print("\nValidation History Summary:")
        print(tabulate(
            val_summary[['loss', 'audio_mse', 'text_mse', 'fusion_mse', 'accuracy']].describe().reset_index(),
            headers="keys",
            tablefmt="grid",
            floatfmt=".4f"
        ))
    
    # Run visualizations if requested
    if visualize:
        visualize_vad_emotion_space()
        
        # Try to create plots if matplotlib is working
        try:
            # Plot training curves
            plt.figure(figsize=(15, 10))
            
            # Loss plot
            plt.subplot(2, 2, 1)
            plt.plot(range(1, CONFIG["num_epochs"]+1), [m['loss'] for m in train_metrics_history], 'b-', label='Training Loss')
            plt.plot(range(1, CONFIG["num_epochs"]+1), [m['loss'] for m in val_metrics_history], 'r-', label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)
            
            # MSE plot
            plt.subplot(2, 2, 2)
            plt.plot(range(1, CONFIG["num_epochs"]+1), [m['audio_mse'] for m in val_metrics_history], 'g-', label='Audio MSE')
            plt.plot(range(1, CONFIG["num_epochs"]+1), [m['text_mse'] for m in val_metrics_history], 'b-', label='Text MSE')
            plt.plot(range(1, CONFIG["num_epochs"]+1), [m['fusion_mse'] for m in val_metrics_history], 'r-', label='Fusion MSE')
            plt.xlabel('Epoch')
            plt.ylabel('MSE')
            plt.title('Validation MSE by Modality')
            plt.legend()
            plt.grid(True)
            
            # Accuracy plot
            plt.subplot(2, 2, 3)
            plt.plot(range(1, CONFIG["num_epochs"]+1), [m['accuracy'] for m in train_metrics_history], 'b-', label='Training Accuracy')
            plt.plot(range(1, CONFIG["num_epochs"]+1), [m['accuracy'] for m in val_metrics_history], 'r-', label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Training and Validation Accuracy')
            plt.legend()
            plt.grid(True)
            
            # Attention weights plot
            plt.subplot(2, 2, 4)
            plt.plot(range(1, CONFIG["num_epochs"]+1), [m['avg_audio_weight'] for m in val_metrics_history], 'g-', label='Audio Weight')
            plt.plot(range(1, CONFIG["num_epochs"]+1), [m['avg_text_weight'] for m in val_metrics_history], 'b-', label='Text Weight')
            plt.xlabel('Epoch')
            plt.ylabel('Average Attention Weight')
            plt.title('Fusion Model Attention Weights')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(DEFAULT_LOGS_DIR, 'training_history.png'))
            print("Training history plots saved as 'training_history.png'")
        except Exception as e:
            print(f"Could not create training history plots: {e}")
    
    # Print final summary
    print("\nEmotion recognition completed!")
    print("-" * 80)
    print("SUMMARY OF THE TWO-STEP APPROACH:")
    print("1. Modality → VAD Conversion:")
    print(f"   - Audio MSE: {val_metrics['audio_mse']:.4f}")
    print(f"   - Text MSE: {val_metrics['text_mse']:.4f}")
    print(f"   - Fusion MSE: {val_metrics['fusion_mse']:.4f}")
    print(f"   - Audio Accuracy: {val_metrics.get('audio_accuracy', 0):.2f}%")
    print(f"   - Text Accuracy: {val_metrics.get('text_accuracy', 0):.2f}%")
    print(f"   - Fusion Accuracy: {val_metrics.get('fusion_accuracy', 0):.2f}%")
    print("2. VAD → Emotion Classification:")
    print(f"   - Accuracy: {val_metrics['accuracy']:.2f}%")
    print(f"   - Overall Loss: {val_metrics['loss']:.4f}")
    print("-" * 80)
    
    return val_metrics

if __name__ == "__main__":
    main() 