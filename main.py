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
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tabulate import tabulate
from process_vad import vad_to_emotion
import random
import time
import platform

# Check if vad_to_emotion_model.py exists and import get_vad_to_emotion_predictor if it does
try:
    from vad_to_emotion_model import get_vad_to_emotion_predictor
    use_ml_vad_to_emotion = True
except ImportError:
    use_ml_vad_to_emotion = False

# Constants
IEMOCAP_PATH = "Datasets/IEMOCAP_full_release"
BERT_MODEL = "bert-base-uncased"

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

NUM_EPOCHS = 5  # Reduced for demonstration
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
RANDOM_SEED = 42
AUDIO_FEATURE_DIM = 68

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Model tracking file
MODEL_TRACKER_FILE = "model_tracker.txt"

def generate_run_id():
    """Generate a unique run ID for model checkpoints"""
    timestamp = int(time.time())
    random_suffix = random.randint(1000, 9999)
    return f"{timestamp}_{random_suffix}"

def load_model_paths_from_tracker():
    """Load model paths from the tracker file if it exists"""
    model_paths = {
        'audio_model_path': None,
        'text_model_path': None,
        'fusion_model_path': None,
        'vad_to_emotion_model_path': None
    }
    
    if not os.path.exists(MODEL_TRACKER_FILE):
        return model_paths
        
    try:
        with open(MODEL_TRACKER_FILE, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if '=' in line:
                key, value = [part.strip() for part in line.split('=', 1)]
                if key in model_paths and value != 'None':
                    model_paths[key] = value
                    
        return model_paths
    except Exception as e:
        print(f"Error reading model tracker file: {e}")
        return model_paths

def update_model_tracker(model_paths, performance_metrics):
    """Update the model tracker file with new paths and metrics"""
    try:
        with open(MODEL_TRACKER_FILE, 'w') as f:
            f.write("# Model Tracking File\n")
            f.write(f"# Last Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Latest Model Paths\n")
            for key, value in model_paths.items():
                f.write(f"{key} = {value if value else 'None'}\n")
            
            f.write("\n## Performance Metrics\n")
            f.write("# Modality to VAD Performance\n")
            f.write(f"audio_to_vad_mse = {performance_metrics.get('audio_mse', 'Unknown')}\n")
            f.write(f"text_to_vad_mse = {performance_metrics.get('text_mse', 'Unknown')}\n")
            f.write(f"fusion_to_vad_mse = {performance_metrics.get('fusion_mse', 'Unknown')}\n")
            
            f.write("\n# VAD to Emotion Performance\n")
            f.write(f"vad_to_emotion_accuracy = {performance_metrics.get('accuracy', 'Unknown')}\n")
            
            f.write("\n# End-to-End Performance\n")
            f.write(f"end_to_end_accuracy = {performance_metrics.get('accuracy', 'Unknown')}\n")
            
            f.write("\n## Notes\n")
            if performance_metrics.get('accuracy', 0) < 0.4:
                f.write("# The VAD to emotion mapping is the clear bottleneck in the pipeline.\n")
                f.write("# Consider replacing the rule-based approach with a machine learning model.\n")
            
        print(f"Updated model tracker file: {MODEL_TRACKER_FILE}")
    except Exception as e:
        print(f"Error updating model tracker file: {e}")

class AudioVADModel(nn.Module):
    """
    Model to convert audio features to VAD (valence-arousal-dominance) values.
    
    Uses a fully connected neural network with batch normalization and dropout 
    for regularization. Improved with residual connections and layer normalization.
    """
    def __init__(self, input_dim=AUDIO_FEATURE_DIM):
        super(AudioVADModel, self).__init__()
        
        # Improved architecture with residual connections
        self.input_norm = nn.BatchNorm1d(input_dim)
        
        # First block
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        # Residual block 1
        self.res1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.LayerNorm(256)
        )
        
        # Residual block 2
        self.res2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.LayerNorm(256)
        )
        
        # Output layer
        self.output = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 3),
            nn.Tanh()  # Scale values between -1 and 1
        )
        
        # Initialize weights for better convergence
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, x):
        x = self.input_norm(x)
        
        # First block
        x1 = self.fc1(x)
        
        # Residual block 1
        res1_out = self.res1(x1)
        x2 = x1 + res1_out  # Residual connection
        
        # Residual block 2
        res2_out = self.res2(x2)
        x3 = x2 + res2_out  # Residual connection
        
        # Output
        vad = self.output(x3)
        
        return vad

class TextVADModel(nn.Module):
    """
    Model to convert text embeddings to VAD (valence-arousal-dominance) values.
    
    Uses RoBERTa for text encoding and a fully connected neural network with
    batch normalization for VAD prediction. RoBERTa generally outperforms BERT on many tasks.
    """
    def __init__(self, bert_model="roberta-base"):
        super(TextVADModel, self).__init__()
        # Use RoBERTa instead of BERT for better performance
        self.roberta = RobertaModel.from_pretrained(bert_model)
        
        # Freeze early layers to prevent overfitting
        for param in self.roberta.embeddings.parameters():
            param.requires_grad = False
        
        for i, param in enumerate(self.roberta.encoder.layer):
            if i < 8:  # Freeze first 8 layers
                for param in param.parameters():
                    param.requires_grad = False
        
        # Use a deeper network for VAD prediction
        self.fc = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3),  # Output: valence, arousal, dominance
            nn.Tanh()  # Scale values between -1 and 1
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        vad_values = self.fc(cls_output)
        return vad_values

class FusionVADModel(nn.Module):
    """
    Model to combine audio and text VAD predictions.
    
    Uses a cross-attention mechanism to effectively combine information
    from both modalities with learnable interactions between them.
    """
    def __init__(self, feature_dim=3):
        super(FusionVADModel, self).__init__()
        
        # Feature dimension (VAD = 3)
        self.feature_dim = feature_dim
        
        # Initial projection layers
        self.audio_proj = nn.Linear(feature_dim, 64)
        self.text_proj = nn.Linear(feature_dim, 64)
        
        # Cross-attention layers
        # Query, Key, Value projections for audio attending to text
        self.audio_query = nn.Linear(64, 64)
        self.text_key = nn.Linear(64, 64)
        self.text_value = nn.Linear(64, 64)
        
        # Query, Key, Value projections for text attending to audio
        self.text_query = nn.Linear(64, 64)
        self.audio_key = nn.Linear(64, 64)
        self.audio_value = nn.Linear(64, 64)
        
        # Layer norm for attention outputs
        self.layer_norm1 = nn.LayerNorm(64)
        self.layer_norm2 = nn.LayerNorm(64)
        
        # Integration layer
        self.integration = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Final VAD prediction layer
        self.vad_predictor = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, feature_dim),
            nn.Tanh()  # Scale values between -1 and 1
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, audio_vad, text_vad):
        batch_size = audio_vad.size(0)
        
        # Project inputs to higher dimension
        audio_features = self.audio_proj(audio_vad)  # [batch_size, 64]
        text_features = self.text_proj(text_vad)     # [batch_size, 64]
        
        # Cross-attention: audio attending to text
        audio_queries = self.audio_query(audio_features).unsqueeze(1)  # [batch_size, 1, 64]
        text_keys = self.text_key(text_features).unsqueeze(1)          # [batch_size, 1, 64]
        text_values = self.text_value(text_features).unsqueeze(1)      # [batch_size, 1, 64]
        
        # Compute attention scores
        audio_text_scores = torch.matmul(audio_queries, text_keys.transpose(-2, -1)) / (64 ** 0.5)  # [batch_size, 1, 1]
        audio_text_attention = torch.softmax(audio_text_scores, dim=-1)
        
        # Apply attention to values
        audio_attended_text = torch.matmul(audio_text_attention, text_values).squeeze(1)  # [batch_size, 64]
        audio_attended_text = self.layer_norm1(audio_attended_text + audio_features)  # residual connection
        
        # Cross-attention: text attending to audio
        text_queries = self.text_query(text_features).unsqueeze(1)     # [batch_size, 1, 64]
        audio_keys = self.audio_key(audio_features).unsqueeze(1)       # [batch_size, 1, 64]
        audio_values = self.audio_value(audio_features).unsqueeze(1)   # [batch_size, 1, 64]
        
        # Compute attention scores
        text_audio_scores = torch.matmul(text_queries, audio_keys.transpose(-2, -1)) / (64 ** 0.5)  # [batch_size, 1, 1]
        text_audio_attention = torch.softmax(text_audio_scores, dim=-1)
        
        # Apply attention to values
        text_attended_audio = torch.matmul(text_audio_attention, audio_values).squeeze(1)  # [batch_size, 64]
        text_attended_audio = self.layer_norm2(text_attended_audio + text_features)  # residual connection
        
        # Calculate attention weights for returning
        audio_weight = audio_text_attention.mean(dim=1).squeeze(-1)  # [batch_size]
        text_weight = text_audio_attention.mean(dim=1).squeeze(-1)   # [batch_size]
        weights = torch.stack([audio_weight, text_weight], dim=1)     # [batch_size, 2]
        
        # Concatenate attended features
        combined_features = torch.cat([audio_attended_text, text_attended_audio], dim=1)  # [batch_size, 128]
        
        # Integrate features
        integrated = self.integration(combined_features)  # [batch_size, 64]
        
        # Predict final VAD values
        vad_values = self.vad_predictor(integrated)  # [batch_size, 3]
        
        return vad_values, weights

class IEMOCAPDataset(Dataset):
    """
    IEMOCAP dataset for multimodal emotion recognition.
    
    Prepares data for both audio and text modalities.
    Updated to work with RoBERTa tokenizer.
    """
    def __init__(self, data, tokenizer=None, max_len=128):
        self.data = data
        
        # Initialize tokenizer if not provided
        if tokenizer is None:
            from transformers import RobertaTokenizer
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        else:
            self.tokenizer = tokenizer
            
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['text']
        
        # Prepare audio features
        audio_features = torch.tensor(row['audio_features'], dtype=torch.float)
        
        # Prepare text features
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
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

def main():
    """Main function to run the emotion recognition pipeline"""
    print("-" * 80)
    print("IEMOCAP EMOTION RECOGNITION WITH TWO-STEP APPROACH")
    print("Step 1: Convert modalities (audio/text) to VAD tuples")
    print("Step 2: Map VAD tuples to emotion categories")
    print("-" * 80)
    
    # Check if we can use ML-based VAD to emotion mapping
    global vad_to_emotion
    if use_ml_vad_to_emotion:
        print("Using machine learning based VAD-to-emotion mapping")
        vad_to_emotion = get_vad_to_emotion_predictor()
    else:
        print("Using rule-based VAD-to-emotion mapping")
    
    # Load model paths from tracker
    model_paths = load_model_paths_from_tracker()
    print("\nChecking for existing models:")
    for key, path in model_paths.items():
        print(f"  {key}: {'Found' if path and os.path.exists(path) else 'Not found'}")
    
    # Generate a new run ID for this training session
    run_id = generate_run_id()
    print(f"\nRun ID for this session: {run_id}")
    
    # Try to parse IEMOCAP data first
    print("\nAttempting to parse IEMOCAP dataset...")
    df = parse_iemocap_emotions()
    
    # If no data or error, create mock data
    if df is None or len(df) == 0:
        print("Could not parse IEMOCAP data, creating mock dataset instead.")
        df = create_mock_dataset(n_samples=1000)  # Increased for better training
    
    print(f"Dataset created with {len(df)} samples")
    print("\nEmotion distribution:")
    emotion_counts = df['emotion'].value_counts()
    print(tabulate(
        [[emotion, count] for emotion, count in emotion_counts.items()],
        headers=["Emotion", "Count"],
        tablefmt="grid"
    ))
    
    # Save processed data for reference
    df.to_csv("processed_data.csv", index=False)
    
    # Initialize tokenizer (using RoBERTa instead of BERT)
    from transformers import RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    
    # Split data - stratify to ensure balanced emotion distribution
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_SEED, 
        stratify=df['emotion']
    )
    
    print(f"\nTrain set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    
    # Create datasets and dataloaders
    train_dataset = IEMOCAPDataset(train_df, tokenizer)
    val_dataset = IEMOCAPDataset(val_df, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize models
    audio_model = AudioVADModel().to(DEVICE)
    text_model = TextVADModel().to(DEVICE)
    fusion_model = FusionVADModel().to(DEVICE)
    
    # Load existing models if available
    if all(model_paths[key] and os.path.exists(model_paths[key]) for key in ['audio_model_path', 'text_model_path', 'fusion_model_path']):
        print("\nLoading existing models...")
        try:
            audio_model.load_state_dict(torch.load(model_paths['audio_model_path']))
            text_model.load_state_dict(torch.load(model_paths['text_model_path']))
            fusion_model.load_state_dict(torch.load(model_paths['fusion_model_path']))
            print("Models loaded successfully!")
            
            # Evaluate immediately with loaded models
            print("\nEvaluating loaded models...")
            val_metrics, val_pred_emotions, val_true_emotions, val_details_df = evaluate(
                audio_model, text_model, fusion_model, val_loader, nn.MSELoss(), DEVICE
            )
            print_metrics_table(val_metrics, title="Validation Metrics (Loaded Models)")
            print_classification_report_table(
                val_true_emotions, val_pred_emotions,
                title="Validation Classification Report (Loaded Models)"
            )
            
            # Skip training if user specifies
            user_input = input("\nDo you want to skip training and use these models? (y/n): ")
            if user_input.lower() in ['y', 'yes']:
                # Skip to visualization and summary
                plot_confusion_matrix(val_true_emotions, val_pred_emotions)
                visualize_vad_emotion_space()
                
                # Print summary
                print("\nEmotion recognition demo completed!")
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
                
                # Update model tracker
                update_model_tracker(model_paths, val_metrics)
                return
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Proceeding with training new models...")
    
    # Initialize optimizer with learning rate scheduler
    optimizer = torch.optim.AdamW(
        list(audio_model.parameters()) + 
        list(text_model.parameters()) + 
        list(fusion_model.parameters()),
        lr=LEARNING_RATE,
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
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
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
            model_paths = {
                'audio_model_path': f"audio_vad_model_{run_id}.pt",
                'text_model_path': f"text_vad_model_{run_id}.pt",
                'fusion_model_path': f"fusion_vad_model_{run_id}.pt",
                'vad_to_emotion_model_path': None
            }
            
            # Save models
            torch.save(audio_model.state_dict(), model_paths['audio_model_path'])
            torch.save(text_model.state_dict(), model_paths['text_model_path'])
            torch.save(fusion_model.state_dict(), model_paths['fusion_model_path'])
            
            # Also save with standard names for backward compatibility
            torch.save(audio_model.state_dict(), "audio_vad_model.pt")
            torch.save(text_model.state_dict(), "text_vad_model.pt")
            torch.save(fusion_model.state_dict(), "fusion_vad_model.pt")
            
            # Save best validation details
            val_details_df.to_csv("best_validation_results.csv", index=False)
            
            # Update model tracker
            update_model_tracker(model_paths, val_metrics)
            
            print("Models saved!")
    
    print("\nTraining completed!")
    
    # Plot confusion matrix for the final epoch
    plot_confusion_matrix(val_true_emotions, val_pred_emotions)
    
    # Create and save summary tables
    train_summary = pd.DataFrame(train_metrics_history)
    val_summary = pd.DataFrame(val_metrics_history)
    
    train_summary.to_csv("train_metrics_history.csv", index=False)
    val_summary.to_csv("val_metrics_history.csv", index=False)
    
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
    
    # Visualize VAD-to-emotion mapping
    visualize_vad_emotion_space()
    
    # Try to create plots if matplotlib is working
    try:
        # Plot training curves
        plt.figure(figsize=(15, 10))
        
        # Loss plot
        plt.subplot(2, 2, 1)
        plt.plot(range(1, NUM_EPOCHS+1), [m['loss'] for m in train_metrics_history], 'b-', label='Training Loss')
        plt.plot(range(1, NUM_EPOCHS+1), [m['loss'] for m in val_metrics_history], 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # MSE plot
        plt.subplot(2, 2, 2)
        plt.plot(range(1, NUM_EPOCHS+1), [m['audio_mse'] for m in val_metrics_history], 'g-', label='Audio MSE')
        plt.plot(range(1, NUM_EPOCHS+1), [m['text_mse'] for m in val_metrics_history], 'b-', label='Text MSE')
        plt.plot(range(1, NUM_EPOCHS+1), [m['fusion_mse'] for m in val_metrics_history], 'r-', label='Fusion MSE')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('Validation MSE by Modality')
        plt.legend()
        plt.grid(True)
        
        # Accuracy plot
        plt.subplot(2, 2, 3)
        plt.plot(range(1, NUM_EPOCHS+1), [m['accuracy'] for m in train_metrics_history], 'b-', label='Training Accuracy')
        plt.plot(range(1, NUM_EPOCHS+1), [m['accuracy'] for m in val_metrics_history], 'r-', label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Attention weights plot
        plt.subplot(2, 2, 4)
        plt.plot(range(1, NUM_EPOCHS+1), [m['avg_audio_weight'] for m in val_metrics_history], 'g-', label='Audio Weight')
        plt.plot(range(1, NUM_EPOCHS+1), [m['avg_text_weight'] for m in val_metrics_history], 'b-', label='Text Weight')
        plt.xlabel('Epoch')
        plt.ylabel('Average Attention Weight')
        plt.title('Fusion Model Attention Weights')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        print("Training history plots saved as 'training_history.png'")
    except Exception as e:
        print(f"Could not create training history plots: {e}")
    
    print("\nEmotion recognition demo completed!")
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

if __name__ == "__main__":
    main() 