#!/usr/bin/env python3
"""
Modal script to train and compare different model architectures for EMOD project.
Trains each model for 20 epochs on H100 GPUs with detailed per-epoch logging.
"""

import modal
import os
import json
from pathlib import Path
import argparse

# Local paths for data and code
LOCAL_DATA_PATH = str(Path("./IEMOCAP_Final.csv").absolute())
LOCAL_CODE_PATH = str(Path("./emod.py").absolute())

# Define model architectures to compare
TEXT_MODELS = [
    "roberta-base",              # Default baseline
    "microsoft/deberta-v3-base", # Replacing BERT
    "google/electra-base-discriminator", # Adding ELECTRA
    "distilbert-base-uncased",   # Smaller, faster alternative
    "xlnet-base-cased",          # Alternative architecture
    "albert-base-v2"             # Parameter-efficient model
]

AUDIO_FEATURES = [
    "mfcc",                      # Traditional MFCCs
    "spectrogram",               # Time-frequency representation
    "prosodic",                  # Hand-crafted features like pitch, energy
    "wav2vec",                   # Pre-trained speech model embeddings
]

FUSION_TYPES = [
    "early",                     # Concatenate features before processing
    "late",                      # Process separately, then combine
    "hybrid",                    # Combination of early and late fusion
    "attention"                  # Cross-modal attention mechanism
]

ML_CLASSIFIERS = [
    "random_forest",
    "gradient_boosting",
    "svm",
    "logistic_regression",
    "mlp"
]

# Define the Modal image with all required dependencies
image = (
    modal.Image.debian_slim()
    .pip_install([
        "torch",
        "transformers",
        "pandas",
        "numpy",
        "scikit-learn", 
        "matplotlib",
        "tqdm",
        "librosa",
        "seaborn",
        "wandb",
        "huggingface_hub",
        "tiktoken",
        "sentencepiece",
    ])
    .run_commands([
        # Install ffmpeg for audio processing
        "apt-get update && apt-get install -y ffmpeg"
    ])
    # Add local files to the container
    .add_local_file(LOCAL_DATA_PATH, "/root/IEMOCAP_Final.csv")
    .add_local_file(LOCAL_CODE_PATH, "/root/emod.py")
)

# Create Modal app
app = modal.App("emod-model-comparison", image=image)

# Create persistent volume for storing results
volume = modal.Volume.from_name("emod-results-vol", create_if_missing=True)
VOLUME_PATH = "/root/results"

@app.function(
    gpu="H100",
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60 * 8  # 8 hour timeout
)
def train_text_model(model_name, num_epochs=20):
    """Train a text-only model for VAD prediction with the specified architecture"""
    import os
    import sys
    import torch
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from transformers import AutoModel, AutoTokenizer
    import json
    from datetime import datetime
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from tqdm import tqdm
    
    # Convert num_epochs to int if it's a string
    num_epochs = int(num_epochs)
    
    # Set up environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(VOLUME_PATH, f"text_model_{model_name.replace('-', '_')}_{timestamp}")
    log_dir = os.path.join(model_dir, "logs")
    model_save_dir = os.path.join(model_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = pd.read_csv("/root/IEMOCAP_Final.csv")
    
    # Basic preprocessing
    X_text = df['Transcript'].values
    
    # Parse VAD values from string arrays and compute mean
    def parse_array_str(array_str):
        try:
            # Handle array-like strings: "[3, 4]" -> [3, 4]
            if isinstance(array_str, str) and '[' in array_str:
                values = [float(x.strip()) for x in array_str.strip('[]').split(',')]
                return sum(values) / len(values)  # Return mean value
            else:
                return float(array_str)
        except:
            # Default to neutral value if parsing fails
            return 3.0  # Middle of scale 1-5
    
    # Apply parsing to VAD columns
    df['Valence_parsed'] = df['Valence'].apply(parse_array_str)
    df['Arousal_parsed'] = df['Arousal'].apply(parse_array_str)
    df['Dominance_parsed'] = df['Dominance'].apply(parse_array_str)
    
    # Create VAD array with parsed values
    y_vad = df[['Valence_parsed', 'Arousal_parsed', 'Dominance_parsed']].values.astype(np.float32)
    
    # Map emotions to 4 standard classes: happy, angry, sad, neutral
    emotion_map = {
        'Happiness': 'happy',
        'Excited': 'happy',
        'Surprise': 'happy',
        'Anger': 'angry', 
        'Frustration': 'angry',
        'Disgust': 'angry',
        'Sadness': 'sad',
        'Fear': 'sad',
        'Neutral state': 'neutral',
        'Other': 'neutral'
    }
    
    # Apply emotion mapping - default to neutral if not found
    df['Mapped_Emotion'] = df['Major_emotion'].apply(lambda x: emotion_map.get(x.strip(), 'neutral') if isinstance(x, str) else 'neutral')
    y_emotion = df['Mapped_Emotion'].values
    
    # Split data (indices first)
    indices = np.arange(len(X_text))
    train_indices, temp_indices = train_test_split(indices, test_size=0.3, random_state=42)
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

    # Apply indices to all data splits
    X_train, X_val, X_test = X_text[train_indices], X_text[val_indices], X_text[test_indices]
    y_train_vad, y_val_vad, y_test_vad = y_vad[train_indices], y_vad[val_indices], y_vad[test_indices]
    y_train_emotion, y_val_emotion, y_test_emotion = y_emotion[train_indices], y_emotion[val_indices], y_emotion[test_indices]

    print(f"Data split completed: {len(X_train)} train, {len(X_val)} validation, {len(X_test)} test samples")
    
    # Define text dataset class
    class TextVADDataset(Dataset):
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
            
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            return {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "vad_values": torch.tensor(vad, dtype=torch.float)
            }
    
    # Define model architecture for VAD prediction
    class TextVADModel(nn.Module):
        def __init__(self, model_name):
            super(TextVADModel, self).__init__()
            self.text_encoder = AutoModel.from_pretrained(model_name)
            self.dropout = nn.Dropout(0.1)
            
            # Get hidden dimension from the model config
            hidden_dim = self.text_encoder.config.hidden_size
            
            self.regressor = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 3)  # 3 outputs for VAD
            )
            
        def forward(self, input_ids, attention_mask):
            # Get embeddings from the text encoder
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            
            # Use the [CLS] token embedding (first token)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            cls_embedding = self.dropout(cls_embedding)
            
            # Predict VAD values
            vad_pred = self.regressor(cls_embedding)
            return vad_pred
    
    # Initialize tokenizer and model
    print(f"Initializing {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TextVADModel(model_name).to(device)
    
    # Create datasets and dataloaders
    batch_size = 16
    train_dataset = TextVADDataset(X_train, y_train_vad, tokenizer)
    val_dataset = TextVADDataset(X_val, y_val_vad, tokenizer)
    test_dataset = TextVADDataset(X_test, y_test_vad, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Define optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.MSELoss()
    
    # Training loop with detailed logging
    print(f"Starting training for {num_epochs} epochs...")
    
    # Initialize logs
    training_log = {
        "model_name": model_name,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "epoch_logs": []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        epoch_log = {"epoch": epoch + 1, "train_steps": [], "val_loss": None}
        
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            vad_values = batch["vad_values"].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, vad_values)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Log training step
            step_loss = loss.item()
            train_loss += step_loss
            
            # Log every 10 steps
            if (i + 1) % 10 == 0:
                step_log = {
                    "step": i + 1,
                    "loss": step_loss
                }
                epoch_log["train_steps"].append(step_log)
                
                print(f"Epoch {epoch+1}, Step {i+1}/{len(train_loader)}, Loss: {step_loss:.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                vad_values = batch["vad_values"].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, vad_values)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        epoch_log["train_loss"] = avg_train_loss
        epoch_log["val_loss"] = avg_val_loss
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(model_save_dir, "best_model.pt"))
            print(f"Saved best model at epoch {epoch+1}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(model_save_dir, f"model_epoch_{epoch+1}.pt"))
        
        # Add epoch log to training log
        training_log["epoch_logs"].append(epoch_log)
        
        # Save the training log after each epoch
        with open(os.path.join(log_dir, "training_log.json"), 'w') as f:
            json.dump(training_log, f, indent=2)
    
    # Final evaluation on test set
    model.eval()
    test_loss = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            vad_values = batch["vad_values"].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, vad_values)
            test_loss += loss.item()
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(vad_values.cpu().numpy())
    
    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")
    
    # Calculate metrics for each dimension
    all_preds_test = np.vstack(all_preds)
    all_targets_test = np.vstack(all_targets)
    
    # Generate predictions for the training set (needed for ML classifier training)
    print("\nGenerating VAD predictions for training set...")
    all_preds_train = []
    model.eval() # Ensure model is in eval mode
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Predicting on Train Set"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask)
            all_preds_train.append(outputs.cpu().numpy())
    all_preds_train = np.vstack(all_preds_train)
    
    # Ensure predictions match original training set size (handle potential batch drop)
    if len(all_preds_train) != len(y_train_emotion):
       print(f"Warning: Mismatch in train prediction size ({len(all_preds_train)}) and labels ({len(y_train_emotion)}). Truncating labels.")
       y_train_emotion_matched = y_train_emotion[:len(all_preds_train)]
    else:
       y_train_emotion_matched = y_train_emotion

    # Ensure test predictions match original test set size
    if len(all_preds_test) != len(y_test_emotion):
       print(f"Warning: Mismatch in test prediction size ({len(all_preds_test)}) and labels ({len(y_test_emotion)}). Truncating labels.")
       y_test_emotion_matched = y_test_emotion[:len(all_preds_test)]
    else:
       y_test_emotion_matched = y_test_emotion

    # Save VAD predictions and true emotion labels for ML classifier evaluation
    vad_preds_save_path = os.path.join(model_dir, "vad_predictions.npz")
    np.savez_compressed(
        vad_preds_save_path, 
        train_preds=all_preds_train,
        test_preds=all_preds_test,
        train_emotions=y_train_emotion_matched, 
        test_emotions=y_test_emotion_matched
    )
    print(f"VAD predictions saved to {vad_preds_save_path}")

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    metrics = {"mse": [], "rmse": [], "mae": [], "r2": []}
    
    for i, dim in enumerate(["Valence", "Arousal", "Dominance"]):
        # Use the original VAD targets for metric calculation
        # Ensure we compare slices with the same shape [N] vs [N]
        y_true_slice = y_test_vad[:len(all_preds_test), i]
        y_pred_slice = all_preds_test[:, i]
        
        mse = mean_squared_error(y_true_slice, y_pred_slice) 
        metrics["mse"].append(float(mse))
        metrics["rmse"].append(float(np.sqrt(mse)))
        metrics["mae"].append(float(mean_absolute_error(y_true_slice, y_pred_slice)))
        metrics["r2"].append(float(r2_score(y_true_slice, y_pred_slice)))
        
        print(f"{dim} - MSE: {mse:.4f}, RMSE: {metrics['rmse'][-1]:.4f}, MAE: {metrics['mae'][-1]:.4f}, R²: {metrics['r2'][-1]:.4f}")
    
    # Prepare final results summary dictionary (excluding large arrays for return)
    results_summary = {
        "model_name": model_name,
        "config": {
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": optimizer.param_groups[0]['lr'], # Get actual LR
        },
        "final_metrics": {
            "Test Loss": avg_test_loss,
            "Valence": {"MSE": metrics["mse"][0], "RMSE": metrics["rmse"][0], "MAE": metrics["mae"][0], "R2": metrics["r2"][0]},
            "Arousal": {"MSE": metrics["mse"][1], "RMSE": metrics["rmse"][1], "MAE": metrics["mae"][1], "R2": metrics["r2"][1]},
            "Dominance": {"MSE": metrics["mse"][2], "RMSE": metrics["rmse"][2], "MAE": metrics["mae"][2], "R2": metrics["r2"][2]}
        },
        "best_val_loss": best_val_loss
    }

    # Add training log path and prediction path to summary
    training_log_path = os.path.join(log_dir, "training_log.json")
    final_results_path = os.path.join(log_dir, "final_results.json")
    results_summary["training_log_path"] = training_log_path
    results_summary["vad_predictions_path"] = vad_preds_save_path

    # Save final results summary
    with open(final_results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Save detailed training log (includes epoch/step details)
    training_log["final_metrics"] = results_summary["final_metrics"] # Add final metrics here too
    with open(training_log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    volume.commit()
    print(f"Results summary saved to {final_results_path}")
    print(f"Detailed training log saved to {training_log_path}")

    # --- Trigger ML Classifier Evaluation --- 
    print("\nTriggering ML classifier evaluation...")
    ml_eval_results = evaluate_ml_classifiers.remote(model_dir) 
    print("ML classifier evaluation job submitted.")
    # --- End Trigger --- 

    # Return the full training log dictionary
    return training_log

@app.function(
    gpu="H100",
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60 * 8  # 8 hour timeout
)
def train_multimodal_model(text_model_name, audio_feature_type, fusion_type, num_epochs=20):
    """Train a multimodal model with specified text encoder, audio features, and fusion strategy"""
    import os
    import sys
    import torch
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from transformers import AutoModel, AutoTokenizer
    import json
    from datetime import datetime
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from tqdm import tqdm
    import librosa
    
    # Convert num_epochs to int if it's a string
    num_epochs = int(num_epochs)
    
    # Set up environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(VOLUME_PATH, f"multimodal_{text_model_name.replace('-', '_')}_{audio_feature_type}_{fusion_type}_{timestamp}")
    log_dir = os.path.join(model_dir, "logs")
    model_save_dir = os.path.join(model_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = pd.read_csv("/root/IEMOCAP_Final.csv")
    
    # Basic preprocessing
    X_text = df['Transcript'].values
    
    # Parse VAD values from string arrays and compute mean
    def parse_array_str(array_str):
        try:
            # Handle array-like strings: "[3, 4]" -> [3, 4]
            if isinstance(array_str, str) and '[' in array_str:
                values = [float(x.strip()) for x in array_str.strip('[]').split(',')]
                return sum(values) / len(values)  # Return mean value
            else:
                return float(array_str)
        except:
            # Default to neutral value if parsing fails
            return 3.0  # Middle of scale 1-5
    
    # Apply parsing to VAD columns
    df['Valence_parsed'] = df['Valence'].apply(parse_array_str)
    df['Arousal_parsed'] = df['Arousal'].apply(parse_array_str)
    df['Dominance_parsed'] = df['Dominance'].apply(parse_array_str)
    
    # Create VAD array with parsed values
    y_vad = df[['Valence_parsed', 'Arousal_parsed', 'Dominance_parsed']].values.astype(np.float32)
    
    # Map emotions to 4 standard classes: happy, angry, sad, neutral
    emotion_map = {
        'Happiness': 'happy',
        'Excited': 'happy',
        'Surprise': 'happy',
        'Anger': 'angry', 
        'Frustration': 'angry',
        'Disgust': 'angry',
        'Sadness': 'sad',
        'Fear': 'sad',
        'Neutral state': 'neutral',
        'Other': 'neutral'
    }
    
    # Apply emotion mapping - default to neutral if not found
    df['Mapped_Emotion'] = df['Major_emotion'].apply(lambda x: emotion_map.get(x.strip(), 'neutral') if isinstance(x, str) else 'neutral')
    y_emotion = df['Mapped_Emotion'].values
    
    # Define audio feature dimensions
    feature_dims = {
        "mfcc": 40,
        "spectrogram": 128,
        "prosodic": 20, # Placeholder dim, replace if actual features used
        "wav2vec": 256 # Placeholder dim, replace if actual features used
    }

    # Generate synthetic audio features for demonstration
    print(f"Generating {audio_feature_type} audio features...")
    audio_features = np.random.randn(len(df), feature_dims[audio_feature_type])
    
    # Split data (indices first)
    indices = np.arange(len(X_text))
    train_indices, temp_indices = train_test_split(indices, test_size=0.3, random_state=42)
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

    # Apply indices to all data splits
    X_train, X_val, X_test = X_text[train_indices], X_text[val_indices], X_text[test_indices]
    y_train_vad, y_val_vad, y_test_vad = y_vad[train_indices], y_vad[val_indices], y_vad[test_indices]
    y_train_emotion, y_val_emotion, y_test_emotion = y_emotion[train_indices], y_emotion[val_indices], y_emotion[test_indices]

    print(f"Data split completed: {len(X_train)} train, {len(X_val)} validation, {len(X_test)} test samples")
    
    # Define multimodal dataset class
    class MultimodalVADDataset(Dataset):
        def __init__(self, texts, audio_features, vad_values, tokenizer, max_length=128):
            self.texts = texts
            self.audio_features = audio_features
            self.vad_values = vad_values
            self.tokenizer = tokenizer
            self.max_length = max_length
            
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = self.texts[idx]
            audio = self.audio_features[idx]
            vad = self.vad_values[idx]
            
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            return {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "audio_features": torch.tensor(audio, dtype=torch.float),
                "vad_values": torch.tensor(vad, dtype=torch.float)
            }
    
    # Define different fusion model architectures
    
    # Early Fusion Model
    class EarlyFusionModel(nn.Module):
        def __init__(self, text_model_name, audio_dim):
            super(EarlyFusionModel, self).__init__()
            self.text_encoder = AutoModel.from_pretrained(text_model_name)
            hidden_dim = self.text_encoder.config.hidden_size
            
            self.fusion_network = nn.Sequential(
                nn.Linear(hidden_dim + audio_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 3)  # 3 outputs for VAD
            )
            
        def forward(self, input_ids, attention_mask, audio_features):
            # Get text features
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_features = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            
            # Concatenate text and audio features
            fused_features = torch.cat([text_features, audio_features], dim=1)
            
            # Predict VAD values
            vad_pred = self.fusion_network(fused_features)
            return vad_pred
    
    # Late Fusion Model
    class LateFusionModel(nn.Module):
        def __init__(self, text_model_name, audio_dim):
            super(LateFusionModel, self).__init__()
            self.text_encoder = AutoModel.from_pretrained(text_model_name)
            hidden_dim = self.text_encoder.config.hidden_size
            
            # Text processing branch
            self.text_network = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64)
            )
            
            # Audio processing branch
            self.audio_network = nn.Sequential(
                nn.Linear(audio_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32)
            )
            
            # Fusion layer
            self.fusion_network = nn.Sequential(
                nn.Linear(64 + 32, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 3)  # 3 outputs for VAD
            )
            
        def forward(self, input_ids, attention_mask, audio_features):
            # Get text features
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_features = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            
            # Process text and audio separately
            text_processed = self.text_network(text_features)
            audio_processed = self.audio_network(audio_features)
            
            # Concatenate processed features
            fused_features = torch.cat([text_processed, audio_processed], dim=1)
            
            # Predict VAD values
            vad_pred = self.fusion_network(fused_features)
            return vad_pred
    
    # Hybrid Fusion Model
    class HybridFusionModel(nn.Module):
        def __init__(self, text_model_name, audio_dim):
            super(HybridFusionModel, self).__init__()
            self.text_encoder = AutoModel.from_pretrained(text_model_name)
            hidden_dim = self.text_encoder.config.hidden_size
            
            # Early fusion branch
            self.early_fusion = nn.Sequential(
                nn.Linear(hidden_dim + audio_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            
            # Late fusion branches
            self.text_network = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            
            self.audio_network = nn.Sequential(
                nn.Linear(audio_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            
            self.late_fusion = nn.Sequential(
                nn.Linear(128 + 64, 128),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            
            # Final prediction layer
            self.output_layer = nn.Sequential(
                nn.Linear(128 + 128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 3)  # 3 outputs for VAD
            )
            
        def forward(self, input_ids, attention_mask, audio_features):
            # Get text features
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_features = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            
            # Early fusion
            early_concat = torch.cat([text_features, audio_features], dim=1)
            early_output = self.early_fusion(early_concat)
            
            # Late fusion
            text_processed = self.text_network(text_features)
            audio_processed = self.audio_network(audio_features)
            late_concat = torch.cat([text_processed, audio_processed], dim=1)
            late_output = self.late_fusion(late_concat)
            
            # Combine early and late fusion
            combined = torch.cat([early_output, late_output], dim=1)
            vad_pred = self.output_layer(combined)
            
            return vad_pred
    
    # Attention Fusion Model
    class AttentionFusionModel(nn.Module):
        def __init__(self, text_model_name, audio_dim):
            super(AttentionFusionModel, self).__init__()
            self.text_encoder = AutoModel.from_pretrained(text_model_name)
            hidden_dim = self.text_encoder.config.hidden_size
            
            # Text processing
            self.text_network = nn.Linear(hidden_dim, 128)
            
            # Audio processing
            self.audio_network = nn.Linear(audio_dim, 128)
            
            # Attention mechanism
            self.query_text = nn.Linear(128, 64)
            self.key_audio = nn.Linear(128, 64)
            self.value_audio = nn.Linear(128, 64)
            
            # Final layers
            self.fusion_network = nn.Sequential(
                nn.Linear(128 + 64, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 3)  # 3 outputs for VAD
            )
            
        def forward(self, input_ids, attention_mask, audio_features):
            # Get text features
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_features = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            
            # Process text and audio
            text_processed = self.text_network(text_features)  # [batch, 128]
            audio_processed = self.audio_network(audio_features)  # [batch, 128]
            
            # Calculate attention scores
            queries = self.query_text(text_processed)  # [batch, 64]
            keys = self.key_audio(audio_processed)  # [batch, 64]
            values = self.value_audio(audio_processed)  # [batch, 64]
            
            # Attention weights
            attention_scores = torch.matmul(queries.unsqueeze(1), keys.unsqueeze(2)).squeeze(1)  # [batch, 1]
            attention_weights = torch.sigmoid(attention_scores)  # [batch, 1]
            
            # Apply attention
            audio_attended = attention_weights * values  # [batch, 64]
            
            # Combine features
            combined = torch.cat([text_processed, audio_attended], dim=1)  # [batch, 128+64]
            
            # Final prediction
            vad_pred = self.fusion_network(combined)
            
            return vad_pred
    
    # Select model architecture based on fusion type
    print(f"Initializing {fusion_type} fusion model with {text_model_name} and {audio_feature_type} features...")
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    audio_dim = feature_dims[audio_feature_type]
    
    if fusion_type == "early":
        model = EarlyFusionModel(text_model_name, audio_dim)
    elif fusion_type == "late":
        model = LateFusionModel(text_model_name, audio_dim)
    elif fusion_type == "hybrid":
        model = HybridFusionModel(text_model_name, audio_dim)
    elif fusion_type == "attention":
        model = AttentionFusionModel(text_model_name, audio_dim)
    else:
        raise ValueError(f"Unsupported fusion type: {fusion_type}")
    
    model = model.to(device)
    
    # Create datasets and dataloaders
    batch_size = 16
    train_dataset = MultimodalVADDataset(X_train, audio_features[train_indices], y_train_vad, tokenizer)
    val_dataset = MultimodalVADDataset(X_val, audio_features[val_indices], y_val_vad, tokenizer)
    test_dataset = MultimodalVADDataset(X_test, audio_features[test_indices], y_test_vad, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Define optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.MSELoss()
    
    # Training loop with detailed logging
    print(f"Starting training for {num_epochs} epochs...")
    
    # Initialize logs
    training_log = {
        "text_model": text_model_name,
        "audio_feature": audio_feature_type,
        "fusion_type": fusion_type,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "epoch_logs": []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        epoch_log = {"epoch": epoch + 1, "train_steps": [], "val_loss": None}
        
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            audio_features = batch["audio_features"].to(device)
            vad_values = batch["vad_values"].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, audio_features)
            loss = criterion(outputs, vad_values)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Log training step
            step_loss = loss.item()
            train_loss += step_loss
            
            # Log every 10 steps
            if (i + 1) % 10 == 0:
                step_log = {
                    "step": i + 1,
                    "loss": step_loss
                }
                epoch_log["train_steps"].append(step_log)
                
                print(f"Epoch {epoch+1}, Step {i+1}/{len(train_loader)}, Loss: {step_loss:.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                audio_features = batch["audio_features"].to(device)
                vad_values = batch["vad_values"].to(device)
                
                outputs = model(input_ids, attention_mask, audio_features)
                loss = criterion(outputs, vad_values)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        epoch_log["train_loss"] = avg_train_loss
        epoch_log["val_loss"] = avg_val_loss
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(model_save_dir, "best_model.pt"))
            print(f"Saved best model at epoch {epoch+1}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(model_save_dir, f"model_epoch_{epoch+1}.pt"))
        
        # Add epoch log to training log
        training_log["epoch_logs"].append(epoch_log)
        
        # Save the training log after each epoch
        with open(os.path.join(log_dir, "training_log.json"), 'w') as f:
            json.dump(training_log, f, indent=2)
    
    # Final evaluation on test set
    model.eval()
    test_loss = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            audio_features = batch["audio_features"].to(device)
            vad_values = batch["vad_values"].to(device)
            
            outputs = model(input_ids, attention_mask, audio_features)
            loss = criterion(outputs, vad_values)
            test_loss += loss.item()
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(vad_values.cpu().numpy())
    
    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")
    
    # Calculate metrics for each dimension
    all_preds_test = np.vstack(all_preds)
    all_targets_test = np.vstack(all_targets)
    
    # Generate predictions for the training set (needed for ML classifier training)
    print("\nGenerating VAD predictions for training set...")
    all_preds_train = []
    model.eval() # Ensure model is in eval mode
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Predicting on Train Set"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            audio_features = batch["audio_features"].to(device)
            outputs = model(input_ids, attention_mask, audio_features)
            all_preds_train.append(outputs.cpu().numpy())
    all_preds_train = np.vstack(all_preds_train)
    
    # Ensure predictions match original training set size (handle potential batch drop)
    if len(all_preds_train) != len(y_train_emotion):
       print(f"Warning: Mismatch in train prediction size ({len(all_preds_train)}) and labels ({len(y_train_emotion)}). Truncating labels.")
       y_train_emotion_matched = y_train_emotion[:len(all_preds_train)]
    else:
       y_train_emotion_matched = y_train_emotion

    # Ensure test predictions match original test set size
    if len(all_preds_test) != len(y_test_emotion):
       print(f"Warning: Mismatch in test prediction size ({len(all_preds_test)}) and labels ({len(y_test_emotion)}). Truncating labels.")
       y_test_emotion_matched = y_test_emotion[:len(all_preds_test)]
    else:
       y_test_emotion_matched = y_test_emotion

    # Save VAD predictions and true emotion labels for ML classifier evaluation
    vad_preds_save_path = os.path.join(model_dir, "vad_predictions.npz")
    np.savez_compressed(
        vad_preds_save_path, 
        train_preds=all_preds_train,
        test_preds=all_preds_test,
        train_emotions=y_train_emotion_matched, 
        test_emotions=y_test_emotion_matched
    )
    print(f"VAD predictions saved to {vad_preds_save_path}")

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    metrics = {"mse": [], "rmse": [], "mae": [], "r2": []}
    
    for i, dim in enumerate(["Valence", "Arousal", "Dominance"]):
        # Use the original VAD targets for metric calculation
        # Ensure we compare slices with the same shape [N] vs [N]
        y_true_slice = y_test_vad[:len(all_preds_test), i]
        y_pred_slice = all_preds_test[:, i]
        
        mse = mean_squared_error(y_true_slice, y_pred_slice) 
        metrics["mse"].append(float(mse))
        metrics["rmse"].append(float(np.sqrt(mse)))
        metrics["mae"].append(float(mean_absolute_error(y_true_slice, y_pred_slice)))
        metrics["r2"].append(float(r2_score(y_true_slice, y_pred_slice)))
        
        print(f"{dim} - MSE: {mse:.4f}, RMSE: {metrics['rmse'][-1]:.4f}, MAE: {metrics['mae'][-1]:.4f}, R²: {metrics['r2'][-1]:.4f}")
    
    # Prepare final results summary dictionary (excluding large arrays for return)
    results_summary = {
        "text_model": text_model_name,
        "audio_feature": audio_feature_type,
        "fusion_type": fusion_type,
        "config": {
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": optimizer.param_groups[0]['lr'], # Get actual LR
        },
        "final_metrics": {
            "Test Loss": avg_test_loss,
            "Valence": {"MSE": metrics["mse"][0], "RMSE": metrics["rmse"][0], "MAE": metrics["mae"][0], "R2": metrics["r2"][0]},
            "Arousal": {"MSE": metrics["mse"][1], "RMSE": metrics["rmse"][1], "MAE": metrics["mae"][1], "R2": metrics["r2"][1]},
            "Dominance": {"MSE": metrics["mse"][2], "RMSE": metrics["rmse"][2], "MAE": metrics["mae"][2], "R2": metrics["r2"][2]}
        },
        "best_val_loss": best_val_loss
    }

    # Add training log path and prediction path to summary
    training_log_path = os.path.join(log_dir, "training_log.json")
    final_results_path = os.path.join(log_dir, "final_results.json")
    results_summary["training_log_path"] = training_log_path
    results_summary["vad_predictions_path"] = vad_preds_save_path

    # Save final results summary
    with open(final_results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Save detailed training log (includes epoch/step details)
    training_log["final_metrics"] = results_summary["final_metrics"] # Add final metrics here too
    with open(training_log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    volume.commit()
    print(f"Results summary saved to {final_results_path}")
    print(f"Detailed training log saved to {training_log_path}")

    # --- Trigger ML Classifier Evaluation --- 
    print("\nTriggering ML classifier evaluation...")
    ml_eval_results = evaluate_ml_classifiers.remote(model_dir) 
    print("ML classifier evaluation job submitted.")
    # --- End Trigger --- 

    # Return the full training log dictionary
    return training_log

@app.function(
    volumes={VOLUME_PATH: volume}
)
def evaluate_ml_classifiers(vad_results_dir, num_classifiers=5):
    """Evaluate different ML classifiers on the VAD-to-emotion classification task using predictions from a specific VAD model run."""
    import os
    import pandas as pd
    import numpy as np
    import json
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    from datetime import datetime
    
    # Define classifiers
    classifiers = {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
        "svm": SVC(kernel='rbf', probability=True, random_state=42),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
    }
    
    # --- Updated Data Loading --- 
    vad_preds_path = os.path.join(vad_results_dir, "vad_predictions.npz")
    print(f"Loading VAD predictions from {vad_preds_path}...")
    try:
        vad_data = np.load(vad_preds_path, allow_pickle=True)
        X_train = vad_data["train_preds"]
        X_test = vad_data["test_preds"]
        y_train = vad_data["train_emotions"]
        y_test = vad_data["test_emotions"]
        print(f"Data loaded: {len(X_train)} train samples, {len(X_test)} test samples")
    except Exception as e:
        print(f"Error loading VAD predictions from {vad_preds_path}: {e}")
        return { "error": str(e) } # Return error if loading fails
    # --- End Update --- 

    # Evaluate each classifier
    results = []
    
    for name, clf in list(classifiers.items())[:num_classifiers]:
        print(f"Evaluating {name}...")
        
        # Train classifier
        clf.fit(X_train, y_train)
        
        # Predict
        y_pred = clf.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        weighted_f1 = f1_score(y_test, y_pred, average='weighted')
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        report = classification_report(y_test, y_pred, output_dict=True)
        
        print(f"{name}: Accuracy={accuracy:.4f}, Weighted F1={weighted_f1:.4f}, Macro F1={macro_f1:.4f}")
        
        # Save results
        result = {
            "classifier": name,
            "accuracy": float(accuracy),
            "weighted_f1": float(weighted_f1),
            "macro_f1": float(macro_f1),
            "classification_report": report
        }
        results.append(result)
    
    # --- Updated Result Saving --- 
    ml_results_path_json = os.path.join(vad_results_dir, "ml_classifier_results.json")
    ml_results_path_csv = os.path.join(vad_results_dir, "ml_classifier_results.csv")
    
    # Save overall results as JSON
    with open(ml_results_path_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Also save as CSV for easy viewing
    df_results = pd.DataFrame([{
        "classifier": r["classifier"],
        "accuracy": r["accuracy"],
        "weighted_f1": r["weighted_f1"],
        "macro_f1": r["macro_f1"]
    } for r in results])
    
    df_results.to_csv(ml_results_path_csv, index=False)
    
    volume.commit() # Commit results to volume
    print(f"ML Classifier results saved to {vad_results_dir}")
    # --- End Update --- 
    
    return {
        "ml_results_dir": vad_results_dir,
        "best_classifier": max(results, key=lambda x: x["weighted_f1"])["classifier"],
        "best_accuracy": max(results, key=lambda x: x["accuracy"])["accuracy"],
        "best_weighted_f1": max(results, key=lambda x: x["weighted_f1"])["weighted_f1"]
    }

@app.function(
    volumes={VOLUME_PATH: volume}
)
def initialize_volume():
    """Initialize the volume and display its contents"""
    import os
    
    # Make sure directory exists
    os.makedirs(VOLUME_PATH, exist_ok=True)
    
    # List contents
    print(f"Contents of {VOLUME_PATH}:")
    for item in os.listdir(VOLUME_PATH):
        print(f"  - {item}")
    
    return f"Volume initialized at {VOLUME_PATH}"

@app.local_entrypoint()
def main():
    """Main entry point for running the model comparison experiment"""
    parser = argparse.ArgumentParser(description="Run model comparison experiments on Modal")
    parser.add_argument("--experiment", type=str, choices=["text", "multimodal"], required=True,
                        help="Type of experiment to run: text-only VAD or multimodal VAD (both trigger ML eval)")
    parser.add_argument("--text_model", type=str, default="roberta-base",
                        help="Text model to use (for text and multimodal experiments)")
    parser.add_argument("--audio_feature", type=str, default="mfcc",
                        help="Audio feature extraction method (for multimodal experiment)")
    parser.add_argument("--fusion_type", type=str, default="early",
                        help="Fusion strategy (for multimodal experiment)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs to train")
    args = parser.parse_args()
    
    # Initialize volume
    print("Initializing volume...")
    volume_status = initialize_volume.remote()
    print(volume_status)
    
    if args.experiment == "text":
        # Run text model experiment
        print(f"Starting text model experiment with {args.text_model}...")
        # Note: ML eval is triggered internally now
        result = train_text_model.remote(args.text_model, args.epochs)
        print(f"Text model VAD training submitted. Check logs for progress and final results.")
        
    elif args.experiment == "multimodal":
        # Run multimodal experiment
        print(f"Starting multimodal experiment with {args.text_model}, {args.audio_feature} features, and {args.fusion_type} fusion...")
        # Note: ML eval is triggered internally now
        result = train_multimodal_model.remote(args.text_model, args.audio_feature, args.fusion_type, args.epochs)
        print(f"Multimodal VAD training submitted. Check logs for progress and final results.")
        
    # The actual results (including ML eval) will be printed in the logs of the remote functions.
    # The 'result' variable here might only contain the initial summary if the remote call finishes quickly.
    print("\nMain script finished submission. Monitor Modal logs for detailed results.")

if __name__ == "__main__":
    main() 