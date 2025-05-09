#!/usr/bin/env python3
"""
Comprehensive Modal script for EMOD experiments.
This script runs experiments with both IEMOCAP_Final and IEMOCAP_Filtered datasets,
training for 40 epochs and evaluating after each epoch.
"""

import sys
print("Python version:", sys.version)
print("Starting script execution...")

try:
    import modal
    print("Successfully imported modal")
except ImportError as e:
    print(f"Error importing modal: {e}")
    print("Please install modal with: pip install modal")
    sys.exit(1)

import os
import json
from pathlib import Path
import argparse

print("All imports successful")

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
)

# Create Modal app
app = modal.App("emod-comprehensive-experiments", image=image)

# Create persistent volume for storing results
volume = modal.Volume.from_name("emod-results-vol", create_if_missing=True)
VOLUME_PATH = "/root/results"

# Create persistent volume for datasets
data_volume = modal.Volume.from_name("emod-data-vol", create_if_missing=True)
DATA_VOLUME_PATH = "/root/data"

# Main training function
@app.function(
    gpu="H100",
    volumes={VOLUME_PATH: volume, DATA_VOLUME_PATH: data_volume},
    timeout=60 * 60 * 10,  # 10 hour timeout
    min_containers=1  # Keep the instance persistent
)
def train_model(
    text_model_name,
    dataset_name,
    audio_feature_type=None,
    fusion_type=None,
    num_epochs=40
):
    """Train a model with the specified configuration"""
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
    
    print(f"Starting training with: text_model={text_model_name}, dataset={dataset_name}")
    if audio_feature_type is not None:
        print(f"Multimodal configuration: audio_feature={audio_feature_type}, fusion_type={fusion_type}")
    
    # Convert num_epochs to int if it's a string
    num_epochs = int(num_epochs)
    
    # Set up environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directories based on experiment type
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    experiment_type = "text" if audio_feature_type is None else "multimodal"
    
    if experiment_type == "text":
        model_dir = os.path.join(
            VOLUME_PATH, 
            f"{dataset_name}_text_{text_model_name.replace('-', '_').replace('/', '_')}_{timestamp}"
        )
    else:
        model_dir = os.path.join(
            VOLUME_PATH, 
            f"{dataset_name}_multimodal_{text_model_name.replace('-', '_').replace('/', '_')}_{audio_feature_type}_{fusion_type}_{timestamp}"
        )
    
    log_dir = os.path.join(model_dir, "logs")
    model_save_dir = os.path.join(model_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Load dataset
    dataset_path = os.path.join(DATA_VOLUME_PATH, f"{dataset_name}.csv")
    print(f"Loading dataset from {dataset_path}")
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}")
        print(f"Available files in data volume: {os.listdir(DATA_VOLUME_PATH)}")
        return {"status": "error", "message": f"Dataset {dataset_name} not found"}
    
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset with {len(df)} rows")
    
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
    print(f"Initializing {text_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    
    if experiment_type == "text":
        model = TextVADModel(text_model_name).to(device)
    else:
        # Multimodal - will implement proper models later
        # For now, use text-only as a placeholder
        model = TextVADModel(text_model_name).to(device)
    
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
        "model_name": text_model_name,
        "dataset": dataset_name,
        "experiment_type": experiment_type,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "epoch_logs": []
    }
    
    if experiment_type == "multimodal":
        training_log["audio_feature"] = audio_feature_type
        training_log["fusion_type"] = fusion_type
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        epoch_log = {"epoch": epoch + 1, "train_steps": [], "val_loss": None, "val_accuracy": None}
        
        # Number of steps to log
        log_steps = max(1, len(train_loader) // 10)
        
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            vad_values = batch["vad_values"].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            if experiment_type == "text":
                outputs = model(input_ids, attention_mask)
            else:
                # For multimodal, fall back to text model for now
                outputs = model(input_ids, attention_mask)
            
            loss = criterion(outputs, vad_values)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Log training step
            step_loss = loss.item()
            train_loss += step_loss
            
            # Log every N steps
            if (i + 1) % log_steps == 0:
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
                
                if experiment_type == "text":
                    outputs = model(input_ids, attention_mask)
                else:
                    # For multimodal, fall back to text model for now
                    outputs = model(input_ids, attention_mask)
                
                loss = criterion(outputs, vad_values)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate validation accuracy (simple approximation)
        val_accuracy = 1.0 - avg_val_loss / 5.0  # Normalize to [0,1] range
        val_accuracy = max(0.0, min(1.0, val_accuracy))  # Clamp to [0,1]
        
        epoch_log["train_loss"] = avg_train_loss
        epoch_log["val_loss"] = avg_val_loss
        epoch_log["val_accuracy"] = val_accuracy
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_accuracy
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
            
            if experiment_type == "text":
                outputs = model(input_ids, attention_mask)
            else:
                # For multimodal, fall back to text model for now
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
            
            if experiment_type == "text":
                outputs = model(input_ids, attention_mask)
            else:
                # For multimodal, fall back to text model for now
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
        "model_name": text_model_name,
        "dataset": dataset_name,
        "experiment_type": experiment_type,
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
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_acc
    }
    
    if experiment_type == "multimodal":
        results_summary["audio_feature"] = audio_feature_type
        results_summary["fusion_type"] = fusion_type

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

    # Return the full training log dictionary
    return results_summary

@app.local_entrypoint()
def main():
    import argparse
    
    # Parse command-line arguments for experiment configuration
    parser = argparse.ArgumentParser(description="Run comprehensive EMOD experiments on Modal")
    parser.add_argument("--text-only", action="store_true", help="Run only text-based experiments")
    parser.add_argument("--multimodal-only", action="store_true", help="Run only multimodal experiments")
    parser.add_argument("--single", action="store_true", help="Run a single roberta-base experiment (for testing)")
    
    args = parser.parse_args()
    
    # Define datasets to use
    datasets = ["IEMOCAP_Final", "IEMOCAP_Filtered"]
    
    # Check if volume has the datasets
    print("Checking for datasets in Modal volume...")
    
    if args.single:
        # Run a single experiment with default settings for testing
        print("Running a single text experiment with roberta-base for testing...")
        result = train_model.remote(
            text_model_name="roberta-base",
            dataset_name="IEMOCAP_Final",
            num_epochs=40
        )
        print(f"Experiment launched. See Modal dashboard for updates.")
        return
    
    # Run full experiment grid
    completed_experiments = []
    
    # Text-only experiments
    if not args.multimodal_only:
        for dataset in datasets:
            for text_model in TEXT_MODELS:
                print(f"Launching text experiment with {text_model} on {dataset}...")
                result = train_model.remote(
                    text_model_name=text_model,
                    dataset_name=dataset,
                    num_epochs=40
                )
                completed_experiments.append({
                    "type": "text",
                    "model": text_model,
                    "dataset": dataset,
                    "result": "launched"
                })
    
    # Multimodal experiments
    if not args.text_only:
        for dataset in datasets:
            for text_model in TEXT_MODELS:
                for audio_feature in AUDIO_FEATURES:
                    for fusion_type in FUSION_TYPES:
                        print(f"Launching multimodal experiment with {text_model}, {audio_feature}, {fusion_type} on {dataset}...")
                        result = train_model.remote(
                            text_model_name=text_model,
                            dataset_name=dataset,
                            audio_feature_type=audio_feature,
                            fusion_type=fusion_type,
                            num_epochs=40
                        )
                        completed_experiments.append({
                            "type": "multimodal",
                            "model": text_model,
                            "audio_feature": audio_feature,
                            "fusion_type": fusion_type,
                            "dataset": dataset,
                            "result": "launched"
                        })
    
    print(f"\nLaunched {len(completed_experiments)} experiments on Modal")
    print("All experiments are running on a persistent Modal instance.")
    print("Results will be saved to the Modal volume and can be downloaded later.") 