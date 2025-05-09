#!/usr/bin/env python3
"""
Script to run a single EMOD experiment on Modal.
This version is simplified to run without command-line arguments.
"""

import modal
import os
from pathlib import Path

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
app = modal.App("emod-single-experiment", image=image)

# Create persistent volume for storing results
volume = modal.Volume.from_name("emod-results-vol", create_if_missing=True)
VOLUME_PATH = "/root/results"

# Create persistent volume for datasets
data_volume = modal.Volume.from_name("emod-data-vol", create_if_missing=True)
DATA_VOLUME_PATH = "/root/data"

@app.function(
    gpu="H100",
    volumes={VOLUME_PATH: volume, DATA_VOLUME_PATH: data_volume},
    timeout=60 * 60 * 10,  # 10 hour timeout
    min_containers=1  # Keep the instance persistent
)
def run_experiment():
    """Run a single roberta-base experiment on IEMOCAP_Final dataset"""
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
    
    # Configuration settings
    text_model_name = "roberta-base"
    dataset_name = "IEMOCAP_Final"
    num_epochs = 40
    
    print(f"Starting experiment with {text_model_name} on {dataset_name} for {num_epochs} epochs")
    
    # Set up environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(VOLUME_PATH, f"test_run_{text_model_name.replace('-', '_')}_{timestamp}")
    log_dir = os.path.join(model_dir, "logs")
    model_save_dir = os.path.join(model_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Load dataset
    dataset_path = os.path.join(DATA_VOLUME_PATH, f"{dataset_name}.csv")
    print(f"Loading dataset from {dataset_path}")
    
    # List files in data directory
    print(f"Files in data directory: {os.listdir(DATA_VOLUME_PATH)}")
    
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}")
        return {"status": "error", "message": f"Dataset {dataset_name}.csv not found"}
    
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset with {len(df)} rows")
    
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
    
    # Extract text
    X_text = df['Transcript'].values
    
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
        "experiment_type": "text",
        "epochs": num_epochs,
        "batch_size": batch_size,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "epoch_logs": []
    }
    
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
        
        # Commit the volume after each epoch to save progress
        volume.commit()
    
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
    
    # Save VAD predictions and true emotion labels for ML classifier evaluation
    vad_preds_save_path = os.path.join(model_dir, "vad_predictions.npz")
    np.savez_compressed(
        vad_preds_save_path, 
        train_preds=all_preds_train,
        test_preds=all_preds_test,
        train_emotions=y_train_emotion[:len(all_preds_train)], 
        test_emotions=y_test_emotion[:len(all_preds_test)]
    )
    print(f"VAD predictions saved to {vad_preds_save_path}")

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    metrics = {"mse": [], "rmse": [], "mae": [], "r2": []}
    
    for i, dim in enumerate(["Valence", "Arousal", "Dominance"]):
        # Ensure we compare slices with the same shape [N] vs [N]
        y_true_slice = y_test_vad[:len(all_preds_test), i]
        y_pred_slice = all_preds_test[:, i]
        
        mse = mean_squared_error(y_true_slice, y_pred_slice) 
        metrics["mse"].append(float(mse))
        metrics["rmse"].append(float(np.sqrt(mse)))
        metrics["mae"].append(float(mean_absolute_error(y_true_slice, y_pred_slice)))
        metrics["r2"].append(float(r2_score(y_true_slice, y_pred_slice)))
        
        print(f"{dim} - MSE: {mse:.4f}, RMSE: {metrics['rmse'][-1]:.4f}, MAE: {metrics['mae'][-1]:.4f}, R²: {metrics['r2'][-1]:.4f}")
    
    # Prepare final results summary dictionary
    results_summary = {
        "model_name": text_model_name,
        "dataset": dataset_name,
        "experiment_type": "text",
        "config": {
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": optimizer.param_groups[0]['lr'],
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

    # Save final results summary
    final_results_path = os.path.join(log_dir, "final_results.json")
    with open(final_results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Save detailed training log (includes epoch/step details)
    training_log["final_metrics"] = results_summary["final_metrics"]
    training_log_path = os.path.join(log_dir, "training_log.json")
    with open(training_log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    volume.commit()
    print(f"Results saved to {model_dir}")
    
    return {
        "status": "success",
        "model_dir": model_dir,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_acc,
        "test_loss": avg_test_loss
    }

@app.local_entrypoint()
def main():
    print("Running a single roberta-base experiment on IEMOCAP_Final dataset...")
    result = run_experiment.remote()
    print(f"Experiment complete. Result: {result}") 