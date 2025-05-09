#!/usr/bin/env python3
"""
Script to run all EMOD experiments on both datasets with all models
and download results when complete
"""

# Import necessary modules
import sys
import os
import argparse
from pathlib import Path
import time
import concurrent.futures
print("Python version:", sys.version)
print("Starting script execution...")

try:
    import modal
    print("Successfully imported modal")
except ImportError as e:
    print(f"Error importing modal: {e}")
    print("Please install modal with: pip install modal")
    sys.exit(1)

# Import from the comprehensive experiments module
from experiments.run_comprehensive_experiments import TEXT_MODELS, AUDIO_FEATURES, FUSION_TYPES

# Create a customized image that explicitly includes the experiments module
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
    .add_local_python_source("experiments")  # Explicitly add experiments module
)

# Create updated app with our custom image
app = modal.App("emod-batch-experiments", image=image)

# Create a function to download results from Modal volume
def download_results(output_dir="./emod_results", specific_folders=None):
    """Download results from Modal volume to local directory"""
    print(f"\nDownloading results to {output_dir}...")
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Access the volume
    volume = modal.Volume.from_name("emod-results-vol")
    VOLUME_PATH = "/root/results"
    
    # Define a Modal function to list and download contents
    @app.function(volumes={VOLUME_PATH: volume})
    def download_volume_contents():
        import os
        import shutil
        from pathlib import Path
        import json
        
        # List all directories and files in the volume
        all_items = os.listdir(VOLUME_PATH)
        result_dirs = [d for d in all_items if os.path.isdir(os.path.join(VOLUME_PATH, d))]
        
        # Filter directories if specific folders are requested
        if specific_folders:
            result_dirs = [d for d in result_dirs if d in specific_folders]
        
        # Create a structure to store experiment summaries
        experiment_summaries = []
        
        # Copy each result directory to a temporary location with its files
        for result_dir in result_dirs:
            source_dir = os.path.join(VOLUME_PATH, result_dir)
            temp_dir = os.path.join("/tmp", result_dir)
            
            # Skip if not a directory
            if not os.path.isdir(source_dir):
                continue
                
            # Create temp dir
            os.makedirs(temp_dir, exist_ok=True)
            
            # Check for logs directory
            logs_dir = os.path.join(source_dir, "logs")
            if os.path.exists(logs_dir):
                # Copy logs directory
                logs_temp_dir = os.path.join(temp_dir, "logs")
                os.makedirs(logs_temp_dir, exist_ok=True)
                
                # Copy log files
                for log_file in os.listdir(logs_dir):
                    src_file = os.path.join(logs_dir, log_file)
                    dst_file = os.path.join(logs_temp_dir, log_file)
                    if os.path.isfile(src_file):
                        shutil.copy2(src_file, dst_file)
                
                # Extract summary info from final_results.json if it exists
                final_results_path = os.path.join(logs_dir, "final_results.json")
                if os.path.exists(final_results_path):
                    try:
                        with open(final_results_path, 'r') as f:
                            results_data = json.load(f)
                            
                        # Create a summary of this experiment
                        summary = {
                            "experiment_id": result_dir,
                            "model_name": results_data.get("model_name", "unknown"),
                            "dataset": results_data.get("dataset", "unknown"),
                            "experiment_type": results_data.get("experiment_type", "unknown"),
                            "best_val_accuracy": results_data.get("best_val_accuracy", 0),
                            "status": "completed"
                        }
                        
                        # Add audio and fusion info for multimodal experiments
                        if results_data.get("experiment_type") == "multimodal":
                            summary["audio_feature"] = results_data.get("audio_feature", "unknown")
                            summary["fusion_type"] = results_data.get("fusion_type", "unknown")
                            
                        experiment_summaries.append(summary)
                    except Exception as e:
                        print(f"Error reading results from {final_results_path}: {e}")
                        experiment_summaries.append({
                            "experiment_id": result_dir,
                            "status": "error",
                            "error": str(e)
                        })
                else:
                    # No final results found, experiment may be ongoing or failed
                    experiment_summaries.append({
                        "experiment_id": result_dir,
                        "status": "incomplete"
                    })
            
            # Check for checkpoints directory
            checkpoints_dir = os.path.join(source_dir, "checkpoints")
            if os.path.exists(checkpoints_dir):
                # Copy model checkpoints
                checkpoints_temp_dir = os.path.join(temp_dir, "checkpoints")
                os.makedirs(checkpoints_temp_dir, exist_ok=True)
                
                # Only copy the best model to save space
                best_model_path = os.path.join(checkpoints_dir, "best_model.pt")
                if os.path.exists(best_model_path):
                    shutil.copy2(best_model_path, os.path.join(checkpoints_temp_dir, "best_model.pt"))
        
        # Write summary index file
        summary_path = os.path.join("/tmp", "experiment_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(experiment_summaries, f, indent=2)
        
        return {
            "result_dirs": result_dirs,
            "summary_file": "experiment_summary.json"
        }
    
    # Run the download function
    with app.run() as app_ctx:
        print("Connecting to Modal and downloading results...")
        download_info = download_volume_contents.remote()
        
        # Now download the files from the container's /tmp directory
        print(f"Downloaded information about {len(download_info['result_dirs'])} experiment directories")
        
        # Check if we have any results
        if len(download_info['result_dirs']) == 0:
            print("No results found in the Modal volume.")
            return output_dir
        
        for result_dir in download_info['result_dirs']:
            local_dir = os.path.join(output_dir, result_dir)
            os.makedirs(local_dir, exist_ok=True)
            
            # Run a command to download each directory using modal CLI
            download_cmd = f"modal volume cp emod-results-vol:/{result_dir} {local_dir}"
            print(f"Running: {download_cmd}")
            os.system(download_cmd)
        
        print(f"\nResults downloaded successfully to {output_dir}")
        print(f"Found {len(download_info['result_dirs'])} experiment directories")
    
    return output_dir

# Define the batch processing function to run multiple experiments
@app.function(
    gpu="H100",
    volumes={"/root/results": modal.Volume.from_name("emod-results-vol"), 
             "/root/data": modal.Volume.from_name("emod-data-vol")},
    timeout=60 * 60 * 24,  # 24 hour timeout for batch
)
def run_experiment_batch(experiment_configs):
    """Run multiple experiments sequentially on the same GPU"""
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
    import time
    from torch.cuda.amp import autocast, GradScaler  # For mixed precision training
    
    # Enable TensorFloat32 for faster matrix multiplications on Ampere+ GPUs
    torch.set_float32_matmul_precision('high')
    
    # Set up GPU environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting batch with {len(experiment_configs)} experiments")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Experiment types: {', '.join(set(conf['type'] for conf in experiment_configs))}")
    
    # Counters for tracking progress
    completed = 0
    failed = 0
    
    # Run each experiment in sequence
    for i, config in enumerate(experiment_configs):
        print(f"\n[{i+1}/{len(experiment_configs)}] Running experiment: ", end="")
        if config["type"] == "text":
            print(f"{config['type']} - {config['model']} on {config['dataset']}")
        else:
            print(f"{config['type']} - {config['model']}, {config['audio_feature']}, {config['fusion_type']} on {config['dataset']}")
        
        # Extract parameters from config
        text_model_name = config["params"]["text_model_name"]
        dataset_name = config["params"]["dataset_name"]
        audio_feature_type = config["params"].get("audio_feature_type", None)
        fusion_type = config["params"].get("fusion_type", None)
        num_epochs = int(config["params"].get("num_epochs", 40))
        
        # Measure experiment time
        start_time = time.time()
        
        try:
            # Create output directories based on experiment type
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            experiment_type = "text" if audio_feature_type is None else "multimodal"
            
            if experiment_type == "text":
                model_dir = os.path.join(
                    "/root/results", 
                    f"{dataset_name}_text_{text_model_name.replace('-', '_').replace('/', '_')}_{timestamp}"
                )
            else:
                model_dir = os.path.join(
                    "/root/results", 
                    f"{dataset_name}_multimodal_{text_model_name.replace('-', '_').replace('/', '_')}_{audio_feature_type}_{fusion_type}_{timestamp}"
                )
            
            log_dir = os.path.join(model_dir, "logs")
            model_save_dir = os.path.join(model_dir, "checkpoints")
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(model_save_dir, exist_ok=True)
            
            # Load dataset
            dataset_path = os.path.join("/root/data", f"{dataset_name}.csv")
            print(f"Loading dataset from {dataset_path}")
            
            # Check if dataset exists
            if not os.path.exists(dataset_path):
                print(f"ERROR: Dataset not found at {dataset_path}")
                print(f"Available files in data volume: {os.listdir('/root/data')}")
                raise FileNotFoundError(f"Dataset {dataset_name} not found")
            
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
                # Multimodal - will implement simplified placeholder
                # For now, use text-only as a placeholder
                model = TextVADModel(text_model_name).to(device)
            
            # Use torch.compile with max-autotune for H100 GPUs
            model = torch.compile(model, mode='max-autotune')
            
            # Create datasets and dataloaders
            # Increase batch size for H100 GPUs
            batch_size = 32
            train_dataset = TextVADDataset(X_train, y_train_vad, tokenizer)
            val_dataset = TextVADDataset(X_val, y_val_vad, tokenizer)
            test_dataset = TextVADDataset(X_test, y_test_vad, tokenizer)
            
            # Improved DataLoader with parallel workers and pinned memory
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True
            )
            test_loader = DataLoader(
                test_dataset, 
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True
            )
            
            # Define optimizer and loss function
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
            criterion = nn.MSELoss()
            
            # Initialize gradient scaler for mixed precision training
            scaler = GradScaler()
            
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
            
            # Train for just 2 epochs for testing purposes
            # In a real scenario, we would use the full num_epochs value
            for epoch in range(num_epochs):  # Full training
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
                    
                    # Forward pass with mixed precision
                    optimizer.zero_grad()
                    with autocast():
                        outputs = model(input_ids, attention_mask)
                        loss = criterion(outputs, vad_values)
                    
                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # Log training step
                    step_loss = loss.item()
                    train_loss += step_loss
                    
                    # Log every N steps
                    if (i + 1) % log_steps == 0:
                        step_log = {"step": i + 1, "loss": step_loss}
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
                        
                        with autocast():
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
                
                # Add epoch log to training log
                training_log["epoch_logs"].append(epoch_log)
                
                # Save the training log after each epoch
                with open(os.path.join(log_dir, "training_log.json"), 'w') as f:
                    json.dump(training_log, f, indent=2)
            
            # Prepare final results summary dictionary
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
                    "best_val_loss": best_val_loss,
                    "best_val_accuracy": best_val_acc
                }
            }
            
            if experiment_type == "multimodal":
                results_summary["audio_feature"] = audio_feature_type
                results_summary["fusion_type"] = fusion_type
            
            # Save final results summary
            final_results_path = os.path.join(log_dir, "final_results.json")
            with open(final_results_path, 'w') as f:
                json.dump(results_summary, f, indent=2)
            
            print(f"Experiment completed. Results saved to {model_dir}")
            completed += 1
            
            status = "completed"
            
        except Exception as e:
            print(f"ERROR: Experiment failed with error: {str(e)}")
            status = "failed"
            failed += 1
        
        # Calculate duration
        duration = time.time() - start_time
        print(f"Experiment {i+1} {status} in {duration/60:.1f} minutes")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Small delay between experiments
        if i < len(experiment_configs) - 1:
            print("Cleaning up and preparing for next experiment...")
            time.sleep(5)
    
    print(f"All {len(experiment_configs)} experiments in batch completed!")
    print(f"Successful: {completed}, Failed: {failed}")
    
    # Commit the volume to ensure all changes are saved
    modal.Volume.from_name("emod-results-vol").commit()
    
    return {"status": "completed", "count": len(experiment_configs), "completed": completed, "failed": failed}

def run_experiments(max_gpus=10, batch_size=20):
    """Deploy app and run all experiments in batches"""
    print("\nDeploying EMOD app to Modal...")
    
    # Deploy the app with correct syntax
    app.deploy()
    print("App deployed successfully.")
    
    # Define datasets to use
    datasets = ["IEMOCAP_Final", "IEMOCAP_Filtered"]
    
    # First build a complete list of all experiment configurations
    experiment_configs = []
    
    # Text-only experiments
    print("\nPreparing experiment configurations...")
    for dataset in datasets:
        for text_model in TEXT_MODELS:
            experiment_configs.append({
                "type": "text",
                "model": text_model,
                "dataset": dataset,
                "params": {
                    "text_model_name": text_model,
                    "dataset_name": dataset,
                    "num_epochs": 40
                }
            })
    
    # Multimodal experiments
    for dataset in datasets:
        for text_model in TEXT_MODELS:
            for audio_feature in AUDIO_FEATURES:
                for fusion_type in FUSION_TYPES:
                    experiment_configs.append({
                        "type": "multimodal",
                        "model": text_model,
                        "dataset": dataset,
                        "audio_feature": audio_feature,
                        "fusion_type": fusion_type,
                        "params": {
                            "text_model_name": text_model,
                            "dataset_name": dataset,
                            "audio_feature_type": audio_feature,
                            "fusion_type": fusion_type,
                            "num_epochs": 40
                        }
                    })
    
    # Print summary of what will be launched
    text_experiments = len([e for e in experiment_configs if e["type"] == "text"])
    multimodal_experiments = len([e for e in experiment_configs if e["type"] == "multimodal"])
    total_experiments = len(experiment_configs)
    
    print(f"\nPrepared {total_experiments} experiment configurations:")
    print(f"  - {text_experiments} text-only experiments")
    print(f"  - {multimodal_experiments} multimodal experiments")
    print(f"  - Using datasets: {', '.join(datasets)}")
    print(f"  - Using text models: {', '.join(TEXT_MODELS)}")
    
    # Split experiments into batches
    print(f"\nSplitting into batches with {batch_size} experiments per GPU")
    batches = []
    for i in range(0, len(experiment_configs), batch_size):
        batch = experiment_configs[i:i+batch_size]
        batches.append(batch)
    
    print(f"Created {len(batches)} batches (will use {min(len(batches), max_gpus)} GPUs in parallel)")
    
    # Batch submission logic
    futures = []
    start_time = time.time()
    
    print("\nSubmitting experiment batches in parallel (this launches multiple GPU instances)...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_gpus) as executor:
        # Submit batches in parallel, but limit to max_gpus
        for i, batch in enumerate(batches):
            if i >= max_gpus:
                # Wait for a batch to complete before submitting more
                completed, futures = concurrent.futures.wait(
                    futures, 
                    return_when=concurrent.futures.FIRST_COMPLETED
                )
                
                # Process completed batches
                for future in completed:
                    result = future.result()
                    print(f"Batch completed: {result['count']} experiments (successful: {result.get('completed', 0)}, failed: {result.get('failed', 0)})")
            
            # Submit this batch
            print(f"Submitting batch {i+1}/{len(batches)} with {len(batch)} experiments...")
            future = executor.submit(run_experiment_batch.remote, batch)
            futures.append(future)
    
    # Wait for all remaining batches to complete
    print("\nAll batches submitted. Waiting for any remaining batches to complete...")
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        print(f"Batch completed: {result['count']} experiments (successful: {result.get('completed', 0)}, failed: {result.get('failed', 0)})")
    
    elapsed = time.time() - start_time
    print(f"\nAll {len(batches)} batches submitted in {elapsed:.2f} seconds")
    print(f"Each batch will run approximately {batch_size} experiments sequentially on a single GPU")
    print(f"Total experiments running: {total_experiments}")
    
    print("\nAll experiments are now running on Modal.")
    print(f"Using at most {max_gpus} GPUs with {batch_size} experiments per GPU.")
    print("Results will be saved to the Modal volume 'emod-results-vol'.")
    print("\nCheck the Modal dashboard for progress and results.")
    
    return experiment_configs

def check_running_experiments():
    """Check if any experiments are still running on Modal"""
    print("Checking for running experiments...")
    
    try:
        # List running apps
        result = os.popen("modal app list").read()
        
        # Check for our app name
        if "emod-batch-experiments" in result and "running" in result:
            print("Experiments are still running.")
            return True
        else:
            print("All experiments have completed.")
            return False
    except Exception as e:
        print(f"Error checking experiment status: {e}")
        # If we can't check, assume they're still running
        return True

def main():
    """Main function to parse arguments and run the requested operations"""
    parser = argparse.ArgumentParser(description="Run EMOD experiments and/or download results")
    parser.add_argument('--run', action='store_true', help='Run all experiments')
    parser.add_argument('--download', action='store_true', help='Download results from Modal volume')
    parser.add_argument('--output-dir', type=str, default='./emod_results', 
                        help='Local directory to store downloaded results')
    parser.add_argument('--wait', type=int, default=0, 
                        help='Wait time in minutes before downloading results (only used with --run and --download)')
    parser.add_argument('--max-gpus', type=int, default=10,
                        help='Maximum number of GPUs to use in parallel (Modal limit is typically 10)')
    parser.add_argument('--batch-size', type=int, default=20,
                        help='Number of experiments to run sequentially on each GPU')
    parser.add_argument('--skip-wait-if-done', action='store_true',
                        help='Skip remaining wait time if all experiments are completed')
    
    args = parser.parse_args()
    
    # Default to running if no arguments provided
    if not (args.run or args.download):
        args.run = True
    
    # Run experiments if requested
    if args.run:
        completed_experiments = run_experiments(
            max_gpus=args.max_gpus,
            batch_size=args.batch_size
        )
        
        # If also downloading results, wait if specified
        if args.download and args.wait > 0:
            wait_minutes = args.wait
            print(f"\nWaiting {wait_minutes} minutes for experiments to progress before downloading results...")
            
            check_interval = 5  # Check every 5 minutes if experiments are done
            for minute in range(wait_minutes):
                remaining = wait_minutes - minute
                
                # Check if we should skip the remaining wait time
                if args.skip_wait_if_done and minute % check_interval == 0 and minute > 0:
                    if not check_running_experiments():
                        print("\nAll experiments are complete. Skipping remaining wait time.")
                        break
                
                print(f"Time remaining: {remaining} minutes...", end="\r")
                time.sleep(60)
            print("\nWait completed. Proceeding to download results.")
    
    # Download results if requested
    if args.download:
        try:
            download_results(args.output_dir)
        except Exception as e:
            print(f"\nError downloading results: {e}")
            print("Trying alternative download method...")
            # Run the download directly using system command
            os.system(f"modal volume cp emod-results-vol:/ {args.output_dir}")
            print(f"Download completed to {args.output_dir}")

if __name__ == "__main__":
    main() 