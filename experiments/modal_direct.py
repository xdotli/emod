#!/usr/bin/env python3
"""
Direct approach to run emotion recognition on Modal H100 GPUs by mounting local files.
"""

import modal
import os
from pathlib import Path

# Create paths for local files
LOCAL_DATA_PATH = str(Path("./IEMOCAP_Final.csv").absolute())
LOCAL_CODE_PATH = str(Path("./emod.py").absolute())

# Ensure files exist before proceeding
if not os.path.exists(LOCAL_DATA_PATH) or not os.path.exists(LOCAL_CODE_PATH):
    print(f"Error: Required files not found locally:")
    print(f"  Data file: {LOCAL_DATA_PATH}")
    print(f"  Code file: {LOCAL_CODE_PATH}")
    exit(1)

# Define the Modal image with all required dependencies and files
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
    ])
    .add_local_file(LOCAL_DATA_PATH, "/root/IEMOCAP_Final.csv") 
    .add_local_file(LOCAL_CODE_PATH, "/root/emod.py")
)

# Create a Modal app
app = modal.App("emod-direct", image=image)

@app.function(
    gpu="H100",
    timeout=60 * 60 * 4  # 4 hour timeout
)
def train_text_model():
    """Train the text-only emotion recognition model on H100 GPU"""
    import os
    import sys
    import torch
    from sklearn.model_selection import train_test_split
    
    # Set up Python path and working directory
    code_dir = "/root"
    sys.path.append(code_dir)
    os.chdir(code_dir)
    
    # Import our module
    print("Importing emod module...")
    import emod
    
    # Override parameters for more intensive training
    emod.BATCH_SIZE = 16
    num_epochs = 20
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directory to save results
    output_dir = os.path.join(code_dir, "results_text_20epochs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create args for training
    class Args:
        def __init__(self):
            self.data_path = os.path.join(code_dir, "IEMOCAP_Final.csv")
            self.output_dir = output_dir
            self.epochs = num_epochs
            self.seed = 42
            self.save_model = True
    
    # Run training pipeline
    args = Args()
    
    # List directory contents to debug
    print("Files in working directory:")
    for file in os.listdir(code_dir):
        print(f"  - {file}")
    
    print(f"Loading data from {args.data_path}")
    df_model = emod.preprocess_data(args.data_path)
    
    # Prepare data
    X = df_model['Transcript'].values
    y_vad = df_model[['valence', 'arousal', 'dominance']].values
    y_emotion = df_model['Mapped_Emotion'].values
    
    # Split data
    X_train, X_temp, y_vad_train, y_vad_temp, y_emotion_train, y_emotion_temp = train_test_split(
        X, y_vad, y_emotion, test_size=0.3, random_state=args.seed)
    
    X_val, X_test, y_vad_val, y_vad_test, y_emotion_val, y_emotion_test = train_test_split(
        X_temp, y_vad_temp, y_emotion_temp, test_size=0.5, random_state=args.seed)
    
    print(f"Data split completed: {len(X_train)} train, {len(X_val)} validation, {len(X_test)} test samples")
    
    # Initialize tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(emod.MODEL_NAME)
    
    # Train VAD prediction model
    print(f"Training VAD model with {num_epochs} epochs...")
    vad_model = emod.train_vad_model(X_train, y_vad_train, X_val, y_vad_val, tokenizer, num_epochs=args.epochs)
    
    # Evaluate
    print("Evaluating VAD model...")
    vad_preds, vad_targets, vad_metrics = emod.evaluate_vad_model(vad_model, X_test, y_vad_test, tokenizer)
    
    # Generate VAD predictions for emotion classifier
    print("Generating VAD predictions for emotion classifier...")
    vad_train_preds = []
    with torch.no_grad():
        train_dataset = emod.TextVADDataset(X_train, y_vad_train, tokenizer)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=emod.BATCH_SIZE)
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = vad_model(input_ids, attention_mask)
            vad_train_preds.append(outputs.cpu().numpy())
    
    import numpy as np
    vad_train_preds = np.vstack(vad_train_preds)
    
    # Train emotion classifier
    print("Training emotion classifier...")
    emotion_model = emod.train_emotion_classifier(vad_train_preds, y_emotion_train)
    
    # Evaluate emotion classifier
    print("Evaluating emotion classifier...")
    emotion_preds, emotion_metrics = emod.evaluate_emotion_classifier(emotion_model, vad_preds, y_emotion_test)
    
    # Save models
    if args.save_model:
        print("Saving models...")
        model_dir = os.path.join(args.output_dir, "models")
        os.makedirs(model_dir, exist_ok=True)
        torch.save(vad_model.state_dict(), os.path.join(model_dir, "vad_model.pt"))
        torch.save(emotion_model, os.path.join(model_dir, "emotion_model.pt"))
    
    return {
        "message": "Training completed successfully!",
        "vad_metrics": vad_metrics,
        "emotion_metrics": emotion_metrics,
        "data_info": {
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
        }
    }

@app.local_entrypoint()
def main():
    """Entry point for the Modal app"""
    print("Starting training on H100 GPU with direct mounting approach...")
    print(f"Data: {LOCAL_DATA_PATH}")
    print(f"Code: {LOCAL_CODE_PATH}")
    
    result = train_text_model.remote()
    
    print("Training completed!")
    print("Results:", result)

if __name__ == "__main__":
    main() 