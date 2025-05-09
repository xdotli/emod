#!/usr/bin/env python3
"""
Simple script to run emotion recognition training on Modal's H100 GPUs.
"""

import modal
import os
import sys

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
    ])
)

# Create a Modal app
app = modal.App("emod-simple", image=image)

# Create volume for persistent storage
volume = modal.Volume.from_name("emod-vol", create_if_missing=True)
VOLUME_PATH = "/root/emod"

@app.function(
    gpu="H100",
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60 * 4  # 4 hour timeout
)
def train_text_model():
    """Train the text-only emotion recognition model on H100 GPU"""
    import os
    import sys
    import shutil
    
    # Set up Python path
    sys.path.append(VOLUME_PATH)
    os.chdir(VOLUME_PATH)
    
    # Check if required files are present
    if not os.path.exists(os.path.join(VOLUME_PATH, "emod.py")) or not os.path.exists(os.path.join(VOLUME_PATH, "IEMOCAP_Final.csv")):
        return {"error": "Required files not found in volume. Please upload first."}
    
    # Import our module
    import emod
    
    # Override parameters for more intensive training
    emod.BATCH_SIZE = 16
    num_epochs = 20
    
    # Run training with increased epochs
    print("Starting text-only model training with 20 epochs...")
    
    # Create a directory to save results
    output_dir = os.path.join(VOLUME_PATH, "results_text_20epochs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create args for training
    class Args:
        def __init__(self):
            self.data_path = os.path.join(VOLUME_PATH, "IEMOCAP_Final.csv")
            self.output_dir = output_dir
            self.epochs = num_epochs
            self.seed = 42
            self.save_model = True
    
    # Run the full pipeline
    args = Args()
    print(f"Loading data from {args.data_path}")
    df_model = emod.preprocess_data(args.data_path)
    
    # Continue with the rest of the training pipeline
    import torch
    from sklearn.model_selection import train_test_split
    
    # Prepare data
    X = df_model['Transcript'].values
    y_vad = df_model[['valence', 'arousal', 'dominance']].values
    y_emotion = df_model['Mapped_Emotion'].values
    
    # Split data
    X_train, X_temp, y_vad_train, y_vad_temp, y_emotion_train, y_emotion_temp = train_test_split(
        X, y_vad, y_emotion, test_size=0.3, random_state=args.seed)
    
    X_val, X_test, y_vad_val, y_vad_test, y_emotion_val, y_emotion_test = train_test_split(
        X_temp, y_vad_temp, y_emotion_temp, test_size=0.5, random_state=args.seed)
    
    # Initialize tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(emod.MODEL_NAME)
    
    # Train models
    vad_model = emod.train_vad_model(X_train, y_vad_train, X_val, y_vad_val, tokenizer, num_epochs=args.epochs)
    
    # Evaluate
    vad_preds, vad_targets, vad_metrics = emod.evaluate_vad_model(vad_model, X_test, y_vad_test, tokenizer)
    
    # Save results
    return {
        "message": "Training completed successfully!",
        "vad_metrics": vad_metrics
    }

@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60  # 1 hour timeout for upload
)
def upload_files():
    """Upload necessary files to the volume"""
    import os
    import shutil
    
    # Create volume directory if it doesn't exist
    os.makedirs(VOLUME_PATH, exist_ok=True)
    
    # Copy our files
    current_dir = os.getcwd()
    for file in ["emod.py", "IEMOCAP_Final.csv"]:
        src = os.path.join(current_dir, file)
        dst = os.path.join(VOLUME_PATH, file)
        if os.path.exists(src):
            print(f"Copying {src} to {dst}")
            shutil.copy2(src, dst)
        else:
            print(f"Warning: Required file {src} not found")
    
    # List files in volume
    print("Files in volume:")
    for file in os.listdir(VOLUME_PATH):
        print(f"  - {file}")
    
    return "Files uploaded to Modal volume"

@app.local_entrypoint()
def main():
    """Entry point for the Modal app"""
    # First upload files
    print("Uploading files to Modal volume...")
    upload_result = upload_files.remote()
    print(upload_result)
    
    # Then run training
    print("Starting training on H100 GPU...")
    result = train_text_model.remote()
    
    print("Training completed!")
    print("Results:", result)

if __name__ == "__main__":
    main() 