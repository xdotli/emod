#!/usr/bin/env python3
"""
Script to run multimodal (text+audio) emotion recognition training on Modal's H100 GPUs.
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
app = modal.App("emod-multimodal", image=image)

# Create volume for persistent storage
volume = modal.Volume.from_name("emod-vol", create_if_missing=True)
VOLUME_PATH = "/root/emod"

@app.function(
    gpu="H100",
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60 * 6  # 6 hour timeout
)
def train_multimodal_model(fusion_type="early"):
    """Train the multimodal (text+audio) emotion recognition model on H100 GPU"""
    import os
    import sys
    import shutil
    
    # Set up Python path
    sys.path.append(VOLUME_PATH)
    os.chdir(VOLUME_PATH)
    
    # Import our module
    import emod_multimodal
    
    # Override parameters for more intensive training
    emod_multimodal.BATCH_SIZE = 16
    num_epochs = 20
    
    # Run training with increased epochs
    print(f"Starting multimodal model training with {fusion_type} fusion, 20 epochs...")
    
    # Create a directory to save results
    output_dir = os.path.join(VOLUME_PATH, f"results_multimodal_{fusion_type}_20epochs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create args for training
    class Args:
        def __init__(self):
            self.data_path = os.path.join(VOLUME_PATH, "IEMOCAP_Final.csv")
            self.audio_base_path = os.path.join(VOLUME_PATH, "IEMOCAP_full_release")
            self.output_dir = output_dir
            self.epochs = num_epochs
            self.seed = 42
            self.save_model = True
            self.fusion_type = fusion_type
    
    # Run the full pipeline
    args = Args()
    print(f"Loading data from {args.data_path}")
    print(f"Audio base path: {args.audio_base_path}")
    
    # Process data
    df_model, audio_features = emod_multimodal.preprocess_data(args.data_path, args.audio_base_path)
    
    if audio_features is None:
        print("No audio features extracted. Check audio path.")
        return {"error": "Audio features extraction failed"}
    
    # Prepare data
    X = df_model['Transcript'].values
    y_vad = df_model[['valence', 'arousal', 'dominance']].values
    y_emotion = df_model['Mapped_Emotion'].values
    
    # Split data
    import numpy as np
    from sklearn.model_selection import train_test_split
    
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
    
    # Initialize tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(emod_multimodal.MODEL_NAME)
    
    # Train multimodal VAD prediction model
    vad_model = emod_multimodal.train_multimodal_vad_model(
        X_train, audio_train, y_vad_train,
        X_val, audio_val, y_vad_val,
        tokenizer, fusion_type=args.fusion_type, num_epochs=args.epochs
    )
    
    # Evaluate multimodal VAD prediction model
    vad_preds, vad_targets, vad_metrics = emod_multimodal.evaluate_multimodal_vad_model(
        vad_model, X_test, audio_test, y_vad_test, tokenizer
    )
    
    # Generate VAD predictions for training emotion classifier
    import torch
    from tqdm import tqdm
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vad_train_preds = []
    
    with torch.no_grad():
        train_dataset = emod_multimodal.MultimodalVADDataset(X_train, audio_train, y_vad_train, tokenizer)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=emod_multimodal.BATCH_SIZE)
        
        for batch in tqdm(train_loader, desc="Generating VAD predictions for train set"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio_features = batch['audio_features'].to(device)
            outputs = vad_model(input_ids, attention_mask, audio_features)
            vad_train_preds.append(outputs.cpu().numpy())
    
    vad_train_preds = np.vstack(vad_train_preds)
    
    # Train emotion classifier
    emotion_model = emod_multimodal.train_emotion_classifier(vad_train_preds, y_emotion_train)
    
    # Evaluate emotion classifier
    emotion_preds, emotion_metrics = emod_multimodal.evaluate_emotion_classifier(emotion_model, vad_preds, y_emotion_test)
    
    # Save results
    import os
    import torch
    
    data_info = {
        'total_samples': len(df_model),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'emotion_distribution': {emotion: str(count) for emotion, count in zip(*np.unique(y_emotion, return_counts=True))},
        'fusion_type': args.fusion_type
    }
    
    # Save models
    os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
    torch.save(vad_model.state_dict(), os.path.join(args.output_dir, 'models', 'vad_model.pt'))
    torch.save(emotion_model, os.path.join(args.output_dir, 'models', 'emotion_model.pt'))
    
    return {
        "message": "Multimodal training completed successfully!",
        "vad_metrics": vad_metrics,
        "emotion_metrics": emotion_metrics,
        "fusion_type": fusion_type
    }

@app.function(
    volumes={VOLUME_PATH: volume}
)
def upload_files():
    """Upload necessary files to the volume"""
    import os
    import shutil
    
    # Create volume directory
    os.makedirs(VOLUME_PATH, exist_ok=True)
    
    # Copy our files
    current_dir = os.getcwd()
    for file in ["emod.py", "emod_multimodal.py", "IEMOCAP_Final.csv"]:
        src = os.path.join(current_dir, file)
        dst = os.path.join(VOLUME_PATH, file)
        if os.path.exists(src):
            print(f"Copying {src} to {dst}")
            shutil.copy2(src, dst)
    
    # Copy audio directory if it exists
    audio_dir = os.path.join(current_dir, "Datasets", "IEMOCAP_full_release")
    if os.path.exists(audio_dir):
        dest_audio_dir = os.path.join(VOLUME_PATH, "IEMOCAP_full_release")
        print(f"Copying audio directory from {audio_dir} to {dest_audio_dir}")
        if not os.path.exists(dest_audio_dir):
            shutil.copytree(audio_dir, dest_audio_dir)
    
    return "Files uploaded to Modal volume"

@app.local_entrypoint()
def main(fusion_type="early"):
    """Entry point for the Modal app"""
    # First upload files
    print("Uploading files to Modal volume...")
    upload_files.remote()
    
    # Then run training
    print(f"Starting multimodal training with {fusion_type} fusion on H100 GPU...")
    result = train_multimodal_model.remote(fusion_type=fusion_type)
    
    print("Training completed!")
    print("Results:", result)

if __name__ == "__main__":
    main() 