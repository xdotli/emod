import os
import sys
import shutil
from pathlib import Path
import pandas as pd
import modal

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

# Create an app with the image
app = modal.App("emod-training", image=image)

# Add a volume to store model checkpoints and results
volume = modal.Volume.from_name("emod-volume", create_if_missing=True)
VOLUME_PATH = "/root/emod"

@app.function(
    gpu="H100",  # Use H100 GPU
    volumes={VOLUME_PATH: volume},  # Mount path is the key, volume is the value
    timeout=60 * 60 * 6  # 6 hour timeout
)
def train_text_only(data_path, epochs=20, batch_size=16, output_dir="results/text_only", save_model=True):
    """Train the text-only emotion recognition model on GPU"""
    import torch
    import sys
    import os
    
    # Add the current directory to path so we can import from scripts
    volume_path = Path(VOLUME_PATH)
    sys.path.append(str(volume_path))
    
    # Import the entire emod module
    sys.path.insert(0, str(volume_path))
    import emod
    
    # Set batch size
    emod.BATCH_SIZE = batch_size
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_path = volume_path / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Build args object similar to what parse_args would return
    class Args:
        def __init__(self):
            self.data_path = str(volume_path / data_path)
            self.output_dir = str(output_path)
            self.epochs = epochs
            self.seed = 42
            self.save_model = save_model
    
    args = Args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Step 1: Load and preprocess data
    df_model = emod.preprocess_data(args.data_path)
    
    # Step 2: Prepare data for VAD prediction
    X = df_model['Transcript'].values
    y_vad = df_model[['valence', 'arousal', 'dominance']].values
    y_emotion = df_model['Mapped_Emotion'].values
    
    # Split data into train, validation, and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_vad_train, y_vad_temp, y_emotion_train, y_emotion_temp = train_test_split(
        X, y_vad, y_emotion, test_size=0.3, random_state=args.seed)
    
    X_val, X_test, y_vad_val, y_vad_test, y_emotion_val, y_emotion_test = train_test_split(
        X_temp, y_vad_temp, y_emotion_temp, test_size=0.5, random_state=args.seed)
    
    # Step 3: Initialize tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(emod.MODEL_NAME)
    
    # Step 4: Train VAD prediction model
    vad_model = emod.train_vad_model(X_train, y_vad_train, X_val, y_vad_val, tokenizer, num_epochs=args.epochs)
    
    # Step 5: Evaluate VAD prediction model
    vad_preds, vad_targets, vad_metrics = emod.evaluate_vad_model(vad_model, X_test, y_vad_test, tokenizer)
    
    # Step 6: Train emotion classifier using predicted VAD values
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import numpy as np
    
    vad_train_preds = []
    with torch.no_grad():
        train_dataset = emod.TextVADDataset(X_train, y_vad_train, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=emod.BATCH_SIZE)
        
        for batch in tqdm(train_loader, desc="Generating VAD predictions for train set"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = vad_model(input_ids, attention_mask)
            vad_train_preds.append(outputs.cpu().numpy())
    
    vad_train_preds = np.vstack(vad_train_preds)
    
    # Step 7: Train emotion classifier
    emotion_model = emod.train_emotion_classifier(vad_train_preds, y_emotion_train)
    
    # Step 8: Evaluate emotion classifier
    emotion_preds, emotion_metrics = emod.evaluate_emotion_classifier(emotion_model, vad_preds, y_emotion_test)
    
    # Step 9: Save results
    data_info = {
        'total_samples': len(df_model),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'emotion_distribution': {emotion: int(count) for emotion, count in zip(*np.unique(y_emotion, return_counts=True))},
    }
    
    emod.save_results(args.output_dir, vad_metrics, emotion_metrics, data_info)
    
    # Step 10: Save models if requested
    if args.save_model:
        os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
        torch.save(vad_model.state_dict(), os.path.join(args.output_dir, 'models', 'vad_model.pt'))
        torch.save(emotion_model, os.path.join(args.output_dir, 'models', 'emotion_model.pt'))
        print("Models saved successfully")
    
    return {
        'vad_metrics': vad_metrics,
        'emotion_metrics': emotion_metrics,
        'data_info': data_info
    }

@app.function(
    gpu="H100",  # Use H100 GPU
    volumes={VOLUME_PATH: volume},  # Mount path is the key, volume is the value
    timeout=60 * 60 * 6  # 6 hour timeout
)
def train_multimodal(data_path, audio_dir, epochs=20, batch_size=16, fusion_type='early', output_dir="results/multimodal", save_model=True):
    """Train the multimodal (text+audio) emotion recognition model on GPU"""
    import torch
    import sys
    import os
    
    # Add the current directory to path so we can import from scripts
    volume_path = Path(VOLUME_PATH)
    sys.path.append(str(volume_path))
    
    # Import the entire emod_multimodal module
    sys.path.insert(0, str(volume_path))
    import emod_multimodal
    
    # Set batch size
    emod_multimodal.BATCH_SIZE = batch_size
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_path = volume_path / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Build args object similar to what parse_args would return
    class Args:
        def __init__(self):
            self.data_path = str(volume_path / data_path)
            self.audio_base_path = str(volume_path / audio_dir) if audio_dir else None
            self.output_dir = str(output_path)
            self.epochs = epochs
            self.seed = 42
            self.save_model = save_model
            self.fusion_type = fusion_type
    
    args = Args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Step 1: Load and preprocess data (including audio)
    df_model, audio_features = emod_multimodal.preprocess_data(args.data_path, args.audio_base_path)
    
    if audio_features is None:
        print("No audio features extracted. Please provide a valid audio_base_path.")
        return
    
    # Step 2: Prepare data for VAD prediction
    X = df_model['Transcript'].values
    y_vad = df_model[['valence', 'arousal', 'dominance']].values
    y_emotion = df_model['Mapped_Emotion'].values
    
    # Step 3: Split data
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
    
    # Step 4: Initialize tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(emod_multimodal.MODEL_NAME)
    
    # Step 5: Train multimodal VAD prediction model
    vad_model = emod_multimodal.train_multimodal_vad_model(
        X_train, audio_train, y_vad_train,
        X_val, audio_val, y_vad_val,
        tokenizer, fusion_type=args.fusion_type, num_epochs=args.epochs
    )
    
    # Step 6: Evaluate multimodal VAD prediction model
    vad_preds, vad_targets, vad_metrics = emod_multimodal.evaluate_multimodal_vad_model(
        vad_model, X_test, audio_test, y_vad_test, tokenizer
    )
    
    # Step 7: Generate VAD predictions for training emotion classifier
    from tqdm import tqdm
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
    
    # Step 8: Train emotion classifier
    emotion_model = emod_multimodal.train_emotion_classifier(vad_train_preds, y_emotion_train)
    
    # Step 9: Evaluate emotion classifier
    emotion_preds, emotion_metrics = emod_multimodal.evaluate_emotion_classifier(emotion_model, vad_preds, y_emotion_test)
    
    # Step 10: Save results
    data_info = {
        'total_samples': len(df_model),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'emotion_distribution': {emotion: int(count) for emotion, count in zip(*np.unique(y_emotion, return_counts=True))},
        'fusion_type': args.fusion_type
    }
    
    emod_multimodal.save_results(args.output_dir, vad_metrics, emotion_metrics, data_info)
    
    # Step 11: Save models if requested
    if args.save_model:
        os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
        torch.save(vad_model.state_dict(), os.path.join(args.output_dir, 'models', 'vad_model.pt'))
        torch.save(emotion_model, os.path.join(args.output_dir, 'models', 'emotion_model.pt'))
        print("Models saved successfully")
    
    return {
        'vad_metrics': vad_metrics,
        'emotion_metrics': emotion_metrics,
        'data_info': data_info
    }

@app.function(
    volumes={VOLUME_PATH: volume}
)
def upload_data(data_path, audio_dir=None):
    """Upload data to the persistent volume"""
    import os
    import shutil
    
    # Create directory
    os.makedirs(VOLUME_PATH, exist_ok=True)
    
    # Copy the dataset
    if os.path.exists(data_path):
        dest_path = os.path.join(VOLUME_PATH, os.path.basename(data_path))
        print(f"Copying {data_path} to {dest_path}")
        shutil.copy2(data_path, dest_path)
    
    # Copy the code files
    for code_file in ["emod.py", "emod_multimodal.py"]:
        if os.path.exists(code_file):
            dest_path = os.path.join(VOLUME_PATH, code_file)
            print(f"Copying {code_file} to {dest_path}")
            shutil.copy2(code_file, dest_path)
    
    # Copy audio directory if specified
    if audio_dir and os.path.isdir(audio_dir):
        dest_audio_dir = os.path.join(VOLUME_PATH, os.path.basename(audio_dir))
        print(f"Copying audio files from {audio_dir} to {dest_audio_dir}")
        shutil.copytree(audio_dir, dest_audio_dir, dirs_exist_ok=True)
    
    return "Data upload completed."

@app.local_entrypoint()
def main(model_type="text", data_path="IEMOCAP_Final.csv", audio_dir=None, epochs=20, batch_size=16, fusion_type="early"):
    """Entrypoint to run the training on Modal with H100 GPU"""
    # Upload the required files to the volume
    print(f"Uploading data to Modal volume...")
    upload_data.remote(data_path=data_path, audio_dir=audio_dir)
    
    # Run the training
    print(f"Starting {model_type} model training with H100 GPU...")
    print(f"- Epochs: {epochs}")
    print(f"- Batch size: {batch_size}")
    
    if model_type.lower() == "text":
        print("Training text-only model...")
        results = train_text_only.remote(
            data_path=os.path.basename(data_path),
            epochs=epochs,
            batch_size=batch_size,
            output_dir=f"results/text_only_{epochs}epochs",
            save_model=True
        )
    else:
        print(f"Training multimodal model with {fusion_type} fusion...")
        results = train_multimodal.remote(
            data_path=os.path.basename(data_path),
            audio_dir=os.path.basename(audio_dir) if audio_dir else None,
            epochs=epochs,
            batch_size=batch_size,
            fusion_type=fusion_type,
            output_dir=f"results/multimodal_{fusion_type}_{epochs}epochs",
            save_model=True
        )
    
    print("Training completed!")
    print("Results:", results)

if __name__ == "__main__":
    main() 