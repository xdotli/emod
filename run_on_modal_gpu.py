#!/usr/bin/env python3
"""
Run the emotion recognition training directly on Modal H100 GPUs.
This uses a simplified approach that runs your existing script on a Modal container.
"""

import modal

# Create a Modal image with required dependencies
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
app = modal.App("emod-direct", image=image)

@app.function(
    gpu="H100",
    timeout=60 * 60 * 6  # 6 hour timeout
)
def run_emotion_training(epochs=20, batch_size=16):
    """
    Run the emotion recognition model training directly inside a Modal container
    """
    import subprocess
    import sys
    import os
    
    # Download the repo code
    subprocess.run(["git", "clone", "https://github.com/yourusername/emod.git", "/root/emod"])
    os.chdir("/root/emod")
    
    # Run the original script with the specified parameters
    cmd = [
        "python", 
        "emod.py", 
        "--data_path", "IEMOCAP_Final.csv",
        "--output_dir", f"src/results/text_only_{epochs}epochs",
        "--epochs", str(epochs),
        "--save_model"
    ]
    
    # Execute the script with output streaming back to the console
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Stream output back
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
        sys.stdout.flush()
    
    process.wait()
    
    # Check if process completed successfully
    if process.returncode != 0:
        print(f"Training failed with return code {process.returncode}")
        return f"Training failed with return code {process.returncode}"
    
    return "Training completed successfully!"

@app.local_entrypoint()
def main(epochs=20, batch_size=16):
    """Entry point for the Modal app"""
    print(f"Starting emotion recognition training on H100 GPU with {epochs} epochs...")
    result = run_emotion_training.remote(epochs=epochs, batch_size=batch_size)
    print(result) 