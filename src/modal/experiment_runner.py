#!/usr/bin/env python3
"""
Experiment Runner Module for EMOD

This module handles running grid search experiments on Modal.
It manages launching Modal jobs for different model combinations.
"""

import os
import sys
import subprocess
import json
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

# Import from modal_setup
from src.modal.modal_setup import authenticate_modal, ModalSetup

# Default model options
DEFAULT_TEXT_MODELS = [
    "roberta-base",              # Default baseline
    "microsoft/deberta-v3-base", # Replacing BERT
    "distilbert-base-uncased",   # Smaller, faster alternative
    "xlnet-base-cased",          # Alternative architecture
    "albert-base-v2"             # Parameter-efficient model
]

DEFAULT_AUDIO_FEATURES = [
    "mfcc",                      # Traditional MFCCs
    "spectrogram",               # Time-frequency representation
    "prosodic",                  # Hand-crafted features
    "wav2vec"                    # Pre-trained speech model
]

DEFAULT_FUSION_TYPES = [
    "early",                     # Concatenate features before processing
    "late",                      # Process separately, then combine
    "hybrid",                    # Combination of early and late fusion
    "attention"                  # Cross-modal attention mechanism
]

DEFAULT_ML_CLASSIFIERS = [
    "random_forest",
    "gradient_boosting",
    "svm",
    "logistic_regression",
    "mlp"
]

def ensure_modal_installed() -> bool:
    """Check if Modal CLI is installed and set up"""
    try:
        # Check Modal version
        result = subprocess.run(
            ["modal", "--version"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Modal CLI not found or not properly installed.")
        print("Please install it with: pip install modal")
        print("Then run: modal token new")
        return False

def build_experiment_commands(
    text_models: List[str],
    audio_features: Optional[List[str]] = None,
    fusion_types: Optional[List[str]] = None,
    ml_classifiers: List[str] = DEFAULT_ML_CLASSIFIERS,
    epochs: int = 20,
    batch_size: int = 16
) -> List[Dict[str, Any]]:
    """
    Build commands for all experiment combinations
    
    Returns a list of dictionaries with experiment configuration and command
    """
    commands = []
    
    # Text-only experiments
    if audio_features is None or fusion_types is None:
        for text_model in text_models:
            # Clean model name for file paths
            clean_model_name = text_model.replace('/', '_')
            
            cmd = {
                "type": "text",
                "text_model": text_model,
                "epochs": epochs,
                "batch_size": batch_size,
                "command": [
                    "python", "-m", "modal", "run", "experiments/modal_single_experiment.py",
                    "--experiment-type", "text",
                    "--text-model", text_model,
                    "--epochs", str(epochs)
                ]
            }
            commands.append(cmd)
    
    # Multimodal experiments
    else:
        for text_model in text_models:
            for audio_feature in audio_features:
                for fusion_type in fusion_types:
                    # Clean model name for file paths
                    clean_model_name = text_model.replace('/', '_')
                    
                    cmd = {
                        "type": "multimodal",
                        "text_model": text_model,
                        "audio_feature": audio_feature,
                        "fusion_type": fusion_type,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "command": [
                            "python", "-m", "modal", "run", "experiments/modal_single_experiment.py",
                            "--experiment-type", "multimodal",
                            "--text-model", text_model,
                            "--audio-feature", audio_feature,
                            "--fusion-type", fusion_type,
                            "--epochs", str(epochs)
                        ]
                    }
                    commands.append(cmd)
    
    return commands

def run_command(cmd: List[str], dry_run: bool = False) -> bool:
    """Run a command and return success status"""
    if dry_run:
        print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        return True
    
    try:
        process = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return False

def log_experiment(experiment: Dict[str, Any], status: bool) -> None:
    """Log experiment details to the experiment log file"""
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create or append to experiment log
    log_path = os.path.join(log_dir, "experiment_log.jsonl")
    
    # Format log entry
    log_entry = {
        "timestamp": timestamp,
        "status": "success" if status else "failed",
        "experiment_type": experiment["type"],
        "config": {
            k: v for k, v in experiment.items() 
            if k not in ["type", "command"]
        }
    }
    
    # Append to log file
    with open(log_path, 'a') as f:
        f.write(json.dumps(log_entry) + "\n")

def run_experiment_grid(
    text_models: List[str] = DEFAULT_TEXT_MODELS,
    audio_features: Optional[List[str]] = None,
    fusion_types: Optional[List[str]] = None,
    ml_classifiers: List[str] = DEFAULT_ML_CLASSIFIERS,
    epochs: int = 20,
    batch_size: int = 16,
    parallel: bool = False,
    dry_run: bool = False
) -> bool:
    """
    Run a grid of experiments with different model configurations
    
    Args:
        text_models: List of text encoders to evaluate
        audio_features: List of audio feature types (for multimodal)
        fusion_types: List of fusion strategies (for multimodal)
        ml_classifiers: List of ML classifiers for emotion classification
        epochs: Number of training epochs
        batch_size: Batch size for training
        parallel: Whether to run experiments in parallel
        dry_run: If True, print commands without executing
        
    Returns:
        bool: Success status
    """
    # Ensure Modal is set up
    if not dry_run:
        if not ensure_modal_installed():
            return False
        
        if not authenticate_modal():
            return False
    
    # Build experiment commands
    commands = build_experiment_commands(
        text_models=text_models,
        audio_features=audio_features,
        fusion_types=fusion_types,
        ml_classifiers=ml_classifiers,
        epochs=epochs,
        batch_size=batch_size
    )
    
    print(f"Prepared {len(commands)} experiment combinations")
    
    # Execute commands
    success_count = 0
    
    if not parallel:
        # Sequential execution
        for i, experiment in enumerate(commands):
            print(f"\nRunning experiment {i+1}/{len(commands)}: {experiment['type']} model")
            
            if experiment['type'] == 'text':
                print(f"  - Text model: {experiment['text_model']}")
            else:
                print(f"  - Text model: {experiment['text_model']}")
                print(f"  - Audio features: {experiment['audio_feature']}")
                print(f"  - Fusion type: {experiment['fusion_type']}")
                
            print(f"  - Epochs: {experiment['epochs']}")
            
            status = run_command(experiment["command"], dry_run)
            log_experiment(experiment, status)
            
            if status:
                success_count += 1
                print(f"✓ Experiment {i+1} launched successfully")
            else:
                print(f"✗ Experiment {i+1} failed to launch")
    else:
        # TODO: Implement parallel execution if needed
        # This would likely involve using modal.app.map() or similar
        print("Parallel execution not yet implemented")
        return False
    
    # Report results
    print(f"\nExperiment grid summary: {success_count}/{len(commands)} experiments launched successfully")
    return success_count == len(commands)

if __name__ == "__main__":
    # This can be run directly for testing
    print("Testing experiment runner...")
    run_experiment_grid(
        text_models=["roberta-base"],
        audio_features=["mfcc"],
        fusion_types=["early"],
        epochs=5,
        dry_run=True
    ) 