#!/usr/bin/env python3
"""
IEMOCAP Emotion Recognition System - Entry Point

This is the main entry point for running the IEMOCAP emotion recognition system.
It provides options to run different parts of the pipeline and manages execution.

Usage:
    python run.py [options]

Options:
    --train              Train the models from scratch
    --evaluate           Evaluate the models using existing checkpoints
    --visualize          Generate visualizations
    --use-ml-vad         Use machine learning for VAD to emotion mapping
    --data-path=PATH     Path to custom data (default: auto-detect)
    --checkpoint-dir=DIR Directory to save/load checkpoints (default: checkpoints/)
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Create argument parser
parser = argparse.ArgumentParser(description="IEMOCAP Emotion Recognition System")
parser.add_argument("--train", action="store_true", help="Train models from scratch")
parser.add_argument("--evaluate", action="store_true", help="Evaluate existing models")
parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
parser.add_argument("--use-ml-vad", action="store_true", help="Use ML for VAD to emotion mapping")
parser.add_argument("--data-path", type=str, default=None, help="Path to custom data")
parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory for checkpoints")
args = parser.parse_args()

# Create directories if they don't exist
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Import modules after directory creation to ensure paths exist
from main import main as run_main
import process_vad

# Check if we want to use ML for VAD to emotion mapping
if args.use_ml_vad:
    try:
        from vad_to_emotion_model import main as train_vad_model, get_vad_to_emotion_predictor
        has_ml_vad = True
    except ImportError:
        print("Warning: Could not import ML VAD to emotion model. Using rule-based approach.")
        has_ml_vad = False
else:
    has_ml_vad = False

def main():
    """Main entry point function"""
    print("=" * 80)
    print("IEMOCAP Emotion Recognition System")
    print("=" * 80)
    
    # Set up arguments to pass to main function
    run_args = {
        "force_train": args.train,
        "evaluate_only": args.evaluate,
        "visualize": args.visualize,
        "checkpoint_dir": args.checkpoint_dir,
        "data_path": args.data_path
    }
    
    # If we want to use ML VAD to emotion mapping and have the model
    if args.use_ml_vad and has_ml_vad:
        # Check if VAD model needs training
        vad_model_path = os.path.join(args.checkpoint_dir, "vad_to_emotion_model.pkl")
        if not os.path.exists(vad_model_path) or args.train:
            print("\nTraining VAD to emotion model...")
            train_vad_model()
        else:
            print("\nUsing existing VAD to emotion model")
        
        # Set the VAD to emotion function
        run_args["vad_to_emotion_func"] = get_vad_to_emotion_predictor()
    
    # Run the main function with our arguments
    run_main(**run_args)
    
    print("\nAll tasks completed successfully")
    print("=" * 80)

if __name__ == "__main__":
    main() 