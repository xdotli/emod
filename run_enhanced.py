#!/usr/bin/env python3
"""
Enhanced emotion recognition pipeline integrating SOTA models with the existing system.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Union

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from utils.config import get_config
from sota_models.integration import SotaEmotionAnalyzer
import main  # Import the original main module
import run  # Import the original run module

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Enhanced emotion recognition pipeline")
    
    # Input options
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory containing the dataset")
    
    # Model options
    parser.add_argument("--use-sota", action="store_true", default=True,
                        help="Use SOTA models for emotion recognition")
    parser.add_argument("--llm", type=str, default="anthropic/claude-3-7-sonnet-20240620",
                        help="LLM model to use for analysis")
    
    # Evaluation options
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate the model on the test set")
    parser.add_argument("--train", action="store_true",
                        help="Train the model")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to save results")
    
    # Other options
    parser.add_argument("--no-cuda", action="store_true", 
                        help="Disable CUDA even if available")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    return parser.parse_args()

def setup_environment(args):
    """Set up the environment for the pipeline."""
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if required environment variables are set
    config = get_config()
    if args.use_sota and not config['api_keys']['openrouter']:
        logger.warning("OPENROUTER_API_KEY environment variable is not set. SOTA models will be disabled.")
        args.use_sota = False

def run_enhanced_pipeline(args):
    """Run the enhanced emotion recognition pipeline."""
    # Set up environment
    setup_environment(args)
    
    # Initialize the original pipeline components
    original_args = {
        "data_dir": args.data_dir,
        "evaluate": args.evaluate,
        "train": args.train
    }
    
    if args.use_sota:
        # Initialize SOTA components
        sota_analyzer = SotaEmotionAnalyzer(use_cuda=not args.no_cuda)
        
        # Run the enhanced pipeline
        logger.info("Running enhanced emotion recognition pipeline with SOTA models")
        
        # Load the dataset
        data = pd.read_csv(os.path.join(args.data_dir, "processed_data.csv"))
        
        # If evaluating, run on a subset for demonstration
        if args.evaluate:
            # Take a small subset for demonstration
            eval_data = data.sample(min(50, len(data)), random_state=args.seed)
            
            results = []
            for idx, row in eval_data.iterrows():
                # Get text and audio path
                text = row['text']
                audio_path = os.path.join(args.data_dir, "audio", f"{row['id']}.wav")
                
                # Run SOTA analysis
                sota_result = sota_analyzer.analyze_text(
                    text=text,
                    llm_model=args.llm
                )
                
                # Get ground truth emotion
                ground_truth = row['emotion']
                
                # Get predicted emotion
                predicted_emotion = sota_result['llm_analysis']['primary_emotion']['name']
                
                # Store results
                results.append({
                    'id': row['id'],
                    'text': text,
                    'ground_truth': ground_truth,
                    'predicted_emotion': predicted_emotion,
                    'confidence': sota_result['llm_analysis']['primary_emotion']['confidence'],
                    'explanation': sota_result['llm_analysis']['primary_emotion']['explanation']
                })
            
            # Save results
            results_df = pd.DataFrame(results)
            results_path = os.path.join(args.output_dir, "sota_evaluation_results.csv")
            results_df.to_csv(results_path, index=False)
            
            # Calculate metrics
            y_true = results_df['ground_truth']
            y_pred = results_df['predicted_emotion']
            
            # Print classification report
            print("\nSOTA Model Classification Report:")
            print(classification_report(y_true, y_pred))
            
            # Save confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            cm_df = pd.DataFrame(
                cm,
                index=sorted(results_df['ground_truth'].unique()),
                columns=sorted(results_df['predicted_emotion'].unique())
            )
            cm_path = os.path.join(args.output_dir, "sota_confusion_matrix.csv")
            cm_df.to_csv(cm_path)
            
            logger.info(f"SOTA evaluation results saved to {results_path}")
            logger.info(f"SOTA confusion matrix saved to {cm_path}")
        
        # If training, run the original pipeline
        if args.train:
            logger.info("Running original training pipeline")
            run.main()
    else:
        # Run the original pipeline
        logger.info("Running original emotion recognition pipeline")
        run.main()

def main():
    """Main function."""
    args = parse_args()
    run_enhanced_pipeline(args)

if __name__ == "__main__":
    main()
