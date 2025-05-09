#!/usr/bin/env python3
"""
Script to download experiment results from Modal volume
Uses direct Modal CLI commands to get specific files
"""

import os
import subprocess
import json
from pathlib import Path
import time
from tqdm import tqdm
import argparse

def download_experiments(output_dir="./emod_results", limit=20, experiment_type=None, name_pattern=None):
    """Download experiment results from Modal volume"""
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # List experiment directories with ls command
    print(f"Listing experiment directories in Modal volume...")
    result = subprocess.run(
        ["modal", "volume", "ls", "emod-results-vol", "/"], 
        capture_output=True, 
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error listing volume: {result.stderr}")
        return
    
    # Parse directory listing to get experiment directories
    lines = result.stdout.strip().split('\n')
    experiment_dirs = []
    
    for line in lines:
        # Skip empty lines or non-directory entries
        if not line.strip():
            continue
            
        # Extract the directory name
        parts = line.split()
        if len(parts) == 1:  # Simple output format
            dir_name = parts[0]
            if (dir_name.endswith('.csv') or 
                dir_name.startswith('.') or
                dir_name in ['models', 'datasets', 'results']):
                continue
            experiment_dirs.append(dir_name)
    
    # Filter by experiment type if specified
    if experiment_type:
        filtered_dirs = [d for d in experiment_dirs if experiment_type.lower() in d.lower()]
        print(f"Found {len(filtered_dirs)} experiments matching type '{experiment_type}' out of {len(experiment_dirs)} total")
        experiment_dirs = filtered_dirs
        
    # Filter by name pattern if specified
    if name_pattern:
        filtered_dirs = [d for d in experiment_dirs if name_pattern.lower() in d.lower()]
        print(f"Found {len(filtered_dirs)} experiments matching pattern '{name_pattern}' out of {len(experiment_dirs)} total")
        experiment_dirs = filtered_dirs
    else:
        print(f"Found {len(experiment_dirs)} total experiments")
    
    # Sort by timestamp (assuming format with timestamp at the end)
    experiment_dirs.sort(reverse=True)  # Most recent first
    
    # Limit to specified number
    if limit and limit < len(experiment_dirs):
        print(f"Downloading the {limit} most recent experiments")
        experiment_dirs = experiment_dirs[:limit]
    
    # Download each experiment's results
    successful_downloads = []
    
    for i, exp_dir in enumerate(tqdm(experiment_dirs, desc="Downloading experiments")):
        print(f"\n[{i+1}/{len(experiment_dirs)}] Processing: {exp_dir}")
        
        # Create local directories
        local_exp_dir = output_path / exp_dir
        local_logs_dir = local_exp_dir / "logs"
        local_checkpoints_dir = local_exp_dir / "checkpoints"
        
        local_logs_dir.mkdir(parents=True, exist_ok=True)
        local_checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. First try to download final_results.json
        final_results_path = f"/{exp_dir}/logs/final_results.json"
        local_final_results = local_logs_dir / "final_results.json"
        
        try:
            print(f"  Downloading final results...")
            result = subprocess.run(
                ["modal", "volume", "get", "emod-results-vol", final_results_path, str(local_final_results)],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0 or "No such file or directory" in result.stderr:
                print(f"  Warning: Could not find final_results.json for {exp_dir}")
                # Continue anyway in case we find other files
                
            # 2. Download training_log.json
            training_log_path = f"/{exp_dir}/logs/training_log.json"
            local_training_log = local_logs_dir / "training_log.json"
            
            print(f"  Downloading training log...")
            train_result = subprocess.run(
                ["modal", "volume", "get", "emod-results-vol", training_log_path, str(local_training_log)],
                capture_output=True,
                text=True
            )
            
            if train_result.returncode != 0 or "No such file or directory" in train_result.stderr:
                print(f"  Warning: Could not find training_log.json")
                
            # 3. Download best model checkpoint if available
            best_model_path = f"/{exp_dir}/checkpoints/best_model.pt"
            local_best_model = local_checkpoints_dir / "best_model.pt"
            
            print(f"  Downloading best model checkpoint...")
            model_result = subprocess.run(
                ["modal", "volume", "get", "emod-results-vol", best_model_path, str(local_best_model)],
                capture_output=True,
                text=True
            )
            
            if model_result.returncode != 0 or "No such file or directory" in model_result.stderr:
                print(f"  Note: No best model checkpoint found")
            
            # 4. Download VAD predictions if available
            vad_preds_path = f"/{exp_dir}/vad_predictions.npz"
            local_vad_preds = local_exp_dir / "vad_predictions.npz"
            
            print(f"  Downloading VAD predictions if available...")
            vad_result = subprocess.run(
                ["modal", "volume", "get", "emod-results-vol", vad_preds_path, str(local_vad_preds)],
                capture_output=True,
                text=True
            )
            
            if vad_result.returncode != 0 or "No such file or directory" in vad_result.stderr:
                print(f"  Note: No VAD predictions found")
            
            # Create summary if we got anything
            has_any_file = (
                local_final_results.exists() or 
                local_training_log.exists() or 
                local_best_model.exists() or 
                local_vad_preds.exists()
            )
            
            if has_any_file:
                # Try to extract info from final results if available
                summary = {
                    "experiment_id": exp_dir,
                    "has_final_results": local_final_results.exists(),
                    "has_training_log": local_training_log.exists(),
                    "has_model": local_best_model.exists(),
                    "has_vad_preds": local_vad_preds.exists()
                }
                
                # Try to extract more info from final_results.json
                if local_final_results.exists():
                    try:
                        with open(local_final_results, 'r') as f:
                            results_data = json.load(f)
                        
                        # Extract key information
                        summary.update({
                            "model_name": results_data.get("model_name", "unknown"),
                            "dataset": results_data.get("dataset", "unknown"),
                            "experiment_type": results_data.get("experiment_type", "unknown"),
                            "best_val_accuracy": results_data.get("best_val_accuracy", 0)
                        })
                        
                        # Add additional information for multimodal experiments
                        if "audio_feature" in results_data:
                            summary["audio_feature"] = results_data["audio_feature"]
                        if "fusion_type" in results_data:
                            summary["fusion_type"] = results_data["fusion_type"]
                    except Exception as e:
                        print(f"  Error processing results: {e}")
                
                successful_downloads.append(summary)
                print(f"  Successfully downloaded files for {exp_dir}")
        
        except Exception as e:
            print(f"  Error processing {exp_dir}: {e}")
    
    # Save summary of downloaded experiments
    if successful_downloads:
        summary_path = output_path / "experiments_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(successful_downloads, f, indent=2)
        
        print(f"\nSuccessfully downloaded {len(successful_downloads)} experiments to {output_dir}")
        print(f"Summary saved to {summary_path}")
    else:
        print("\nNo experiments were successfully downloaded")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download EMOD experiment results from Modal")
    parser.add_argument('--output-dir', type=str, default='./emod_results', 
                        help='Local directory to store downloaded results')
    parser.add_argument('--limit', type=int, default=20,
                        help='Maximum number of experiments to download')
    parser.add_argument('--type', type=str, choices=['text', 'multimodal'], 
                        help='Filter experiments by type (text or multimodal)')
    parser.add_argument('--pattern', type=str,
                        help='Filter experiments by name pattern (e.g., IEMOCAP)')
    
    args = parser.parse_args()
    
    download_experiments(
        output_dir=args.output_dir,
        limit=args.limit,
        experiment_type=args.type,
        name_pattern=args.pattern
    ) 