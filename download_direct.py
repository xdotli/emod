#!/usr/bin/env python3
"""
Simple script to download experiment results directly from Modal volumes
"""

import os
import subprocess
import json
from pathlib import Path

def download_completed_experiments(output_dir="./emod_results", limit=5):
    """Download experiment results directly using Modal CLI commands"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # First list the available volumes
    print("Listing Modal volumes...")
    result = subprocess.run(["modal", "volume", "list"], capture_output=True, text=True)
    print(result.stdout)
    
    # List contents of results volume
    print("\nListing contents of emod-results-vol...")
    result = subprocess.run(["modal", "volume", "ls", "emod-results-vol", "/"], capture_output=True, text=True)
    volume_contents = result.stdout.strip().split("\n")
    experiment_dirs = [line.split()[-1] for line in volume_contents if line.strip() and not line.startswith("ls:")]
    
    # Filter out non-directory items
    experiment_dirs = [d for d in experiment_dirs if not d.startswith(".")]
    
    print(f"Found {len(experiment_dirs)} potential experiment directories")
    if len(experiment_dirs) == 0:
        print("No experiment directories found in volume.")
        return
    
    print(f"Will download up to {limit} most recent experiments")
    
    # Get the most recent experiments (assuming they're named with timestamps)
    experiment_dirs = sorted(experiment_dirs, reverse=True)[:limit]
    
    summaries = []
    
    for i, exp_dir in enumerate(experiment_dirs):
        print(f"\n[{i+1}/{len(experiment_dirs)}] Processing: {exp_dir}")
        
        # Check if logs directory exists
        logs_path = f"/root/results/{exp_dir}/logs"
        result = subprocess.run(
            ["modal", "volume", "ls", "emod-results-vol", logs_path], 
            capture_output=True, text=True
        )
        
        if "final_results.json" not in result.stdout:
            print(f"  No final_results.json found in {logs_path}")
            continue
        
        # Create local directories
        local_exp_dir = output_path / exp_dir
        local_logs_dir = local_exp_dir / "logs" 
        local_checkpoints_dir = local_exp_dir / "checkpoints"
        
        local_logs_dir.mkdir(parents=True, exist_ok=True)
        local_checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Download final_results.json
        final_results_path = f"{logs_path}/final_results.json"
        local_final_results = local_logs_dir / "final_results.json"
        
        print(f"  Downloading final results to {local_final_results}")
        subprocess.run(
            ["modal", "volume", "get", "emod-results-vol", final_results_path, str(local_final_results)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Try to download best model
        best_model_path = f"/root/results/{exp_dir}/checkpoints/best_model.pt"
        local_best_model = local_checkpoints_dir / "best_model.pt"
        
        print(f"  Attempting to download best model to {local_best_model}")
        result = subprocess.run(
            ["modal", "volume", "get", "emod-results-vol", best_model_path, str(local_best_model)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Read and store summary
        if local_final_results.exists():
            try:
                with open(local_final_results, 'r') as f:
                    results_data = json.load(f)
                
                summary = {
                    "experiment_id": exp_dir,
                    "model_name": results_data.get("model_name", "unknown"),
                    "dataset": results_data.get("dataset", "unknown"),
                    "experiment_type": results_data.get("experiment_type", "unknown"),
                    "best_val_accuracy": results_data.get("best_val_accuracy", 0)
                }
                
                if results_data.get("experiment_type") == "multimodal":
                    summary["audio_feature"] = results_data.get("audio_feature", "unknown")
                    summary["fusion_type"] = results_data.get("fusion_type", "unknown")
                
                summaries.append(summary)
                print(f"  Successfully processed {exp_dir}")
            except Exception as e:
                print(f"  Error reading results for {exp_dir}: {e}")
    
    # Save summary of downloaded experiments
    if summaries:
        summary_path = output_path / "experiment_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summaries, f, indent=2)
        print(f"\nSaved summary to {summary_path}")
    
    print(f"\nDownloaded {len(summaries)} experiments to {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download experiment results from Modal")
    parser.add_argument('--output-dir', type=str, default='./emod_results', 
                        help='Local directory to store downloaded results')
    parser.add_argument('--limit', type=int, default=5,
                        help='Maximum number of most recent experiments to download')
    
    args = parser.parse_args()
    
    download_completed_experiments(args.output_dir, args.limit) 