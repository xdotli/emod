#!/usr/bin/env python3
"""
Script to download all successful EMOD experiments from Modal
This identifies experiments with valid results and downloads them to local machine
"""

import os
import subprocess
import json
from pathlib import Path
from tqdm import tqdm
import argparse
import concurrent.futures
import time

def find_successful_experiments():
    """Find all experiments with final_results.json files"""
    print("Searching for successful experiments...")
    
    # First get all experiment directories
    result = subprocess.run(
        ["modal", "volume", "ls", "emod-results-vol", "/"], 
        capture_output=True, 
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error listing volume: {result.stderr}")
        return []
    
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
    
    # Check each experiment for logs/final_results.json
    successful_exps = []
    print(f"Checking {len(experiment_dirs)} experiments for results...")
    
    for exp_dir in tqdm(experiment_dirs, desc="Finding completed experiments"):
        # Check if logs/final_results.json exists
        check_cmd = ["modal", "volume", "ls", "emod-results-vol", f"/{exp_dir}/logs"]
        result = subprocess.run(check_cmd, capture_output=True, text=True)
        
        # If we can list the logs directory and it contains final_results.json
        if result.returncode == 0 and "final_results.json" in result.stdout:
            successful_exps.append(exp_dir)
    
    return successful_exps

def download_experiment(exp_dir, output_path):
    """Download a single experiment's files"""
    # Create local directories
    local_exp_dir = output_path / exp_dir
    local_logs_dir = local_exp_dir / "logs"
    local_checkpoints_dir = local_exp_dir / "checkpoints"
    
    local_logs_dir.mkdir(parents=True, exist_ok=True)
    local_checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    success = False
    summary = {"experiment_id": exp_dir}
    
    try:
        # 1. Download final_results.json
        final_results_path = f"/{exp_dir}/logs/final_results.json"
        local_final_results = local_logs_dir / "final_results.json"
        
        result = subprocess.run(
            ["modal", "volume", "get", "emod-results-vol", final_results_path, str(local_final_results)],
            capture_output=True,
            text=True
        )
        
        has_results = result.returncode == 0 and local_final_results.exists()
        summary["has_final_results"] = has_results
        
        # 2. Download training_log.json
        training_log_path = f"/{exp_dir}/logs/training_log.json"
        local_training_log = local_logs_dir / "training_log.json"
        
        train_result = subprocess.run(
            ["modal", "volume", "get", "emod-results-vol", training_log_path, str(local_training_log)],
            capture_output=True,
            text=True
        )
        
        has_training_log = train_result.returncode == 0 and local_training_log.exists()
        summary["has_training_log"] = has_training_log
        
        # 3. Download best model checkpoint
        best_model_path = f"/{exp_dir}/checkpoints/best_model.pt"
        local_best_model = local_checkpoints_dir / "best_model.pt"
        
        model_result = subprocess.run(
            ["modal", "volume", "get", "emod-results-vol", best_model_path, str(local_best_model)],
            capture_output=True,
            text=True
        )
        
        has_model = model_result.returncode == 0 and local_best_model.exists()
        summary["has_model"] = has_model
        
        # 4. Download VAD predictions if available
        vad_preds_path = f"/{exp_dir}/vad_predictions.npz"
        local_vad_preds = local_exp_dir / "vad_predictions.npz"
        
        vad_result = subprocess.run(
            ["modal", "volume", "get", "emod-results-vol", vad_preds_path, str(local_vad_preds)],
            capture_output=True,
            text=True
        )
        
        has_vad_preds = vad_result.returncode == 0 and local_vad_preds.exists()
        summary["has_vad_preds"] = has_vad_preds
        
        # Extract experiment metadata from results
        if has_results:
            with open(local_final_results, 'r') as f:
                results_data = json.load(f)
            
            summary.update({
                "model_name": results_data.get("model_name", "unknown"),
                "dataset": results_data.get("dataset", "unknown"),
                "experiment_type": results_data.get("experiment_type", "unknown"),
                "best_val_accuracy": results_data.get("best_val_accuracy", 0)
            })
            
            # Add multimodal-specific details if present
            if "audio_feature" in results_data:
                summary["audio_feature"] = results_data["audio_feature"]
            if "fusion_type" in results_data:
                summary["fusion_type"] = results_data["fusion_type"]
            
            success = True
            
    except Exception as e:
        print(f"Error downloading {exp_dir}: {e}")
    
    return success, summary

def download_all_successful(output_dir="./emod_results", workers=3, filter_pattern=None, skip_existing=True):
    """Download all successful experiments to local machine"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # First, find all successful experiments
    successful_exps = find_successful_experiments()
    
    if not successful_exps:
        print("No successful experiments found.")
        return
    
    print(f"Found {len(successful_exps)} successful experiments")
    
    # Apply filter if specified
    if filter_pattern:
        successful_exps = [exp for exp in successful_exps if filter_pattern.lower() in exp.lower()]
        print(f"Filtered to {len(successful_exps)} experiments matching '{filter_pattern}'")
    
    # Check for existing experiments if skip_existing is True
    if skip_existing:
        # Create summary path and load if it exists
        summary_path = output_path / "all_experiments_summary.json"
        existing_exps = set()
        
        if summary_path.exists():
            try:
                with open(summary_path, 'r') as f:
                    existing_data = json.load(f)
                    for exp in existing_data:
                        existing_exps.add(exp.get("experiment_id", ""))
                print(f"Found {len(existing_exps)} previously downloaded experiments")
            except Exception as e:
                print(f"Error reading existing summary: {e}")
        
        # Filter out existing experiments
        to_download = [exp for exp in successful_exps if exp not in existing_exps]
        print(f"Skipping {len(successful_exps) - len(to_download)} already downloaded experiments")
        successful_exps = to_download
    
    if not successful_exps:
        print("All experiments already downloaded. Use --download-all to re-download.")
        return
        
    print(f"Downloading {len(successful_exps)} experiments...")
    
    # Use ThreadPoolExecutor for parallel downloads
    all_summaries = []
    successful_count = 0
    
    if workers > 1 and len(successful_exps) > 1:
        print(f"Using {workers} parallel workers")
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            # Create a mapping of future to experiment name for tracking
            future_to_exp = {
                executor.submit(download_experiment, exp, output_path): exp 
                for exp in successful_exps
            }
            
            for future in tqdm(concurrent.futures.as_completed(future_to_exp), 
                              total=len(future_to_exp),
                              desc="Downloading experiments"):
                exp = future_to_exp[future]
                try:
                    success, summary = future.result()
                    if success:
                        successful_count += 1
                    all_summaries.append(summary)
                except Exception as e:
                    print(f"Error processing {exp}: {e}")
    else:
        # Sequential downloads
        for exp in tqdm(successful_exps, desc="Downloading experiments"):
            success, summary = download_experiment(exp, output_path)
            if success:
                successful_count += 1
            all_summaries.append(summary)
    
    # Load existing summaries if available
    if skip_existing and summary_path.exists():
        try:
            with open(summary_path, 'r') as f:
                existing_summaries = json.load(f)
                # Add only non-duplicate summaries
                existing_ids = {s.get("experiment_id") for s in existing_summaries}
                for summary in all_summaries:
                    if summary.get("experiment_id") not in existing_ids:
                        existing_summaries.append(summary)
                all_summaries = existing_summaries
        except Exception as e:
            print(f"Error merging with existing summaries: {e}")
    
    # Save comprehensive summary
    summary_path = output_path / "all_experiments_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)
    
    print(f"\nSuccessfully downloaded {successful_count} new experiments")
    print(f"Total experiments in summary: {len(all_summaries)}")
    print(f"Results saved to {output_dir}")
    
    # Create dataset-based index
    dataset_index = {}
    for summary in all_summaries:
        dataset = summary.get("dataset", "unknown")
        if dataset not in dataset_index:
            dataset_index[dataset] = []
        dataset_index[dataset].append(summary)
    
    dataset_index_path = output_path / "dataset_index.json"
    with open(dataset_index_path, 'w') as f:
        json.dump(dataset_index, f, indent=2)
    print(f"Dataset index saved to {dataset_index_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download all successful EMOD experiments")
    parser.add_argument('--output-dir', type=str, default='./emod_results', 
                        help='Local directory to store downloaded results')
    parser.add_argument('--workers', type=int, default=3,
                        help='Number of parallel downloads (set to 1 for sequential)')
    parser.add_argument('--filter', type=str, default=None,
                        help='Optional filter pattern to match experiment names')
    parser.add_argument('--download-all', action='store_true',
                        help='Re-download all experiments, even if already downloaded')
    
    args = parser.parse_args()
    
    start_time = time.time()
    download_all_successful(
        output_dir=args.output_dir,
        workers=args.workers,
        filter_pattern=args.filter,
        skip_existing=not args.download_all
    )
    print(f"Total download time: {(time.time() - start_time)/60:.1f} minutes") 