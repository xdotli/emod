#!/usr/bin/env python3
"""
Script to run all EMOD experiments on both datasets with all models
and download results when complete
"""

# Import necessary modules
import sys
import os
import argparse
from pathlib import Path
import time
import concurrent.futures
print("Python version:", sys.version)
print("Starting script execution...")

try:
    import modal
    print("Successfully imported modal")
except ImportError as e:
    print(f"Error importing modal: {e}")
    print("Please install modal with: pip install modal")
    sys.exit(1)

# Import from the comprehensive experiments module
from experiments.run_comprehensive_experiments import app, train_model, TEXT_MODELS, AUDIO_FEATURES, FUSION_TYPES

# Create a function to download results from Modal volume
def download_results(output_dir="./emod_results", specific_folders=None):
    """Download results from Modal volume to local directory"""
    print(f"\nDownloading results to {output_dir}...")
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Access the volume
    volume = modal.Volume.from_name("emod-results-vol")
    VOLUME_PATH = "/root/results"
    
    # Define a Modal function to list and download contents
    @modal.function(volumes={VOLUME_PATH: volume})
    def download_volume_contents():
        import os
        import shutil
        from pathlib import Path
        import json
        
        # List all directories and files in the volume
        all_items = os.listdir(VOLUME_PATH)
        result_dirs = [d for d in all_items if os.path.isdir(os.path.join(VOLUME_PATH, d))]
        
        # Filter directories if specific folders are requested
        if specific_folders:
            result_dirs = [d for d in result_dirs if d in specific_folders]
        
        # Create a structure to store experiment summaries
        experiment_summaries = []
        
        # Copy each result directory to a temporary location with its files
        for result_dir in result_dirs:
            source_dir = os.path.join(VOLUME_PATH, result_dir)
            temp_dir = os.path.join("/tmp", result_dir)
            
            # Skip if not a directory
            if not os.path.isdir(source_dir):
                continue
                
            # Create temp dir
            os.makedirs(temp_dir, exist_ok=True)
            
            # Check for logs directory
            logs_dir = os.path.join(source_dir, "logs")
            if os.path.exists(logs_dir):
                # Copy logs directory
                logs_temp_dir = os.path.join(temp_dir, "logs")
                os.makedirs(logs_temp_dir, exist_ok=True)
                
                # Copy log files
                for log_file in os.listdir(logs_dir):
                    src_file = os.path.join(logs_dir, log_file)
                    dst_file = os.path.join(logs_temp_dir, log_file)
                    if os.path.isfile(src_file):
                        shutil.copy2(src_file, dst_file)
                
                # Extract summary info from final_results.json if it exists
                final_results_path = os.path.join(logs_dir, "final_results.json")
                if os.path.exists(final_results_path):
                    try:
                        with open(final_results_path, 'r') as f:
                            results_data = json.load(f)
                            
                        # Create a summary of this experiment
                        summary = {
                            "experiment_id": result_dir,
                            "model_name": results_data.get("model_name", "unknown"),
                            "dataset": results_data.get("dataset", "unknown"),
                            "experiment_type": results_data.get("experiment_type", "unknown"),
                            "best_val_accuracy": results_data.get("best_val_accuracy", 0),
                            "status": "completed"
                        }
                        
                        # Add audio and fusion info for multimodal experiments
                        if results_data.get("experiment_type") == "multimodal":
                            summary["audio_feature"] = results_data.get("audio_feature", "unknown")
                            summary["fusion_type"] = results_data.get("fusion_type", "unknown")
                            
                        experiment_summaries.append(summary)
                    except Exception as e:
                        print(f"Error reading results from {final_results_path}: {e}")
                        experiment_summaries.append({
                            "experiment_id": result_dir,
                            "status": "error",
                            "error": str(e)
                        })
                else:
                    # No final results found, experiment may be ongoing or failed
                    experiment_summaries.append({
                        "experiment_id": result_dir,
                        "status": "incomplete"
                    })
            
            # Check for checkpoints directory
            checkpoints_dir = os.path.join(source_dir, "checkpoints")
            if os.path.exists(checkpoints_dir):
                # Copy model checkpoints
                checkpoints_temp_dir = os.path.join(temp_dir, "checkpoints")
                os.makedirs(checkpoints_temp_dir, exist_ok=True)
                
                # Only copy the best model to save space
                best_model_path = os.path.join(checkpoints_dir, "best_model.pt")
                if os.path.exists(best_model_path):
                    shutil.copy2(best_model_path, os.path.join(checkpoints_temp_dir, "best_model.pt"))
        
        # Write summary index file
        summary_path = os.path.join("/tmp", "experiment_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(experiment_summaries, f, indent=2)
        
        return {
            "result_dirs": result_dirs,
            "summary_file": "experiment_summary.json"
        }
    
    # Run the download function
    with app.run() as app_ctx:
        print("Connecting to Modal and downloading results...")
        download_info = download_volume_contents.remote()
        
        # Now download the files from the container's /tmp directory
        print(f"Downloaded information about {len(download_info['result_dirs'])} experiment directories")
        for result_dir in download_info['result_dirs']:
            local_dir = os.path.join(output_dir, result_dir)
            os.makedirs(local_dir, exist_ok=True)
            
            # Run a command to download each directory
            download_cmd = f"modal volume cp emod-results-vol:/{result_dir} {local_dir}"
            print(f"Running: {download_cmd}")
            os.system(download_cmd)
        
        print(f"\nResults downloaded successfully to {output_dir}")
        print(f"Found {len(download_info['result_dirs'])} experiment directories")
    
    return output_dir

# Helper function to launch a single experiment
def launch_single_experiment(config):
    """Launch a single experiment with the given configuration"""
    try:
        task_id = train_model.remote(**config["params"])
        return {"config": config, "status": "submitted", "task_id": str(task_id)}
    except Exception as e:
        return {"config": config, "status": "error", "error": str(e)}

def run_experiments():
    """Deploy app and run all experiments"""
    print("\nDeploying EMOD app to Modal...")
    
    # Deploy the app with correct syntax
    app.deploy()
    print("App deployed successfully.")
    
    # Define datasets to use
    datasets = ["IEMOCAP_Final", "IEMOCAP_Filtered"]
    
    # First build a complete list of all experiment configurations
    experiment_configs = []
    
    # Text-only experiments
    print("\nPreparing experiment configurations...")
    for dataset in datasets:
        for text_model in TEXT_MODELS:
            experiment_configs.append({
                "type": "text",
                "model": text_model,
                "dataset": dataset,
                "params": {
                    "text_model_name": text_model,
                    "dataset_name": dataset,
                    "num_epochs": 40
                }
            })
    
    # Multimodal experiments
    for dataset in datasets:
        for text_model in TEXT_MODELS:
            for audio_feature in AUDIO_FEATURES:
                for fusion_type in FUSION_TYPES:
                    experiment_configs.append({
                        "type": "multimodal",
                        "model": text_model,
                        "dataset": dataset,
                        "audio_feature": audio_feature,
                        "fusion_type": fusion_type,
                        "params": {
                            "text_model_name": text_model,
                            "dataset_name": dataset,
                            "audio_feature_type": audio_feature,
                            "fusion_type": fusion_type,
                            "num_epochs": 40
                        }
                    })
    
    # Print summary of what will be launched
    text_experiments = len([e for e in experiment_configs if e["type"] == "text"])
    multimodal_experiments = len([e for e in experiment_configs if e["type"] == "multimodal"])
    total_experiments = len(experiment_configs)
    
    print(f"\nPrepared {total_experiments} experiment configurations:")
    print(f"  - {text_experiments} text-only experiments")
    print(f"  - {multimodal_experiments} multimodal experiments")
    print(f"  - Using datasets: {', '.join(datasets)}")
    print(f"  - Using text models: {', '.join(TEXT_MODELS)}")
    
    # METHOD 1: Use concurrent.futures to launch all experiments in parallel locally
    print(f"\nLaunching all {total_experiments} experiments simultaneously...")
    start_time = time.time()
    
    # Use ThreadPoolExecutor to submit all experiments at once
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        # Submit all tasks at once
        future_to_config = {executor.submit(launch_single_experiment, config): config for config in experiment_configs}
        
        # Process results as they complete (non-blocking)
        completed = 0
        success = 0
        errors = 0
        
        print(f"Submitting {total_experiments} experiments to Modal in parallel...")
        
        for future in concurrent.futures.as_completed(future_to_config):
            completed += 1
            result = future.result()
            
            if result["status"] == "submitted":
                success += 1
            else:
                errors += 1
                
            # Print progress every 10 experiments or when complete
            if completed % 10 == 0 or completed == total_experiments:
                elapsed = time.time() - start_time
                print(f"Progress: {completed}/{total_experiments} submitted ({success} successful, {errors} failed) in {elapsed:.2f} seconds")
    
    # Final summary
    elapsed = time.time() - start_time
    print(f"\nAll experiments submitted in {elapsed:.2f} seconds")
    print(f"Successfully submitted: {success}/{total_experiments}")
    if errors > 0:
        print(f"Failed submissions: {errors}")
    
    print("\nAll experiments are running in parallel on Modal.")
    print("Each experiment is running on its own GPU instance.")
    print("Results will be saved to the Modal volume 'emod-results-vol'.")
    print("\nCheck the Modal dashboard for progress and results.")
    
    return experiment_configs

def main():
    """Main function to parse arguments and run the requested operations"""
    parser = argparse.ArgumentParser(description="Run EMOD experiments and/or download results")
    parser.add_argument('--run', action='store_true', help='Run all experiments')
    parser.add_argument('--download', action='store_true', help='Download results from Modal volume')
    parser.add_argument('--output-dir', type=str, default='./emod_results', 
                        help='Local directory to store downloaded results')
    parser.add_argument('--wait', type=int, default=0, 
                        help='Wait time in minutes before downloading results (only used with --run and --download)')
    parser.add_argument('--max-workers', type=int, default=20,
                        help='Maximum number of concurrent workers for submitting experiments')
    
    args = parser.parse_args()
    
    # Default to running if no arguments provided
    if not (args.run or args.download):
        args.run = True
    
    # Run experiments if requested
    if args.run:
        completed_experiments = run_experiments()
        
        # If also downloading results, wait if specified
        if args.download and args.wait > 0:
            wait_minutes = args.wait
            print(f"\nWaiting {wait_minutes} minutes for experiments to progress before downloading results...")
            for minute in range(wait_minutes):
                remaining = wait_minutes - minute
                print(f"Time remaining: {remaining} minutes...", end="\r")
                time.sleep(60)
            print("\nWait completed. Proceeding to download results.")
    
    # Download results if requested
    if args.download:
        download_results(args.output_dir)

if __name__ == "__main__":
    main() 