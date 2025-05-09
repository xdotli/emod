#!/usr/bin/env python3
"""
Script to download EMOD experiment results from Modal volume
This avoids the nesting issue with @app.function decorators
"""

import os
import sys
import argparse
from pathlib import Path
import json
import time
import subprocess
from tqdm import tqdm

try:
    import modal
    print("Successfully imported modal")
except ImportError as e:
    print(f"Error importing modal: {e}")
    print("Please install modal with: pip install modal")
    sys.exit(1)

# Create a simple app just for downloading
app = modal.App("emod-results-downloader")

# Create persistent volume reference
volume = modal.Volume.from_name("emod-results-vol")
VOLUME_PATH = "/root/results"

# Define this function at global scope to avoid nesting issues
@app.function(volumes={VOLUME_PATH: volume}, timeout=3600)
def list_results():
    """List all experiment results in the volume"""
    import os
    import json
    import glob
    
    # List only completed experiments (those with final_results.json)
    try:
        # Find all final_results.json files and get their parent directories
        completed_results = glob.glob(f"{VOLUME_PATH}/**/logs/final_results.json", recursive=True)
        
        # Extract experiment directories from paths
        completed_dirs = []
        experiment_summaries = []
        
        for result_path in completed_results:
            # Get experiment dir (two levels up from final_results.json)
            logs_dir = os.path.dirname(result_path)
            exp_dir = os.path.dirname(logs_dir)
            exp_name = os.path.basename(exp_dir)
            completed_dirs.append(exp_name)
            
            # Read the results data
            try:
                with open(result_path, 'r') as f:
                    results_data = json.load(f)
                
                # Create a summary of this experiment
                summary = {
                    "experiment_id": exp_name,
                    "model_name": results_data.get("model_name", "unknown"),
                    "dataset": results_data.get("dataset", "unknown"),
                    "experiment_type": results_data.get("experiment_type", "unknown"),
                    "best_val_accuracy": results_data.get("best_val_accuracy", 0),
                    "status": "completed"
                }
                
                # Add additional metrics if available
                if "final_metrics" in results_data:
                    summary["metrics"] = results_data["final_metrics"]
                
                # Add audio and fusion info for multimodal experiments
                if results_data.get("experiment_type") == "multimodal":
                    summary["audio_feature"] = results_data.get("audio_feature", "unknown")
                    summary["fusion_type"] = results_data.get("fusion_type", "unknown")
                    
                experiment_summaries.append(summary)
            except Exception as e:
                print(f"Error parsing {result_path}: {e}")
        
        # Return completed experiments
        return {
            "result_dirs": completed_dirs,
            "summaries": experiment_summaries,
            "count": len(completed_dirs)
        }
    except Exception as e:
        return {"error": str(e), "result_dirs": []}

def download_experiment_results(output_dir="./emod_results", specific_folders=None):
    """Download results from Modal volume to local directory"""
    print(f"\nDownloading results to {output_dir}...")
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run the app to get list of results
    with app.run() as app_ctx:
        print("Connecting to Modal and listing completed experiments...")
        info = list_results.remote()
        
        if "error" in info:
            print(f"Error: {info['error']}")
            return None
        
        result_dirs = info["result_dirs"]
        summaries = info["summaries"]
        
        # Check if we have any results
        if len(result_dirs) == 0:
            print("No completed experiments found in the Modal volume.")
            return output_dir
            
        print(f"Found {len(result_dirs)} completed experiments")
        
        # Filter directories if specific folders are requested
        if specific_folders:
            result_dirs = [d for d in result_dirs if d in specific_folders]
            print(f"Filtered to {len(result_dirs)} requested directories")
        
        # Save experiment summary
        summary_path = os.path.join(output_dir, "experiment_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summaries, f, indent=2)
        print(f"Saved experiment summary to {summary_path}")
        
        # If no directories to download, stop here
        if len(result_dirs) == 0:
            print("No experiment directories to download after filtering.")
            return output_dir
        
        # Start a Modal shell client for direct access
        print("\nStarting Modal shell client to download results...")
        shell_cmd = ["modal", "shell", "emod-results-downloader"]
        try:
            # Use subprocess to open a modal shell
            with subprocess.Popen(
                shell_cmd, 
                stdin=subprocess.PIPE, 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            ) as shell:
                # Wait for shell initialization
                time.sleep(3)
                
                # First command to verify we're in the shell
                shell.stdin.write("cd /root/results && ls -l | head -5\n")
                shell.stdin.flush()
                time.sleep(2)
                
                # Download each experiment's logs and best model
                for i, result_dir in enumerate(tqdm(result_dirs, desc="Downloading experiments")):
                    local_log_dir = os.path.join(output_dir, result_dir, "logs")
                    local_model_dir = os.path.join(output_dir, result_dir, "checkpoints")
                    
                    # Create output directories
                    os.makedirs(local_log_dir, exist_ok=True)
                    os.makedirs(local_model_dir, exist_ok=True)
                    
                    # Download logs (only key files, not all training details)
                    shell.stdin.write(f"cd /root/results/{result_dir}/logs && "
                                     f"cat final_results.json > {local_log_dir}/final_results.json\n")
                    shell.stdin.flush()
                    time.sleep(0.5)
                    
                    # Download best model if exists
                    best_model_path = f"/root/results/{result_dir}/checkpoints/best_model.pt"
                    shell.stdin.write(f"if [ -f {best_model_path} ]; then "
                                     f"cat {best_model_path} > {local_model_dir}/best_model.pt; "
                                     f"echo 'Downloaded model for {result_dir}'; fi\n")
                    shell.stdin.flush()
                    time.sleep(0.5)
                
                # Exit the shell
                shell.stdin.write("exit\n")
                shell.stdin.flush()
                
                # Wait for shell to exit
                print("\nWaiting for Modal shell to exit...")
                shell.wait()
        
        except Exception as e:
            print(f"\nError using Modal shell: {e}")
            print("Using alternative download method with direct volume access...")
            # Alternative: Use modal volume get for each experiment
            
            # Use the more direct `modal volume get` command for specific files
            for i, result_dir in enumerate(tqdm(result_dirs, desc="Downloading experiments")):
                # Create output directories
                local_dir = os.path.join(output_dir, result_dir)
                local_log_dir = os.path.join(local_dir, "logs")
                local_model_dir = os.path.join(local_dir, "checkpoints")
                os.makedirs(local_log_dir, exist_ok=True)
                os.makedirs(local_model_dir, exist_ok=True)
                
                # Download just the final_results.json file
                results_cmd = f"modal volume get emod-results-vol /root/results/{result_dir}/logs/final_results.json {local_log_dir}/final_results.json"
                os.system(results_cmd)
                
                # Download the best model if it exists
                model_cmd = f"modal volume get emod-results-vol /root/results/{result_dir}/checkpoints/best_model.pt {local_model_dir}/best_model.pt"
                # Ignore errors if file doesn't exist
                os.system(model_cmd + " 2>/dev/null")
    
    print(f"\nResults downloaded successfully to {output_dir}")
    return output_dir

def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(description="Download EMOD experiment results from Modal")
    parser.add_argument('--output-dir', type=str, default='./emod_results', 
                        help='Local directory to store downloaded results')
    parser.add_argument('--experiments', type=str, nargs='+',
                        help='Specific experiment folders to download (optional)')
    parser.add_argument('--top', type=int, default=None,
                        help='Download only the top N experiments by validation accuracy')
    
    args = parser.parse_args()
    
    # Download the results
    download_experiment_results(
        output_dir=args.output_dir,
        specific_folders=args.experiments
    )

if __name__ == "__main__":
    main() 