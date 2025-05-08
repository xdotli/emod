#!/usr/bin/env python3
"""
Simple script to directly launch a series of EMOD experiments on Modal
"""

import subprocess
import time
import sys
from datetime import datetime

# Define key experiments to run
EXPERIMENTS = [
    # Text-only experiments with different models
    {
        "name": "text_roberta",
        "command": ["python", "-m", "modal", "run", "modal_single_experiment.py"]
    },
    {
        "name": "text_deberta", 
        "command": ["python", "-m", "modal", "run", "modal_single_experiment.py"]
    },
    {
        "name": "text_distilbert",
        "command": ["python", "-m", "modal", "run", "modal_single_experiment.py"]
    },
    # Multimodal experiments
    {
        "name": "multimodal_roberta_mfcc_early",
        "command": ["python", "-m", "modal", "run", "modal_single_experiment.py"]
    },
    {
        "name": "multimodal_roberta_mfcc_late",
        "command": ["python", "-m", "modal", "run", "modal_single_experiment.py"]
    }
]

def main():
    print(f"=== EMOD Experiment Launcher ===")
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Will launch {len(EXPERIMENTS)} experiments on Modal")
    
    # Launch all experiments
    for i, exp in enumerate(EXPERIMENTS):
        print(f"\nLaunching experiment {i+1}/{len(EXPERIMENTS)}: {exp['name']}")
        try:
            # Run the command
            result = subprocess.run(exp["command"], check=True)
            print(f"✓ Experiment {exp['name']} launched successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error launching experiment {exp['name']}: {e}")
        
        # Wait a bit between experiment launches
        if i < len(EXPERIMENTS) - 1:
            print("Waiting 2 seconds before next launch...")
            time.sleep(2)
    
    print(f"\nAll experiments launched. Results will be saved to the Modal volume.")
    print(f"You can download them later with:")
    print(f"  python emod_cli.py results")
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
if __name__ == "__main__":
    main() 