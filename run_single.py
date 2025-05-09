#!/usr/bin/env python3
"""
Script to run a single EMOD experiment with roberta-base on IEMOCAP_Final
"""

# Import necessary modules
import sys
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
from experiments.run_comprehensive_experiments import app, train_model

def main():
    """Main function to deploy app and run experiment"""
    print("\nDeploying EMOD app to Modal...")
    
    # Deploy the app with correct syntax
    app.deploy()
    print("App deployed successfully.")
    
    print("\nRunning a single experiment with roberta-base on IEMOCAP_Final")
    
    # Run the experiment
    result = train_model.remote(
        text_model_name="roberta-base",
        dataset_name="IEMOCAP_Final",
        num_epochs=40
    )
    
    print("\nExperiment launched on Modal.")
    print("Check the Modal dashboard for progress and results.")
    print("Results will be saved to the Modal volume 'emod-results-vol'.")

if __name__ == "__main__":
    main() 