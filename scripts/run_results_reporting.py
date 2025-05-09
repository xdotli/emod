#!/usr/bin/env python3
"""
Run the enhanced logging and report generation for EMOD experiments.
This script will collect all results, generate detailed metrics, and create a comprehensive report.
"""

import os
import subprocess
import sys
from datetime import datetime

def run_enhanced_logging():
    """Run the enhanced logging script to collect and process all experiment results"""
    print("Running enhanced logging...")
    try:
        subprocess.run(["python", "enhanced_logging.py"], check=True)
        print("Enhanced logging completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running enhanced logging: {e}")
        return False

def run_report_generation():
    """Run the report generation script to create a comprehensive report"""
    print("Running report generation...")
    try:
        subprocess.run(["python", "generate_report.py"], check=True)
        print("Report generation completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running report generation: {e}")
        return False

def create_results_directory():
    """Create the results directory structure if it doesn't exist"""
    os.makedirs("./results", exist_ok=True)
    os.makedirs("./reports", exist_ok=True)
    print("Created results and reports directories.")

def check_modal_results():
    """Check if Modal volume results are available locally"""
    if not os.path.exists("./results") or len(os.listdir("./results")) == 0:
        print("Warning: No results found in ./results directory.")
        return False
    return True

def main():
    """Main function to run the full reporting process"""
    print(f"=== EMOD Experiment Results Reporting ===")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create necessary directories
    create_results_directory()
    
    # Check if Modal results are available
    if not check_modal_results():
        print("You may need to download Modal results first:")
        print("  modal volume get emod-results-vol:/ ./results")
        return
    
    # Run enhanced logging
    if not run_enhanced_logging():
        print("Enhanced logging failed. Stopping.")
        return
    
    # Run report generation
    if not run_report_generation():
        print("Report generation failed.")
        return
    
    print("\nResults processing and reporting complete.")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nReports are available in the ./reports directory.")

if __name__ == "__main__":
    main() 