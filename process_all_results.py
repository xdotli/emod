#!/usr/bin/env python3
"""
Main entry point for processing all EMOD experiment results.
This script orchestrates the complete workflow:
1. Download results from Modal volume
2. Process and log detailed metrics
3. Generate comprehensive reports
"""

import os
import argparse
import subprocess
from datetime import datetime
import sys

def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f" {title} ".center(60, "="))
    print("=" * 60)

def run_script(script_name, description):
    """Run a Python script and report success/failure"""
    print_header(description)
    try:
        result = subprocess.run([sys.executable, script_name], check=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running {script_name}: {e}")
        return False

def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description="Process EMOD experiment results")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading results from Modal")
    parser.add_argument("--download-only", action="store_true", help="Only download results without processing")
    parser.add_argument("--list-only", action="store_true", help="Only list Modal volume contents without downloading")
    parser.add_argument("--target-dir", default="./results", help="Target directory for results (default: ./results)")
    args = parser.parse_args()
    
    # Print start banner
    print_header("EMOD EXPERIMENT RESULTS PROCESSING")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {os.getcwd()}")
    
    # Ensure required directories exist
    os.makedirs(args.target_dir, exist_ok=True)
    os.makedirs("./reports", exist_ok=True)
    
    # Step 1: Download results from Modal (if requested)
    if not args.skip_download:
        download_cmd = [sys.executable, "download_modal_results.py", "--target-dir", args.target_dir]
        if args.list_only:
            download_cmd.append("--list-only")
        
        try:
            subprocess.run(download_cmd, check=True)
            print("✓ Results download completed")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error downloading results: {e}")
            if not args.list_only:
                return
    
    # If only downloading or listing, exit after that step
    if args.download_only or args.list_only:
        print("Download/list operation completed as requested. Exiting.")
        return
    
    # Step 2: Run enhanced logging
    if not run_script("enhanced_logging.py", "Enhanced Results Logging"):
        print("⚠ Enhanced logging failed. Continuing with report generation...")
    
    # Step 3: Generate comprehensive report
    if not run_script("generate_report.py", "Comprehensive Report Generation"):
        print("⚠ Report generation failed.")
        return
    
    # Print completion message
    print_header("PROCESSING COMPLETE")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nResults are available in:")
    print(f"  - Raw data: {args.target_dir}")
    print(f"  - Reports: ./reports")
    print(f"  - Main report: ./reports/experiment_report.html")

if __name__ == "__main__":
    main() 