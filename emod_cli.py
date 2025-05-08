#!/usr/bin/env python3
"""
EMOD - Unified CLI for Emotion Recognition Experiments

This script provides a unified command-line interface for:
1. Running grid search experiments on Modal
2. Downloading and processing results 
3. Generating comprehensive reports

All functionality is consolidated into this single entry point.
"""

import os
import sys
import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

# Import modules
from experiment_runner import run_experiment_grid
from results_processor import download_results, process_results
from report_generator import generate_report

# Default directories
RESULTS_DIR = "./results"
REPORTS_DIR = "./reports"

def setup_directories():
    """Ensure all required directories exist"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    print(f"✓ Directories setup complete")

def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f" {title} ".center(60, "="))
    print("=" * 60)

def handle_experiments(args):
    """Handle experiment grid search on Modal"""
    print_header("RUNNING EXPERIMENT GRID")
    
    # Define experiment grid parameters
    grid_params = {
        "text_models": args.text_models.split(",") if args.text_models else ["roberta-base"],
        "audio_features": args.audio_features.split(",") if args.audio_features else ["mfcc"],
        "fusion_types": args.fusion_types.split(",") if args.fusion_types else ["early"],
        "ml_classifiers": args.ml_classifiers.split(",") if args.ml_classifiers else ["gradient_boosting"],
        "epochs": args.epochs,
        "batch_size": args.batch_size
    }
    
    # Log configuration
    print(f"Configuration:")
    for param, value in grid_params.items():
        print(f"  - {param}: {value}")
    
    # Confirm with user if grid is large
    total_runs = (
        len(grid_params["text_models"]) * 
        (1 if not args.multimodal else len(grid_params["audio_features"]) * len(grid_params["fusion_types"]))
    )
    
    if total_runs > 3 and not args.yes:
        confirm = input(f"This will launch {total_runs} Modal jobs. Continue? (y/n): ")
        if confirm.lower() != 'y':
            print("Aborted.")
            return
    
    # Run experiment grid
    success = run_experiment_grid(
        text_models=grid_params["text_models"],
        audio_features=grid_params["audio_features"] if args.multimodal else None,
        fusion_types=grid_params["fusion_types"] if args.multimodal else None,
        ml_classifiers=grid_params["ml_classifiers"],
        epochs=grid_params["epochs"],
        batch_size=grid_params["batch_size"],
        parallel=args.parallel,
        dry_run=args.dry_run
    )
    
    if success:
        print("✓ Experiment grid launched successfully")
        print("\nTo retrieve results later, run:")
        print(f"  python {sys.argv[0]} results")
    else:
        print("✗ Failed to launch experiment grid")

def handle_results(args):
    """Handle downloading and processing results"""
    print_header("PROCESSING EXPERIMENT RESULTS")
    
    # Download results if requested
    if not args.skip_download:
        print("Downloading results from Modal...")
        success = download_results(
            target_dir=args.target_dir,
            list_only=args.list_only
        )
        if not success:
            print("✗ Error downloading results")
            return
        
        if args.list_only or args.download_only:
            print("✓ Download operation completed")
            return
    
    # Process results
    print("Processing experiment results...")
    success = process_results(
        results_dir=args.target_dir,
        output_dir=REPORTS_DIR
    )
    
    if success:
        print("✓ Results processing completed")
    else:
        print("✗ Error processing results")
        return
    
    # Generate report
    if not args.skip_report:
        print("Generating report...")
        report_path = generate_report(
            results_dir=args.target_dir,
            output_dir=REPORTS_DIR,
            template=args.template,
            format=args.format
        )
        
        if report_path:
            print(f"✓ Report generated at {report_path}")
            
            # Automatically open report on macOS
            if sys.platform == 'darwin' and os.path.exists(report_path):
                subprocess.run(['open', report_path])
        else:
            print("✗ Error generating report")

def handle_report(args):
    """Handle report generation only"""
    print_header("GENERATING REPORT")
    
    report_path = generate_report(
        results_dir=args.target_dir,
        output_dir=REPORTS_DIR,
        template=args.template,
        format=args.format
    )
    
    if report_path:
        print(f"✓ Report generated at {report_path}")
        
        # Automatically open report on macOS
        if sys.platform == 'darwin' and os.path.exists(report_path):
            subprocess.run(['open', report_path])
    else:
        print("✗ Error generating report")

def main():
    """Main entry point for EMOD CLI"""
    parser = argparse.ArgumentParser(
        description="EMOD - Unified CLI for Emotion Recognition Experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # === Experiment command ===
    exp_parser = subparsers.add_parser("experiment", help="Run experiments on Modal")
    exp_parser.add_argument("--text-models", type=str, 
                           help="Comma-separated list of text models to evaluate")
    exp_parser.add_argument("--multimodal", action="store_true",
                           help="Run multimodal experiments")
    exp_parser.add_argument("--audio-features", type=str,
                           help="Comma-separated list of audio feature types")
    exp_parser.add_argument("--fusion-types", type=str,
                           help="Comma-separated list of fusion strategies")
    exp_parser.add_argument("--ml-classifiers", type=str,
                           help="Comma-separated list of ML classifiers")
    exp_parser.add_argument("--epochs", type=int, default=20,
                           help="Number of training epochs")
    exp_parser.add_argument("--batch-size", type=int, default=16,
                           help="Batch size for training")
    exp_parser.add_argument("--parallel", action="store_true",
                           help="Run experiments in parallel")
    exp_parser.add_argument("--dry-run", action="store_true",
                           help="Show commands without executing")
    exp_parser.add_argument("--yes", "-y", action="store_true",
                           help="Skip confirmation for large experiment grids")
    
    # === Results command ===
    res_parser = subparsers.add_parser("results", help="Process experiment results")
    res_parser.add_argument("--skip-download", action="store_true",
                           help="Skip downloading results from Modal")
    res_parser.add_argument("--download-only", action="store_true",
                           help="Only download results without processing")
    res_parser.add_argument("--list-only", action="store_true",
                           help="Only list Modal volume contents without downloading")
    res_parser.add_argument("--target-dir", default=RESULTS_DIR,
                           help="Target directory for results")
    res_parser.add_argument("--skip-report", action="store_true",
                           help="Skip report generation after processing")
    res_parser.add_argument("--template", default=None,
                           help="Custom report template file")
    res_parser.add_argument("--format", choices=["html", "markdown"], default="markdown",
                           help="Report output format")
    
    # === Report command ===
    rep_parser = subparsers.add_parser("report", help="Generate report from existing results")
    rep_parser.add_argument("--target-dir", default=RESULTS_DIR,
                           help="Directory containing processed results")
    rep_parser.add_argument("--template", default=None,
                           help="Custom report template file")
    rep_parser.add_argument("--format", choices=["html", "markdown"], default="markdown",
                           help="Report output format")
    
    # Parse arguments and handle commands
    args = parser.parse_args()
    
    # Ensure directories exist
    setup_directories()
    
    # Handle commands
    if args.command == "experiment":
        handle_experiments(args)
    elif args.command == "results":
        handle_results(args)
    elif args.command == "report":
        handle_report(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1) 