#!/usr/bin/env python3
"""
Results Processor Module for EMOD

This module handles downloading results from Modal and processing them
into a structured format for reporting.
"""

import os
import sys
import subprocess
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# Default paths
RESULTS_DIR = "./results"
REPORTS_DIR = "./reports"

def download_results(target_dir: str = RESULTS_DIR, list_only: bool = False, verbose: bool = False) -> bool:
    """
    Download results from Modal volume to local directory
    
    Args:
        target_dir: Local directory to store results
        list_only: Only list contents without downloading
        verbose: Show verbose output
        
    Returns:
        bool: Success status
    """
    # Ensure Modal CLI is installed
    try:
        subprocess.run(["modal", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Modal CLI not found or not properly installed.")
        print("Please install it with: pip install modal")
        return False
    
    # Create target directory if it doesn't exist
    if not list_only:
        os.makedirs(target_dir, exist_ok=True)
    
    # Authenticate with Modal if needed
    try:
        subprocess.run(
            ["modal", "volume", "list"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
    except subprocess.CalledProcessError as e:
        if "Please run 'modal token new'" in e.stderr.decode():
            print("Modal authentication required.")
            try:
                subprocess.run(["modal", "token", "new"], check=True)
            except subprocess.CalledProcessError:
                print("Failed to authenticate with Modal.")
                return False
        else:
            print(f"Error checking Modal authentication: {e}")
            return False
    
    # List or download volume contents
    volume_name = "emod-results-vol"
    
    if list_only:
        print("Listing Modal volume contents...")
        command = ["modal", "volume", "ls", volume_name]
    else:
        print(f"Downloading Modal volume data to {target_dir}...")
        # First get a directory listing to find experiment directories
        try:
            list_result = subprocess.run(
                ["modal", "volume", "ls", volume_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            
            # Parse directory listing to find experiment directories
            listing = list_result.stdout.decode()
            print("Found experiments in Modal volume:")
            
            download_count = 0
            for line in listing.splitlines():
                # Skip header and footer lines
                if not line.strip() or "Directory listing" in line or "─" in line or "Type" in line:
                    continue
                
                # Extract directory names - they should start with text_model or multimodal
                parts = line.split()
                if not parts:
                    continue
                
                dir_name = parts[0].strip()
                if dir_name.startswith("text_model_") or dir_name.startswith("multimodal_"):
                    print(f"  - {dir_name}")
                    
                    # Download this experiment directory
                    download_cmd = ["modal", "volume", "get", volume_name, f"/{dir_name}", target_dir]
                    if verbose:
                        print(f"Running: {' '.join(download_cmd)}")
                    
                    try:
                        subprocess.run(download_cmd, check=True, capture_output=not verbose)
                        download_count += 1
                    except subprocess.CalledProcessError as e:
                        print(f"Error downloading {dir_name}: {e}")
            
            print(f"Downloaded {download_count} experiment directories.")
            return download_count > 0
            
        except subprocess.CalledProcessError as e:
            print(f"Error listing Modal volume contents: {e}")
            if e.stderr:
                print(f"Error details: {e.stderr.decode()}")
            return False
    
    try:
        result = subprocess.run(command, check=True, capture_output=not verbose)
        if list_only and not verbose and result.stdout:
            print("Volume contents:")
            print(result.stdout.decode())
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error with Modal volume operation: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr.decode()}")
        return False

def parse_experiment_name(directory: str) -> Dict[str, Any]:
    """Parse an experiment directory name to extract model information"""
    parts = directory.split('_')
    
    if 'multimodal' in directory:
        # Format: multimodal_model_audio_fusion_timestamp
        experiment_type = "multimodal"
        try:
            text_model = '_'.join(parts[1:-3])
            audio_type = parts[-3]
            fusion_type = parts[-2]
            timestamp = parts[-1]
            return {
                "type": experiment_type,
                "text_model": text_model,
                "audio_feature": audio_type,
                "fusion_type": fusion_type,
                "timestamp": timestamp
            }
        except:
            return {"type": "multimodal", "name": directory}
    else:
        # Format: text_model_model_timestamp
        experiment_type = "text"
        try:
            text_model = '_'.join(parts[1:-1])
            timestamp = parts[-1]
            return {
                "type": experiment_type,
                "text_model": text_model,
                "timestamp": timestamp
            }
        except:
            return {"type": "text", "name": directory}

def load_experiment_results(directory: str, results_dir: str = RESULTS_DIR) -> Dict[str, Any]:
    """Load all results from an experiment directory"""
    full_path = os.path.join(results_dir, directory)
    
    results = {
        "directory": directory,
        "metadata": parse_experiment_name(directory),
        "vad_final_results": None,
        "training_log": None,
        "ml_classifier_results": None,
    }
    
    # Load VAD final results
    final_results_path = os.path.join(full_path, "logs", "final_results.json")
    if os.path.exists(final_results_path):
        try:
            with open(final_results_path, 'r') as f:
                results["vad_final_results"] = json.load(f)
        except Exception as e:
            print(f"Error loading {final_results_path}: {e}")
    
    # Load training log
    training_log_path = os.path.join(full_path, "logs", "training_log.json")
    if os.path.exists(training_log_path):
        try:
            with open(training_log_path, 'r') as f:
                results["training_log"] = json.load(f)
        except Exception as e:
            print(f"Error loading {training_log_path}: {e}")
    
    # Load ML classifier results
    ml_results_path = os.path.join(full_path, "ml_classifier_results.json")
    if os.path.exists(ml_results_path):
        try:
            with open(ml_results_path, 'r') as f:
                results["ml_classifier_results"] = json.load(f)
        except Exception as e:
            print(f"Error loading {ml_results_path}: {e}")
    
    return results

def collect_all_experiment_results(results_dir: str = RESULTS_DIR) -> List[Dict[str, Any]]:
    """Collect results from all experiments in the directory"""
    all_results = []
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Get all experiment directories
    try:
        directories = [d for d in os.listdir(results_dir) 
                      if os.path.isdir(os.path.join(results_dir, d)) and 
                      (d.startswith('text_model_') or d.startswith('multimodal_'))]
        
        # Load results from each directory
        for directory in directories:
            results = load_experiment_results(directory, results_dir)
            all_results.append(results)
            
        print(f"Collected results from {len(all_results)} experiments")
    except Exception as e:
        print(f"Error collecting experiment results: {e}")
    
    return all_results

def extract_metrics_for_comparison(all_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract metrics from all experiments for comparison"""
    comparison_data = []
    
    for result in all_results:
        metadata = result["metadata"]
        experiment_type = metadata.get("type")
        
        # Basic experiment info
        record = {
            "directory": result["directory"],
            "experiment_type": experiment_type,
            "text_model": metadata.get("text_model", "unknown"),
        }
        
        # Add audio and fusion info for multimodal experiments
        if experiment_type == "multimodal":
            record["audio_feature"] = metadata.get("audio_feature", "unknown")
            record["fusion_type"] = metadata.get("fusion_type", "unknown")
        
        # VAD metrics (Stage 1)
        if result["vad_final_results"]:
            metrics = result["vad_final_results"].get("final_metrics", {})
            
            # Add VAD metrics if available
            for dim in ["Valence", "Arousal", "Dominance"]:
                if dim in metrics:
                    for metric_name, value in metrics[dim].items():
                        record[f"{dim.lower()}_{metric_name.lower()}"] = value
            
            # Add overall test loss
            record["vad_test_loss"] = metrics.get("Test Loss")
            
            # Add best validation loss
            record["best_val_loss"] = result["vad_final_results"].get("best_val_loss")
        
        # ML classifier metrics (Stage 2)
        if result["ml_classifier_results"]:
            best_accuracy = 0
            best_f1 = 0
            best_clf = ""
            
            for clf_result in result["ml_classifier_results"]:
                clf_name = clf_result.get("classifier", "unknown")
                accuracy = clf_result.get("accuracy", 0)
                weighted_f1 = clf_result.get("weighted_f1", 0)
                
                # Track best classifier
                if weighted_f1 > best_f1:
                    best_f1 = weighted_f1
                    best_accuracy = accuracy
                    best_clf = clf_name
                
                # Add individual classifier metrics
                record[f"clf_{clf_name}_accuracy"] = accuracy
                record[f"clf_{clf_name}_f1"] = weighted_f1
            
            # Add best classifier info
            record["best_classifier"] = best_clf
            record["best_classifier_accuracy"] = best_accuracy
            record["best_classifier_f1"] = best_f1
        
        comparison_data.append(record)
    
    return comparison_data

def generate_training_curves(all_results: List[Dict[str, Any]], output_dir: str = REPORTS_DIR) -> str:
    """
    Generate training curve plots for each experiment
    
    Returns:
        str: Path to curves directory
    """
    os.makedirs(output_dir, exist_ok=True)
    curves_path = os.path.join(output_dir, "training_curves")
    os.makedirs(curves_path, exist_ok=True)
    
    for result in all_results:
        # Skip if no training log
        if not result["training_log"] or "epoch_logs" not in result["training_log"]:
            continue
        
        # Extract experiment info
        metadata = result["metadata"]
        experiment_type = metadata.get("type")
        model_name = metadata.get("text_model", "unknown")
        
        # Create plot title and filename
        if experiment_type == "multimodal":
            audio_feature = metadata.get("audio_feature", "unknown")
            fusion_type = metadata.get("fusion_type", "unknown")
            title = f"Multimodal Model: {model_name}\nAudio: {audio_feature}, Fusion: {fusion_type}"
            filename = f"{result['directory']}_training_curve.png"
        else:
            title = f"Text Model: {model_name}"
            filename = f"{result['directory']}_training_curve.png"
        
        # Extract epoch data
        epoch_logs = result["training_log"]["epoch_logs"]
        epochs = [log.get("epoch", i+1) for i, log in enumerate(epoch_logs)]
        train_losses = [log.get("train_loss", 0) for log in epoch_logs]
        val_losses = [log.get("val_loss", 0) for log in epoch_logs]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plt.savefig(os.path.join(curves_path, filename))
        plt.close()
    
    return curves_path

def generate_comparison_tables(comparison_data: List[Dict[str, Any]], output_dir: str = REPORTS_DIR) -> str:
    """
    Generate comparison tables for all experiments
    
    Returns:
        str: Path to main comparison table
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Save full comparison data
    full_table_path = os.path.join(output_dir, "full_comparison.csv")
    df.to_csv(full_table_path, index=False)
    
    # Create separate tables for text and multimodal experiments
    if 'experiment_type' in df.columns:
        text_df = df[df['experiment_type'] == 'text']
        multimodal_df = df[df['experiment_type'] == 'multimodal']
        
        # Save separate tables if they have data
        if len(text_df) > 0:
            text_table_path = os.path.join(output_dir, "text_model_comparison.csv")
            text_df.to_csv(text_table_path, index=False)
        
        if len(multimodal_df) > 0:
            multimodal_table_path = os.path.join(output_dir, "multimodal_comparison.csv")
            multimodal_df.to_csv(multimodal_table_path, index=False)
    
    return full_table_path

def generate_epoch_metrics_table(experiment_dir: str, results: Dict[str, Any], results_dir: str = RESULTS_DIR) -> Optional[str]:
    """Generate a table of per-epoch metrics for a single experiment"""
    if "training_log" not in results or "epoch_logs" not in results["training_log"]:
        return None
    
    # Extract epoch logs
    epoch_logs = results["training_log"]["epoch_logs"]
    
    # Create a dataframe from the epoch logs
    epoch_data = []
    for log in epoch_logs:
        epoch_data.append({
            "Epoch": log.get("epoch"),
            "Train Loss": log.get("train_loss"),
            "Val Loss": log.get("val_loss")
        })
    
    df = pd.DataFrame(epoch_data)
    
    # Save to CSV
    output_path = os.path.join(results_dir, experiment_dir, "epoch_metrics.csv")
    df.to_csv(output_path, index=False)
    
    print(f"Saved epoch metrics to {output_path}")
    return output_path

def generate_stage1_metrics_table(experiment_dir: str, results: Dict[str, Any], results_dir: str = RESULTS_DIR) -> Optional[str]:
    """Generate a table of Stage 1 (VAD prediction) metrics"""
    vad_results_key = "vad_final_results" if "vad_final_results" in results else "final_metrics"
    
    if vad_results_key not in results:
        return None
    
    metrics = results[vad_results_key].get("final_metrics", {})
    
    # Extract VAD metrics
    stage1_data = []
    for dim in ["Valence", "Arousal", "Dominance"]:
        if dim in metrics:
            dim_metrics = metrics[dim]
            stage1_data.append({
                "Dimension": dim,
                "MSE": dim_metrics.get("MSE"),
                "RMSE": dim_metrics.get("RMSE"),
                "MAE": dim_metrics.get("MAE"),
                "R²": dim_metrics.get("R2")
            })
    
    df = pd.DataFrame(stage1_data)
    
    # Save to CSV
    output_path = os.path.join(results_dir, experiment_dir, "stage1_metrics.csv")
    df.to_csv(output_path, index=False)
    
    print(f"Saved Stage 1 metrics to {output_path}")
    return output_path

def generate_stage2_metrics_table(experiment_dir: str, results: Dict[str, Any], results_dir: str = RESULTS_DIR) -> Optional[str]:
    """Generate a table of Stage 2 (emotion classification) metrics"""
    if "ml_classifier_results" not in results:
        return None
    
    ml_results = results["ml_classifier_results"]
    
    # Skip if ml_results is None or empty
    if not ml_results:
        print(f"No ML classifier results found for {experiment_dir}")
        return None
    
    # Extract metrics for each classifier
    stage2_data = []
    for clf_result in ml_results:
        stage2_data.append({
            "Classifier": clf_result.get("classifier"),
            "Accuracy": clf_result.get("accuracy"),
            "Weighted F1": clf_result.get("weighted_f1"),
            "Macro F1": clf_result.get("macro_f1")
        })
    
    df = pd.DataFrame(stage2_data)
    
    # Save to CSV
    output_path = os.path.join(results_dir, experiment_dir, "stage2_metrics.csv")
    df.to_csv(output_path, index=False)
    
    print(f"Saved Stage 2 metrics to {output_path}")
    return output_path

def generate_training_curve(experiment_dir: str, results: Dict[str, Any], results_dir: str = RESULTS_DIR) -> Optional[str]:
    """Generate a training curve plot for a single experiment"""
    if "training_log" not in results or "epoch_logs" not in results["training_log"]:
        return None
    
    # Extract epoch logs
    epoch_logs = results["training_log"]["epoch_logs"]
    
    # Extract epoch numbers and losses
    epochs = [log.get("epoch", i+1) for i, log in enumerate(epoch_logs)]
    train_losses = [log.get("train_loss", 0) for log in epoch_logs]
    val_losses = [log.get("val_loss", 0) for log in epoch_logs]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title(f"Training Curve for {experiment_dir}")
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    output_path = os.path.join(results_dir, experiment_dir, "training_curve.png")
    plt.savefig(output_path)
    plt.close()
    
    print(f"Saved training curve to {output_path}")
    return output_path

def process_results(results_dir: str = RESULTS_DIR, output_dir: str = REPORTS_DIR) -> bool:
    """
    Process all experiment results
    
    Args:
        results_dir: Directory containing raw experiment results
        output_dir: Directory to save processed results
        
    Returns:
        bool: Success status
    """
    # Make sure directories exist
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all experiment results
    all_results = collect_all_experiment_results(results_dir)
    
    if not all_results:
        print("No experiment results found to process")
        return False
    
    # Process each experiment
    experiment_count = 0
    for result in all_results:
        experiment_dir = result["directory"]
        
        # Generate detailed metrics for individual experiments
        generate_epoch_metrics_table(experiment_dir, result, results_dir)
        generate_stage1_metrics_table(experiment_dir, result, results_dir)
        generate_stage2_metrics_table(experiment_dir, result, results_dir)
        generate_training_curve(experiment_dir, result, results_dir)
        
        experiment_count += 1
    
    # Extract metrics for comparison
    comparison_data = extract_metrics_for_comparison(all_results)
    
    # Generate unified training curves
    curves_path = generate_training_curves(all_results, output_dir)
    
    # Generate comparison tables
    tables_path = generate_comparison_tables(comparison_data, output_dir)
    
    # Generate experiment summary CSV
    summary_data = []
    for result in all_results:
        # Extract experiment type and model info
        metadata = result["metadata"]
        experiment_type = metadata.get("type")
        
        summary_record = {
            "Directory": result["directory"],
            "Type": "Text-only" if experiment_type == "text" else "Multimodal",
            "Model": metadata.get("text_model", "unknown")
        }
        
        # Add multimodal-specific info
        if experiment_type == "multimodal":
            summary_record["Audio"] = metadata.get("audio_feature", "")
            summary_record["Fusion"] = metadata.get("fusion_type", "")
        
        # Add metrics
        if result["vad_final_results"]:
            metrics = result["vad_final_results"].get("final_metrics", {})
            
            # Calculate average metrics across VAD dimensions
            vad_mses = []
            vad_r2s = []
            
            for dim in ["Valence", "Arousal", "Dominance"]:
                if dim in metrics:
                    dim_metrics = metrics[dim]
                    if "MSE" in dim_metrics:
                        vad_mses.append(dim_metrics["MSE"])
                    if "R2" in dim_metrics:
                        vad_r2s.append(dim_metrics["R2"])
            
            if vad_mses:
                summary_record["Avg VAD MSE"] = sum(vad_mses) / len(vad_mses)
            if vad_r2s:
                summary_record["Avg VAD R²"] = sum(vad_r2s) / len(vad_r2s)
            
            # Add test loss
            summary_record["VAD Test Loss"] = metrics.get("Test Loss")
        
        # Add Stage 2 best classifier metrics
        if result["ml_classifier_results"]:
            best_clf = None
            best_f1 = -1
            
            for clf_result in result["ml_classifier_results"]:
                if clf_result.get("weighted_f1", 0) > best_f1:
                    best_f1 = clf_result.get("weighted_f1", 0)
                    best_clf = clf_result
            
            if best_clf:
                summary_record["Best Classifier"] = best_clf.get("classifier")
                summary_record["Best Accuracy"] = best_clf.get("accuracy")
                summary_record["Best F1"] = best_clf.get("weighted_f1")
        
        summary_data.append(summary_record)
    
    # Save experiment summary
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(results_dir, "experiment_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved experiment summary to {summary_path}")
    
    print(f"Processed {experiment_count} experiments")
    return True

if __name__ == "__main__":
    # This can be run directly for testing
    if len(sys.argv) > 1 and sys.argv[1] == "download":
        download_results(list_only="--list-only" in sys.argv)
    else:
        process_results() 