#!/usr/bin/env python3
"""
Enhanced logging for EMOD model experiments.
This script collects all results from Modal experiments and provides comprehensive metrics.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Define the path to the Modal results
VOLUME_PATH = "./results"  # Local path (will be mounted to Modal volume)

def collect_experiment_results():
    """Collect all experiment results from the volume"""
    experiment_dirs = []
    
    # Create the results directory if it doesn't exist
    os.makedirs(VOLUME_PATH, exist_ok=True)
    
    # List all experiment directories
    for item in os.listdir(VOLUME_PATH):
        item_path = os.path.join(VOLUME_PATH, item)
        if os.path.isdir(item_path) and ('text_model_' in item or 'multimodal_' in item):
            experiment_dirs.append(item)
    
    print(f"Found {len(experiment_dirs)} experiment directories")
    return experiment_dirs

def extract_logs_from_experiment(experiment_dir):
    """Extract detailed logs from a single experiment directory"""
    full_path = os.path.join(VOLUME_PATH, experiment_dir)
    results = {}
    
    # Load final metrics
    final_results_path = os.path.join(full_path, "logs", "final_results.json")
    if os.path.exists(final_results_path):
        try:
            with open(final_results_path, 'r') as f:
                results["final_metrics"] = json.load(f)
        except Exception as e:
            print(f"Error loading final results from {experiment_dir}: {e}")
    
    # Load training logs
    training_log_path = os.path.join(full_path, "logs", "training_log.json")
    if os.path.exists(training_log_path):
        try:
            with open(training_log_path, 'r') as f:
                results["training_log"] = json.load(f)
        except Exception as e:
            print(f"Error loading training log from {experiment_dir}: {e}")
    
    # Load ML classifier results
    ml_results_path = os.path.join(full_path, "ml_classifier_results.json")
    if os.path.exists(ml_results_path):
        try:
            with open(ml_results_path, 'r') as f:
                results["ml_classifier_results"] = json.load(f)
        except Exception as e:
            print(f"Error loading ML classifier results from {experiment_dir}: {e}")
    
    return results

def generate_epoch_metrics_table(experiment_dir, results):
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
    output_path = os.path.join(VOLUME_PATH, experiment_dir, "epoch_metrics.csv")
    df.to_csv(output_path, index=False)
    
    print(f"Saved epoch metrics to {output_path}")
    return output_path

def generate_stage1_metrics_table(experiment_dir, results):
    """Generate a table of Stage 1 (VAD prediction) metrics"""
    if "final_metrics" not in results:
        return None
    
    metrics = results["final_metrics"].get("final_metrics", {})
    
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
    output_path = os.path.join(VOLUME_PATH, experiment_dir, "stage1_metrics.csv")
    df.to_csv(output_path, index=False)
    
    print(f"Saved Stage 1 metrics to {output_path}")
    return output_path

def generate_stage2_metrics_table(experiment_dir, results):
    """Generate a table of Stage 2 (emotion classification) metrics"""
    if "ml_classifier_results" not in results:
        return None
    
    ml_results = results["ml_classifier_results"]
    
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
    output_path = os.path.join(VOLUME_PATH, experiment_dir, "stage2_metrics.csv")
    df.to_csv(output_path, index=False)
    
    print(f"Saved Stage 2 metrics to {output_path}")
    return output_path

def generate_training_curve(experiment_dir, results):
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
    output_path = os.path.join(VOLUME_PATH, experiment_dir, "training_curve.png")
    plt.savefig(output_path)
    plt.close()
    
    print(f"Saved training curve to {output_path}")
    return output_path

def generate_summary_report():
    """Generate a summary report of all experiment results"""
    experiment_dirs = collect_experiment_results()
    
    # Initialize summary data
    summary_data = []
    
    # Process each experiment
    for exp_dir in experiment_dirs:
        # Extract experiment type and model info from directory name
        exp_info = {}
        if "text_model_" in exp_dir:
            exp_info["Type"] = "Text-only"
            parts = exp_dir.split("text_model_")[1].split("_")
            exp_info["Model"] = '_'.join(parts[:-1]) if len(parts) > 1 else parts[0]
        elif "multimodal_" in exp_dir:
            exp_info["Type"] = "Multimodal"
            parts = exp_dir.split("multimodal_")[1].split("_")
            if len(parts) >= 4:  # model_audio_fusion_timestamp format
                exp_info["Model"] = parts[0]
                exp_info["Audio"] = parts[1]
                exp_info["Fusion"] = parts[2]
        
        # Extract logs
        results = extract_logs_from_experiment(exp_dir)
        
        # Skip if no results were found
        if not results:
            continue
        
        # Generate detailed metrics tables
        generate_epoch_metrics_table(exp_dir, results)
        generate_stage1_metrics_table(exp_dir, results)
        generate_stage2_metrics_table(exp_dir, results)
        generate_training_curve(exp_dir, results)
        
        # Extract key metrics for summary
        summary_record = {**exp_info, "Directory": exp_dir}
        
        # Add Stage 1 metrics (VAD prediction)
        if "final_metrics" in results:
            metrics = results["final_metrics"].get("final_metrics", {})
            
            # Extract average metrics across all dimensions
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
            if "Test Loss" in metrics:
                summary_record["VAD Test Loss"] = metrics["Test Loss"]
        
        # Add Stage 2 metrics (emotion classification)
        if "ml_classifier_results" in results:
            ml_results = results["ml_classifier_results"]
            
            # Find best classifier by weighted F1
            best_clf = None
            best_f1 = -1
            
            for clf_result in ml_results:
                if clf_result.get("weighted_f1", 0) > best_f1:
                    best_f1 = clf_result.get("weighted_f1", 0)
                    best_clf = clf_result
            
            if best_clf:
                summary_record["Best Classifier"] = best_clf.get("classifier")
                summary_record["Best Accuracy"] = best_clf.get("accuracy")
                summary_record["Best F1"] = best_clf.get("weighted_f1")
        
        summary_data.append(summary_record)
    
    # Create summary CSV
    if summary_data:
        df = pd.DataFrame(summary_data)
        output_path = os.path.join(VOLUME_PATH, "experiment_summary.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved experiment summary to {output_path}")
    
    return summary_data

if __name__ == "__main__":
    summary = generate_summary_report()
    print(f"Processed {len(summary)} experiments")
    print("All results have been logged and summarized.") 