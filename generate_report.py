#!/usr/bin/env python3
"""
Generate a comprehensive report of all EMOD experiment results.
This script analyzes all results in the Modal volume and creates detailed reports.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Define the path to the results volume
VOLUME_PATH = "/root/results" if os.path.exists("/root/results") else "./results"

def parse_experiment_name(directory):
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

def load_experiment_results(directory):
    """Load all results from an experiment directory"""
    full_path = os.path.join(VOLUME_PATH, directory)
    
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
        except:
            print(f"Error loading {final_results_path}")
    
    # Load training log
    training_log_path = os.path.join(full_path, "logs", "training_log.json")
    if os.path.exists(training_log_path):
        try:
            with open(training_log_path, 'r') as f:
                results["training_log"] = json.load(f)
        except:
            print(f"Error loading {training_log_path}")
    
    # Load ML classifier results
    ml_results_path = os.path.join(full_path, "ml_classifier_results.json")
    if os.path.exists(ml_results_path):
        try:
            with open(ml_results_path, 'r') as f:
                results["ml_classifier_results"] = json.load(f)
        except:
            print(f"Error loading {ml_results_path}")
    
    return results

def collect_all_experiment_results():
    """Collect results from all experiments in the volume"""
    all_results = []
    
    # Create results directory if it doesn't exist
    os.makedirs(VOLUME_PATH, exist_ok=True)
    
    # Get all experiment directories
    try:
        directories = [d for d in os.listdir(VOLUME_PATH) 
                      if os.path.isdir(os.path.join(VOLUME_PATH, d)) and 
                      (d.startswith('text_model_') or d.startswith('multimodal_'))]
        
        # Load results from each directory
        for directory in directories:
            results = load_experiment_results(directory)
            all_results.append(results)
            
        print(f"Collected results from {len(all_results)} experiments")
    except Exception as e:
        print(f"Error collecting experiment results: {e}")
    
    return all_results

def extract_metrics_for_comparison(all_results):
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

def generate_training_curves(all_results, output_dir="./reports"):
    """Generate training curve plots for each experiment"""
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

def generate_comparison_tables(comparison_data, output_dir="./reports"):
    """Generate comparison tables for all experiments"""
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

def generate_full_report(output_dir="./reports"):
    """Generate a comprehensive report of all experiment results"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all experiment results
    print("Collecting experiment results...")
    all_results = collect_all_experiment_results()
    
    # Extract metrics for comparison
    print("Extracting metrics for comparison...")
    comparison_data = extract_metrics_for_comparison(all_results)
    
    # Generate training curves
    print("Generating training curves...")
    curves_path = generate_training_curves(all_results, output_dir)
    
    # Generate comparison tables
    print("Generating comparison tables...")
    tables_path = generate_comparison_tables(comparison_data, output_dir)
    
    # Generate HTML report
    print("Generating HTML report...")
    html_report_path = os.path.join(output_dir, "experiment_report.html")
    
    # Create basic HTML report
    with open(html_report_path, 'w') as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>EMOD Experiment Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .section {{ margin-bottom: 30px; }}
            </style>
        </head>
        <body>
            <h1>EMOD Experiment Results Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2>Summary</h2>
                <p>Total experiments analyzed: {len(all_results)}</p>
                <p>Text-only models: {sum(1 for r in all_results if r['metadata']['type'] == 'text')}</p>
                <p>Multimodal models: {sum(1 for r in all_results if r['metadata']['type'] == 'multimodal')}</p>
            </div>
            
            <div class="section">
                <h2>Top Performing Models</h2>
                <h3>Stage 1: VAD Prediction (by Valence MSE)</h3>
                {generate_top_models_html(comparison_data, 'valence_mse', 'Valence MSE', ascending=True)}
                
                <h3>Stage 2: Emotion Classification (by F1 Score)</h3>
                {generate_top_models_html(comparison_data, 'best_classifier_f1', 'F1 Score', ascending=False)}
            </div>
            
            <div class="section">
                <h2>All Experiments</h2>
                <p>See the <a href="{os.path.basename(tables_path)}">full comparison table</a> for detailed metrics.</p>
                <p>Training curves are available in the <a href="{os.path.basename(curves_path)}">training_curves</a> directory.</p>
            </div>
        </body>
        </html>
        """)
    
    print(f"Report generation complete. Results saved to {output_dir}")
    return {
        "report_path": html_report_path,
        "tables_path": tables_path,
        "curves_path": curves_path,
        "experiment_count": len(all_results)
    }

def generate_top_models_html(comparison_data, sort_key, metric_name, ascending=True, top_n=5):
    """Generate HTML table for top performing models"""
    if not comparison_data or sort_key not in comparison_data[0]:
        return "<p>No data available</p>"
    
    # Filter out entries with missing values
    filtered_data = [d for d in comparison_data if sort_key in d and d[sort_key] is not None]
    
    # Sort and get top N
    sorted_data = sorted(filtered_data, key=lambda x: x[sort_key], reverse=not ascending)
    top_data = sorted_data[:top_n]
    
    # Generate HTML table
    html = "<table>\n<tr><th>Model</th><th>Type</th>"
    
    # Add multimodal columns if needed
    if any(d.get('experiment_type') == 'multimodal' for d in top_data):
        html += "<th>Audio</th><th>Fusion</th>"
    
    html += f"<th>{metric_name}</th><th>Best Classifier</th><th>Classifier F1</th></tr>\n"
    
    # Add rows
    for d in top_data:
        exp_type = d.get('experiment_type', 'unknown')
        html += f"<tr><td>{d.get('text_model', 'unknown')}</td><td>{exp_type}</td>"
        
        # Add multimodal details if applicable
        if exp_type == 'multimodal':
            html += f"<td>{d.get('audio_feature', '')}</td><td>{d.get('fusion_type', '')}</td>"
        elif any(d.get('experiment_type') == 'multimodal' for d in top_data):
            html += "<td>-</td><td>-</td>"
        
        # Add metrics
        html += f"<td>{d.get(sort_key, 'N/A')}</td>"
        html += f"<td>{d.get('best_classifier', 'N/A')}</td>"
        html += f"<td>{d.get('best_classifier_f1', 'N/A')}</td></tr>\n"
    
    html += "</table>"
    return html

if __name__ == "__main__":
    # Set local volume path if running outside of Modal
    if not os.path.exists(VOLUME_PATH):
        VOLUME_PATH = "./results"
        os.makedirs(VOLUME_PATH, exist_ok=True)
    
    # Generate the report
    report_info = generate_full_report()
    print(f"Report generated successfully. Found {report_info['experiment_count']} experiments.")
    print(f"Report saved to {report_info['report_path']}") 