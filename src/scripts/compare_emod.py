#!/usr/bin/env python3
"""
Script to compare text-only and multimodal approaches for EMOD.

This script loads metrics from text-only and multimodal models and generates visualizations
to compare their performance.
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare text-only and multimodal approaches")
    parser.add_argument('--text_results', type=str, required=True,
                        help='Directory containing results from text-only approach')
    parser.add_argument('--multimodal_results', type=str, required=True,
                        help='Directory containing results from multimodal approach')
    parser.add_argument('--output_dir', type=str, default='comparisons',
                        help='Directory to save comparison visualizations')
    
    return parser.parse_args()

def load_metrics(results_dir):
    """Load metrics from a results directory."""
    metrics_path = os.path.join(results_dir, 'metrics.json')
    if not os.path.exists(metrics_path):
        print(f"Metrics file not found: {metrics_path}")
        return None
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics

def compare_vad_prediction(text_metrics, multimodal_metrics, output_dir):
    """Compare VAD prediction performance."""
    print("Comparing VAD prediction performance...")
    
    # Prepare data for plotting
    metrics_names = ['MSE', 'RMSE', 'MAE']
    dimensions = ['Valence', 'Arousal', 'Dominance']
    
    # Create a figure for each metric
    for i, metric_name in enumerate(['mse', 'rmse', 'mae']):
        plt.figure(figsize=(10, 6))
        
        text_values = np.array(text_metrics['vad_metrics'][metric_name])
        multimodal_values = np.array(multimodal_metrics['vad_metrics'][metric_name])
        
        x = np.arange(len(dimensions))
        width = 0.35
        
        plt.bar(x - width/2, text_values, width, label='Text-only')
        plt.bar(x + width/2, multimodal_values, width, label='Multimodal')
        
        plt.xlabel('VAD Dimension', fontsize=12)
        plt.ylabel(f'{metrics_names[i]} Value', fontsize=12)
        plt.title(f'{metrics_names[i]} Comparison: Text-only vs Multimodal', fontsize=14)
        plt.xticks(x, dimensions, fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add percentage improvement
        for j in range(len(dimensions)):
            improvement = (text_values[j] - multimodal_values[j]) / text_values[j] * 100
            if improvement > 0:  # Improvement (lower error is better)
                plt.text(j, max(text_values[j], multimodal_values[j]) * 1.05, 
                        f"{improvement:.1f}%↓", 
                        ha='center', fontsize=10, color='green')
            else:  # Deterioration
                plt.text(j, max(text_values[j], multimodal_values[j]) * 1.05, 
                        f"{-improvement:.1f}%↑", 
                        ha='center', fontsize=10, color='red')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'vad_{metric_name}_comparison.png'))
        plt.close()
    
    # Compare R² scores
    plt.figure(figsize=(10, 6))
    
    text_r2 = np.array(text_metrics['vad_metrics']['r2'])
    multimodal_r2 = np.array(multimodal_metrics['vad_metrics']['r2'])
    
    x = np.arange(len(dimensions))
    width = 0.35
    
    plt.bar(x - width/2, text_r2, width, label='Text-only')
    plt.bar(x + width/2, multimodal_r2, width, label='Multimodal')
    
    plt.xlabel('VAD Dimension', fontsize=12)
    plt.ylabel('R² Score', fontsize=12)
    plt.title('R² Score Comparison: Text-only vs Multimodal', fontsize=14)
    plt.xticks(x, dimensions, fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add percentage improvement
    for j in range(len(dimensions)):
        # For R², higher is better
        improvement = (multimodal_r2[j] - text_r2[j]) / abs(text_r2[j]) * 100 if text_r2[j] != 0 else np.inf
        if improvement > 0:  # Improvement
            plt.text(j, max(text_r2[j], multimodal_r2[j]) * 1.05, 
                    f"{improvement:.1f}%↑", 
                    ha='center', fontsize=10, color='green')
        else:  # Deterioration
            plt.text(j, max(text_r2[j], multimodal_r2[j]) * 1.05, 
                    f"{-improvement:.1f}%↓", 
                    ha='center', fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vad_r2_comparison.png'))
    plt.close()

def compare_emotion_classification(text_metrics, multimodal_metrics, output_dir):
    """Compare emotion classification performance."""
    print("Comparing emotion classification performance...")
    
    # Prepare data for plotting
    metrics = ['accuracy', 'f1_weighted', 'f1_macro']
    metric_labels = ['Accuracy', 'F1 (weighted)', 'F1 (macro)']
    
    plt.figure(figsize=(10, 6))
    
    text_values = [text_metrics['emotion_metrics'][m] for m in metrics]
    multimodal_values = [multimodal_metrics['emotion_metrics'][m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, text_values, width, label='Text-only')
    plt.bar(x + width/2, multimodal_values, width, label='Multimodal')
    
    plt.xlabel('Metric', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Emotion Classification Performance: Text-only vs Multimodal', fontsize=14)
    plt.xticks(x, metric_labels, fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add percentage improvement
    for j in range(len(metrics)):
        improvement = (multimodal_values[j] - text_values[j]) / text_values[j] * 100
        if improvement > 0:  # Improvement
            plt.text(j, max(text_values[j], multimodal_values[j]) * 1.05, 
                    f"{improvement:.1f}%↑", 
                    ha='center', fontsize=10, color='green')
        else:  # Deterioration
            plt.text(j, max(text_values[j], multimodal_values[j]) * 1.05, 
                    f"{-improvement:.1f}%↓", 
                    ha='center', fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'emotion_performance_comparison.png'))
    plt.close()
    
    # Compare per-class F1 scores
    try:
        text_report = text_metrics['emotion_metrics']['classification_report']
        multimodal_report = multimodal_metrics['emotion_metrics']['classification_report']
        
        # Get common classes
        common_classes = []
        for cls in text_report:
            if cls not in ['accuracy', 'macro avg', 'weighted avg'] and cls in multimodal_report:
                common_classes.append(cls)
        
        if common_classes:
            plt.figure(figsize=(12, 6))
            
            text_f1 = [text_report[cls]['f1-score'] for cls in common_classes]
            multimodal_f1 = [multimodal_report[cls]['f1-score'] for cls in common_classes]
            
            x = np.arange(len(common_classes))
            width = 0.35
            
            plt.bar(x - width/2, text_f1, width, label='Text-only')
            plt.bar(x + width/2, multimodal_f1, width, label='Multimodal')
            
            plt.xlabel('Emotion Class', fontsize=12)
            plt.ylabel('F1 Score', fontsize=12)
            plt.title('Emotion Classification F1 Score by Class', fontsize=14)
            plt.xticks(x, [cls.capitalize() for cls in common_classes], fontsize=10)
            plt.legend(fontsize=10)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add percentage improvement
            for j in range(len(common_classes)):
                improvement = (multimodal_f1[j] - text_f1[j]) / text_f1[j] * 100 if text_f1[j] != 0 else np.inf
                if improvement > 0:  # Improvement
                    plt.text(j, max(text_f1[j], multimodal_f1[j]) * 1.05, 
                            f"{improvement:.1f}%↑", 
                            ha='center', fontsize=10, color='green')
                else:  # Deterioration
                    plt.text(j, max(text_f1[j], multimodal_f1[j]) * 1.05, 
                            f"{-improvement:.1f}%↓", 
                            ha='center', fontsize=10, color='red')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'emotion_f1_by_class.png'))
            plt.close()
    except Exception as e:
        print(f"Error comparing per-class F1 scores: {e}")

def create_summary_table(text_metrics, multimodal_metrics, output_dir):
    """Create a summary table of improvements."""
    print("Creating summary table...")
    
    # VAD prediction improvements
    vad_metrics = ['MSE', 'RMSE', 'MAE', 'R²']
    dimensions = ['Valence', 'Arousal', 'Dominance']
    
    # Emotion classification improvements
    emo_metrics = ['Accuracy', 'F1 (weighted)', 'F1 (macro)']
    
    # Create DataFrame for VAD metrics
    vad_data = []
    
    for i, metric in enumerate(['mse', 'rmse', 'mae', 'r2']):
        text_values = np.array(text_metrics['vad_metrics'][metric])
        multimodal_values = np.array(multimodal_metrics['vad_metrics'][metric])
        
        for j, dim in enumerate(dimensions):
            if metric == 'r2':  # For R², higher is better
                impr = (multimodal_values[j] - text_values[j]) / abs(text_values[j]) * 100 if text_values[j] != 0 else np.inf
            else:  # For error metrics, lower is better
                impr = (text_values[j] - multimodal_values[j]) / text_values[j] * 100
            
            vad_data.append({
                'Category': 'VAD Prediction',
                'Dimension': dim,
                'Metric': vad_metrics[i],
                'Text-only': text_values[j],
                'Multimodal': multimodal_values[j],
                'Improvement (%)': impr
            })
    
    # Create DataFrame for emotion metrics
    emo_data = []
    metrics_map = {'accuracy': 'Accuracy', 'f1_weighted': 'F1 (weighted)', 'f1_macro': 'F1 (macro)'}
    
    for metric, label in metrics_map.items():
        text_value = text_metrics['emotion_metrics'][metric]
        multimodal_value = multimodal_metrics['emotion_metrics'][metric]
        impr = (multimodal_value - text_value) / text_value * 100
        
        emo_data.append({
            'Category': 'Emotion Classification',
            'Dimension': 'Overall',
            'Metric': label,
            'Text-only': text_value,
            'Multimodal': multimodal_value,
            'Improvement (%)': impr
        })
    
    # Combine and create table
    all_data = pd.DataFrame(vad_data + emo_data)
    
    # Save as CSV
    all_data.to_csv(os.path.join(output_dir, 'performance_summary.csv'), index=False)
    
    # Create a formatted text summary
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write("EMOD: Text-only vs Multimodal Performance Comparison\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("VAD PREDICTION PERFORMANCE\n")
        f.write("-" * 60 + "\n")
        for dim in dimensions:
            f.write(f"\n{dim} dimension:\n")
            dim_data = all_data[(all_data['Category'] == 'VAD Prediction') & (all_data['Dimension'] == dim)]
            for _, row in dim_data.iterrows():
                f.write(f"  {row['Metric']}: {row['Text-only']:.4f} (text) vs {row['Multimodal']:.4f} (multimodal) = ")
                if row['Improvement (%)'] > 0:
                    f.write(f"{row['Improvement (%)']:.2f}% improvement\n")
                else:
                    f.write(f"{-row['Improvement (%)']:.2f}% deterioration\n")
        
        f.write("\nEMOTION CLASSIFICATION PERFORMANCE\n")
        f.write("-" * 60 + "\n")
        emo_data = all_data[all_data['Category'] == 'Emotion Classification']
        for _, row in emo_data.iterrows():
            f.write(f"  {row['Metric']}: {row['Text-only']:.4f} (text) vs {row['Multimodal']:.4f} (multimodal) = ")
            if row['Improvement (%)'] > 0:
                f.write(f"{row['Improvement (%)']:.2f}% improvement\n")
            else:
                f.write(f"{-row['Improvement (%)']:.2f}% deterioration\n")
                
        # Try to add per-class emotion performance
        try:
            text_report = text_metrics['emotion_metrics']['classification_report']
            multimodal_report = multimodal_metrics['emotion_metrics']['classification_report']
            
            f.write("\nPER-CLASS EMOTION PERFORMANCE (F1 Score)\n")
            f.write("-" * 60 + "\n")
            
            for cls in text_report:
                if cls not in ['accuracy', 'macro avg', 'weighted avg'] and cls in multimodal_report:
                    text_f1 = text_report[cls]['f1-score']
                    multimodal_f1 = multimodal_report[cls]['f1-score']
                    impr = (multimodal_f1 - text_f1) / text_f1 * 100 if text_f1 != 0 else float('inf')
                    
                    f.write(f"  {cls.capitalize()}: {text_f1:.4f} (text) vs {multimodal_f1:.4f} (multimodal) = ")
                    if impr > 0:
                        f.write(f"{impr:.2f}% improvement\n")
                    else:
                        f.write(f"{-impr:.2f}% deterioration\n")
                        
        except Exception as e:
            f.write(f"\nError adding per-class emotion performance: {e}\n")
    
    print(f"Summary table saved to {output_dir}/performance_summary.csv")
    print(f"Text summary saved to {output_dir}/summary.txt")

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load metrics
    print(f"Loading metrics from {args.text_results}...")
    text_metrics = load_metrics(args.text_results)
    
    print(f"Loading metrics from {args.multimodal_results}...")
    multimodal_metrics = load_metrics(args.multimodal_results)
    
    if not text_metrics or not multimodal_metrics:
        print("Failed to load metrics from one or both approaches.")
        return
    
    # Compare VAD prediction performance
    compare_vad_prediction(text_metrics, multimodal_metrics, args.output_dir)
    
    # Compare emotion classification performance
    compare_emotion_classification(text_metrics, multimodal_metrics, args.output_dir)
    
    # Create summary table
    create_summary_table(text_metrics, multimodal_metrics, args.output_dir)
    
    print(f"All comparisons saved to {args.output_dir}")

if __name__ == '__main__':
    main() 