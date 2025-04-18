#!/usr/bin/env python3
"""
Script to visualize the results of the emotion recognition system.
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize emotion recognition results')
    
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing the results')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                        help='Directory to save visualizations')
    
    return parser.parse_args()

def load_metrics(results_dir, prefix=''):
    """Load metrics from a results directory."""
    metrics_path = os.path.join(results_dir, f'{prefix}metrics.json')
    
    if not os.path.exists(metrics_path):
        print(f"Metrics file not found: {metrics_path}")
        return None
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics

def plot_confusion_matrix(metrics, output_dir, prefix=''):
    """Plot confusion matrix."""
    if 'emotion_metrics' not in metrics:
        print("Emotion metrics not found in the metrics file")
        return
    
    # Get confusion matrix
    conf_matrix = np.array(metrics['emotion_metrics']['confusion_matrix'])
    
    # Get labels
    if 'true_emotion_labels' in metrics and 'pred_emotion_labels' in metrics:
        # Get unique labels
        unique_labels = sorted(set(metrics['true_emotion_labels']))
    else:
        # Use indices as labels
        unique_labels = list(range(conf_matrix.shape[0]))
    
    # Normalize by row (true labels)
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        conf_matrix_norm, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues',
        xticklabels=unique_labels,
        yticklabels=unique_labels
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{prefix}confusion_matrix.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {output_path}")

def plot_vad_distribution(metrics, output_dir, prefix=''):
    """Plot VAD distribution."""
    if 'true_vad' not in metrics or 'pred_vad' not in metrics:
        print("VAD values not found in the metrics file")
        return
    
    true_vad = np.array(metrics['true_vad'])
    pred_vad = np.array(metrics['pred_vad'])
    
    dim_names = ['Valence', 'Arousal', 'Dominance']
    
    for i, dim in enumerate(dim_names):
        plt.figure(figsize=(10, 6))
        
        # Plot histograms
        plt.hist(true_vad[:, i], bins=20, alpha=0.5, label='True')
        plt.hist(pred_vad[:, i], bins=20, alpha=0.5, label='Predicted')
        
        plt.xlabel(dim)
        plt.ylabel('Count')
        plt.title(f'{dim} Distribution')
        plt.legend()
        
        # Save the plot
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{prefix}{dim.lower()}_distribution.png')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        print(f"{dim} distribution saved to {output_path}")

def plot_emotion_accuracy(metrics, output_dir, prefix=''):
    """Plot emotion accuracy by class."""
    if 'emotion_metrics' not in metrics or 'classification_report' not in metrics['emotion_metrics']:
        print("Classification report not found in the metrics file")
        return
    
    # Get classification report
    class_report = metrics['emotion_metrics']['classification_report']
    
    # Create DataFrame
    data = []
    for class_name, class_metrics in class_report.items():
        if isinstance(class_metrics, dict) and 'precision' in class_metrics:
            data.append({
                'Emotion': class_name,
                'Precision': class_metrics['precision'],
                'Recall': class_metrics['recall'],
                'F1-score': class_metrics['f1-score'],
                'Support': class_metrics['support']
            })
    
    df = pd.DataFrame(data)
    
    # Filter out classes with low support
    df = df[df['Support'] > 10]
    
    # Sort by F1-score
    df = df.sort_values('F1-score', ascending=False)
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(df))
    width = 0.25
    
    plt.bar(x - width, df['Precision'], width, label='Precision')
    plt.bar(x, df['Recall'], width, label='Recall')
    plt.bar(x + width, df['F1-score'], width, label='F1-score')
    
    plt.xlabel('Emotion')
    plt.ylabel('Score')
    plt.title('Emotion Recognition Performance by Class')
    plt.xticks(x, df['Emotion'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{prefix}emotion_accuracy.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"Emotion accuracy plot saved to {output_path}")

def plot_vad_metrics(metrics, output_dir, prefix=''):
    """Plot VAD metrics."""
    if 'vad_metrics' not in metrics or 'dim_metrics' not in metrics['vad_metrics']:
        print("VAD metrics not found in the metrics file")
        return
    
    # Get VAD metrics
    dim_metrics = metrics['vad_metrics']['dim_metrics']
    
    # Create DataFrame
    data = []
    for dim, metrics_dict in dim_metrics.items():
        data.append({
            'Dimension': dim.capitalize(),
            'MSE': metrics_dict['mse'],
            'RMSE': metrics_dict['rmse'],
            'MAE': metrics_dict['mae']
        })
    
    df = pd.DataFrame(data)
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(df))
    width = 0.25
    
    plt.bar(x - width, df['MSE'], width, label='MSE')
    plt.bar(x, df['RMSE'], width, label='RMSE')
    plt.bar(x + width, df['MAE'], width, label='MAE')
    
    plt.xlabel('Dimension')
    plt.ylabel('Error')
    plt.title('VAD Prediction Error by Dimension')
    plt.xticks(x, df['Dimension'])
    plt.legend()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{prefix}vad_metrics.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"VAD metrics plot saved to {output_path}")

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load metrics
    metrics = load_metrics(args.results_dir, prefix='pipeline_')
    
    if metrics is None:
        # Try loading emotion metrics
        metrics = load_metrics(args.results_dir, prefix='emotion_')
    
    if metrics is None:
        # Try loading VAD metrics
        metrics = load_metrics(args.results_dir, prefix='vad_')
    
    if metrics is None:
        print(f"No metrics found in {args.results_dir}")
        return
    
    # Plot confusion matrix
    plot_confusion_matrix(metrics, args.output_dir)
    
    # Plot VAD distribution
    plot_vad_distribution(metrics, args.output_dir)
    
    # Plot emotion accuracy
    plot_emotion_accuracy(metrics, args.output_dir)
    
    # Plot VAD metrics
    plot_vad_metrics(metrics, args.output_dir)
    
    print(f"All visualizations saved to {args.output_dir}")

if __name__ == '__main__':
    main()
