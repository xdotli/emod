#!/usr/bin/env python3
"""
Script to compare the performance of text-only and multimodal approaches.
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compare emotion recognition models')
    
    parser.add_argument('--text_results_dir', type=str, required=True,
                        help='Directory containing the text-only results')
    parser.add_argument('--multimodal_results_dir', type=str, required=True,
                        help='Directory containing the multimodal results')
    parser.add_argument('--output_dir', type=str, default='comparisons',
                        help='Directory to save comparison visualizations')
    
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

def compare_emotion_accuracy(text_metrics, multimodal_metrics, output_dir):
    """Compare emotion accuracy between text-only and multimodal approaches."""
    if 'emotion_metrics' not in text_metrics or 'emotion_metrics' not in multimodal_metrics:
        print("Emotion metrics not found in one or both metrics files")
        return
    
    # Get accuracy
    text_accuracy = text_metrics['emotion_metrics']['accuracy']
    multimodal_accuracy = multimodal_metrics['emotion_metrics']['accuracy']
    
    # Get F1 scores
    text_f1_macro = text_metrics['emotion_metrics']['f1_macro']
    multimodal_f1_macro = multimodal_metrics['emotion_metrics']['f1_macro']
    
    text_f1_weighted = text_metrics['emotion_metrics']['f1_weighted']
    multimodal_f1_weighted = multimodal_metrics['emotion_metrics']['f1_weighted']
    
    # Create DataFrame
    data = {
        'Metric': ['Accuracy', 'F1 (macro)', 'F1 (weighted)'],
        'Text-only': [text_accuracy, text_f1_macro, text_f1_weighted],
        'Multimodal': [multimodal_accuracy, multimodal_f1_macro, multimodal_f1_weighted]
    }
    
    df = pd.DataFrame(data)
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(df['Metric']))
    width = 0.35
    
    plt.bar(x - width/2, df['Text-only'], width, label='Text-only')
    plt.bar(x + width/2, df['Multimodal'], width, label='Multimodal')
    
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title('Emotion Recognition Performance Comparison')
    plt.xticks(x, df['Metric'])
    plt.legend()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'emotion_accuracy_comparison.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"Emotion accuracy comparison saved to {output_path}")
    
    # Calculate improvement
    improvement = {
        'Accuracy': (multimodal_accuracy - text_accuracy) / text_accuracy * 100,
        'F1 (macro)': (multimodal_f1_macro - text_f1_macro) / text_f1_macro * 100,
        'F1 (weighted)': (multimodal_f1_weighted - text_f1_weighted) / text_f1_weighted * 100
    }
    
    print("\nImprovement with multimodal approach:")
    for metric, value in improvement.items():
        print(f"{metric}: {value:.2f}%")

def compare_vad_metrics(text_metrics, multimodal_metrics, output_dir):
    """Compare VAD metrics between text-only and multimodal approaches."""
    if 'vad_metrics' not in text_metrics or 'text_vad_metrics' not in multimodal_metrics:
        print("VAD metrics not found in one or both metrics files")
        return
    
    # Get MSE
    text_mse = text_metrics['vad_metrics']['mse']
    multimodal_mse = multimodal_metrics['text_vad_metrics']['mse']
    
    # Get RMSE
    text_rmse = text_metrics['vad_metrics']['rmse']
    multimodal_rmse = multimodal_metrics['text_vad_metrics']['rmse']
    
    # Get MAE
    text_mae = text_metrics['vad_metrics']['mae']
    multimodal_mae = multimodal_metrics['text_vad_metrics']['mae']
    
    # Create DataFrame
    data = {
        'Metric': ['MSE', 'RMSE', 'MAE'],
        'Text-only': [text_mse, text_rmse, text_mae],
        'Multimodal': [multimodal_mse, multimodal_rmse, multimodal_mae]
    }
    
    df = pd.DataFrame(data)
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(df['Metric']))
    width = 0.35
    
    plt.bar(x - width/2, df['Text-only'], width, label='Text-only')
    plt.bar(x + width/2, df['Multimodal'], width, label='Multimodal')
    
    plt.xlabel('Metric')
    plt.ylabel('Error')
    plt.title('VAD Prediction Error Comparison')
    plt.xticks(x, df['Metric'])
    plt.legend()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'vad_metrics_comparison.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"VAD metrics comparison saved to {output_path}")
    
    # Calculate improvement
    improvement = {
        'MSE': (text_mse - multimodal_mse) / text_mse * 100,
        'RMSE': (text_rmse - multimodal_rmse) / text_rmse * 100,
        'MAE': (text_mae - multimodal_mae) / text_mae * 100
    }
    
    print("\nError reduction with multimodal approach:")
    for metric, value in improvement.items():
        print(f"{metric}: {value:.2f}%")

def compare_emotion_class_performance(text_metrics, multimodal_metrics, output_dir):
    """Compare emotion class performance between text-only and multimodal approaches."""
    if ('emotion_metrics' not in text_metrics or 
        'classification_report' not in text_metrics['emotion_metrics'] or
        'emotion_metrics' not in multimodal_metrics or
        'classification_report' not in multimodal_metrics['emotion_metrics']):
        print("Classification report not found in one or both metrics files")
        return
    
    # Get classification reports
    text_report = text_metrics['emotion_metrics']['classification_report']
    multimodal_report = multimodal_metrics['emotion_metrics']['classification_report']
    
    # Create DataFrame
    data = []
    
    # Get common classes
    common_classes = set()
    for class_name in text_report.keys():
        if (isinstance(text_report[class_name], dict) and 
            'f1-score' in text_report[class_name] and
            class_name in multimodal_report and
            isinstance(multimodal_report[class_name], dict) and
            'f1-score' in multimodal_report[class_name] and
            'support' in text_report[class_name] and
            text_report[class_name]['support'] > 10):
            common_classes.add(class_name)
    
    # Add data for common classes
    for class_name in sorted(common_classes):
        data.append({
            'Emotion': class_name,
            'Text-only F1': text_report[class_name]['f1-score'],
            'Multimodal F1': multimodal_report[class_name]['f1-score'],
            'Support': text_report[class_name]['support']
        })
    
    df = pd.DataFrame(data)
    
    # Sort by support
    df = df.sort_values('Support', ascending=False)
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(df))
    width = 0.35
    
    plt.bar(x - width/2, df['Text-only F1'], width, label='Text-only')
    plt.bar(x + width/2, df['Multimodal F1'], width, label='Multimodal')
    
    plt.xlabel('Emotion')
    plt.ylabel('F1-score')
    plt.title('Emotion Recognition F1-score by Class')
    plt.xticks(x, df['Emotion'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'emotion_class_comparison.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"Emotion class comparison saved to {output_path}")
    
    # Calculate improvement
    df['Improvement'] = (df['Multimodal F1'] - df['Text-only F1']) / df['Text-only F1'] * 100
    
    print("\nF1-score improvement by emotion class:")
    for _, row in df.iterrows():
        print(f"{row['Emotion']}: {row['Improvement']:.2f}%")

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load metrics
    text_metrics = load_metrics(args.text_results_dir, prefix='pipeline_')
    multimodal_metrics = load_metrics(args.multimodal_results_dir, prefix='test_')
    
    if text_metrics is None or multimodal_metrics is None:
        print("Could not load metrics from one or both directories")
        return
    
    # Compare emotion accuracy
    compare_emotion_accuracy(text_metrics, multimodal_metrics, args.output_dir)
    
    # Compare VAD metrics
    compare_vad_metrics(text_metrics, multimodal_metrics, args.output_dir)
    
    # Compare emotion class performance
    compare_emotion_class_performance(text_metrics, multimodal_metrics, args.output_dir)
    
    print(f"All comparisons saved to {args.output_dir}")

if __name__ == '__main__':
    main()
