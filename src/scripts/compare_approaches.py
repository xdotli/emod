#!/usr/bin/env python3
"""
Script to compare zero-shot and fine-tuned approaches for emotion recognition.
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
    parser = argparse.ArgumentParser(description='Compare zero-shot and fine-tuned approaches')
    
    parser.add_argument('--zero_shot_dir', type=str, required=True,
                        help='Directory containing zero-shot results')
    parser.add_argument('--fine_tuned_dir', type=str, required=True,
                        help='Directory containing fine-tuned results')
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

def compare_emotion_accuracy(zero_shot_metrics, fine_tuned_metrics, output_dir):
    """Compare emotion accuracy between zero-shot and fine-tuned approaches."""
    if 'emotion_metrics' not in zero_shot_metrics or 'emotion_metrics' not in fine_tuned_metrics:
        print("Emotion metrics not found in one or both metrics files")
        return
    
    # Get accuracy
    zero_shot_accuracy = zero_shot_metrics['emotion_metrics']['accuracy']
    fine_tuned_accuracy = fine_tuned_metrics['emotion_metrics']['accuracy']
    
    # Get F1 scores
    zero_shot_f1_macro = zero_shot_metrics['emotion_metrics']['f1_macro']
    fine_tuned_f1_macro = fine_tuned_metrics['emotion_metrics']['f1_macro']
    
    zero_shot_f1_weighted = zero_shot_metrics['emotion_metrics']['f1_weighted']
    fine_tuned_f1_weighted = fine_tuned_metrics['emotion_metrics']['f1_weighted']
    
    # Create DataFrame
    data = {
        'Metric': ['Accuracy', 'F1 (macro)', 'F1 (weighted)'],
        'Zero-shot': [zero_shot_accuracy, zero_shot_f1_macro, zero_shot_f1_weighted],
        'Fine-tuned': [fine_tuned_accuracy, fine_tuned_f1_macro, fine_tuned_f1_weighted]
    }
    
    df = pd.DataFrame(data)
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(df['Metric']))
    width = 0.35
    
    plt.bar(x - width/2, df['Zero-shot'], width, label='Zero-shot')
    plt.bar(x + width/2, df['Fine-tuned'], width, label='Fine-tuned')
    
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
        'Accuracy': (fine_tuned_accuracy - zero_shot_accuracy) / zero_shot_accuracy * 100,
        'F1 (macro)': (fine_tuned_f1_macro - zero_shot_f1_macro) / zero_shot_f1_macro * 100,
        'F1 (weighted)': (fine_tuned_f1_weighted - zero_shot_f1_weighted) / zero_shot_f1_weighted * 100
    }
    
    print("\nImprovement with fine-tuned approach:")
    for metric, value in improvement.items():
        print(f"{metric}: {value:.2f}%")

def compare_vad_metrics(zero_shot_metrics, fine_tuned_metrics, output_dir):
    """Compare VAD metrics between zero-shot and fine-tuned approaches."""
    if 'vad_metrics' not in zero_shot_metrics or 'vad_metrics' not in fine_tuned_metrics:
        print("VAD metrics not found in one or both metrics files")
        return
    
    # Get MSE
    zero_shot_mse = zero_shot_metrics['vad_metrics']['mse']
    fine_tuned_mse = fine_tuned_metrics['vad_metrics']['mse']
    
    # Get RMSE
    zero_shot_rmse = zero_shot_metrics['vad_metrics']['rmse']
    fine_tuned_rmse = fine_tuned_metrics['vad_metrics']['rmse']
    
    # Get MAE
    zero_shot_mae = zero_shot_metrics['vad_metrics']['mae']
    fine_tuned_mae = fine_tuned_metrics['vad_metrics']['mae']
    
    # Create DataFrame
    data = {
        'Metric': ['MSE', 'RMSE', 'MAE'],
        'Zero-shot': [zero_shot_mse, zero_shot_rmse, zero_shot_mae],
        'Fine-tuned': [fine_tuned_mse, fine_tuned_rmse, fine_tuned_mae]
    }
    
    df = pd.DataFrame(data)
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(df['Metric']))
    width = 0.35
    
    plt.bar(x - width/2, df['Zero-shot'], width, label='Zero-shot')
    plt.bar(x + width/2, df['Fine-tuned'], width, label='Fine-tuned')
    
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
        'MSE': (zero_shot_mse - fine_tuned_mse) / zero_shot_mse * 100,
        'RMSE': (zero_shot_rmse - fine_tuned_rmse) / zero_shot_rmse * 100,
        'MAE': (zero_shot_mae - fine_tuned_mae) / zero_shot_mae * 100
    }
    
    print("\nError reduction with fine-tuned approach:")
    for metric, value in improvement.items():
        print(f"{metric}: {value:.2f}%")

def compare_emotion_class_performance(zero_shot_metrics, fine_tuned_metrics, output_dir):
    """Compare emotion class performance between zero-shot and fine-tuned approaches."""
    if ('emotion_metrics' not in zero_shot_metrics or 
        'classification_report' not in zero_shot_metrics['emotion_metrics'] or
        'emotion_metrics' not in fine_tuned_metrics or
        'classification_report' not in fine_tuned_metrics['emotion_metrics']):
        print("Classification report not found in one or both metrics files")
        return
    
    # Get classification reports
    zero_shot_report = zero_shot_metrics['emotion_metrics']['classification_report']
    fine_tuned_report = fine_tuned_metrics['emotion_metrics']['classification_report']
    
    # Create DataFrame
    data = []
    
    # Get common classes
    common_classes = set()
    for class_name in zero_shot_report.keys():
        if (isinstance(zero_shot_report[class_name], dict) and 
            'f1-score' in zero_shot_report[class_name] and
            class_name in fine_tuned_report and
            isinstance(fine_tuned_report[class_name], dict) and
            'f1-score' in fine_tuned_report[class_name] and
            'support' in zero_shot_report[class_name] and
            zero_shot_report[class_name]['support'] > 10):
            common_classes.add(class_name)
    
    # Add data for common classes
    for class_name in sorted(common_classes):
        data.append({
            'Emotion': class_name,
            'Zero-shot F1': zero_shot_report[class_name]['f1-score'],
            'Fine-tuned F1': fine_tuned_report[class_name]['f1-score'],
            'Support': zero_shot_report[class_name]['support']
        })
    
    df = pd.DataFrame(data)
    
    # Sort by support
    df = df.sort_values('Support', ascending=False)
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(df))
    width = 0.35
    
    plt.bar(x - width/2, df['Zero-shot F1'], width, label='Zero-shot')
    plt.bar(x + width/2, df['Fine-tuned F1'], width, label='Fine-tuned')
    
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
    df['Improvement'] = (df['Fine-tuned F1'] - df['Zero-shot F1']) / df['Zero-shot F1'] * 100
    
    print("\nF1-score improvement by emotion class:")
    for _, row in df.iterrows():
        print(f"{row['Emotion']}: {row['Improvement']:.2f}%")

def create_comparison_summary(zero_shot_metrics, fine_tuned_metrics, output_dir):
    """Create a summary of the comparison."""
    if 'emotion_metrics' not in zero_shot_metrics or 'emotion_metrics' not in fine_tuned_metrics:
        print("Emotion metrics not found in one or both metrics files")
        return
    
    # Get accuracy
    zero_shot_accuracy = zero_shot_metrics['emotion_metrics']['accuracy']
    fine_tuned_accuracy = fine_tuned_metrics['emotion_metrics']['accuracy']
    
    # Get F1 scores
    zero_shot_f1_macro = zero_shot_metrics['emotion_metrics']['f1_macro']
    fine_tuned_f1_macro = fine_tuned_metrics['emotion_metrics']['f1_macro']
    
    zero_shot_f1_weighted = zero_shot_metrics['emotion_metrics']['f1_weighted']
    fine_tuned_f1_weighted = fine_tuned_metrics['emotion_metrics']['f1_weighted']
    
    # Calculate improvement
    accuracy_improvement = (fine_tuned_accuracy - zero_shot_accuracy) / zero_shot_accuracy * 100
    f1_macro_improvement = (fine_tuned_f1_macro - zero_shot_f1_macro) / zero_shot_f1_macro * 100
    f1_weighted_improvement = (fine_tuned_f1_weighted - zero_shot_f1_weighted) / zero_shot_f1_weighted * 100
    
    # Get VAD metrics
    if 'vad_metrics' in zero_shot_metrics and 'vad_metrics' in fine_tuned_metrics:
        zero_shot_mse = zero_shot_metrics['vad_metrics']['mse']
        fine_tuned_mse = fine_tuned_metrics['vad_metrics']['mse']
        
        zero_shot_rmse = zero_shot_metrics['vad_metrics']['rmse']
        fine_tuned_rmse = fine_tuned_metrics['vad_metrics']['rmse']
        
        zero_shot_mae = zero_shot_metrics['vad_metrics']['mae']
        fine_tuned_mae = fine_tuned_metrics['vad_metrics']['mae']
        
        # Calculate improvement
        mse_improvement = (zero_shot_mse - fine_tuned_mse) / zero_shot_mse * 100
        rmse_improvement = (zero_shot_rmse - fine_tuned_rmse) / zero_shot_rmse * 100
        mae_improvement = (zero_shot_mae - fine_tuned_mae) / zero_shot_mae * 100
    else:
        zero_shot_mse = zero_shot_rmse = zero_shot_mae = fine_tuned_mse = fine_tuned_rmse = fine_tuned_mae = 0
        mse_improvement = rmse_improvement = mae_improvement = 0
    
    # Create summary file
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, 'comparison_summary.md')
    
    with open(summary_path, 'w') as f:
        f.write("# Comparison of Zero-shot and Fine-tuned Approaches\n\n")
        
        f.write("## Emotion Classification Performance\n\n")
        f.write("| Metric | Zero-shot | Fine-tuned | Improvement |\n")
        f.write("|--------|-----------|------------|-------------|\n")
        f.write(f"| Accuracy | {zero_shot_accuracy:.4f} | {fine_tuned_accuracy:.4f} | {accuracy_improvement:.2f}% |\n")
        f.write(f"| F1 (macro) | {zero_shot_f1_macro:.4f} | {fine_tuned_f1_macro:.4f} | {f1_macro_improvement:.2f}% |\n")
        f.write(f"| F1 (weighted) | {zero_shot_f1_weighted:.4f} | {fine_tuned_f1_weighted:.4f} | {f1_weighted_improvement:.2f}% |\n\n")
        
        f.write("## VAD Prediction Performance\n\n")
        f.write("| Metric | Zero-shot | Fine-tuned | Improvement |\n")
        f.write("|--------|-----------|------------|-------------|\n")
        f.write(f"| MSE | {zero_shot_mse:.4f} | {fine_tuned_mse:.4f} | {mse_improvement:.2f}% |\n")
        f.write(f"| RMSE | {zero_shot_rmse:.4f} | {fine_tuned_rmse:.4f} | {rmse_improvement:.2f}% |\n")
        f.write(f"| MAE | {zero_shot_mae:.4f} | {fine_tuned_mae:.4f} | {mae_improvement:.2f}% |\n\n")
        
        f.write("## Emotion-specific Performance\n\n")
        f.write("See the emotion class comparison plot for detailed performance by emotion class.\n\n")
        
        f.write("## Visualizations\n\n")
        f.write("- `emotion_accuracy_comparison.png`: Comparison of overall emotion classification performance\n")
        f.write("- `vad_metrics_comparison.png`: Comparison of VAD prediction performance\n")
        f.write("- `emotion_class_comparison.png`: Comparison of F1-scores by emotion class\n\n")
        
        f.write("## Analysis\n\n")
        
        if accuracy_improvement > 0:
            f.write("The fine-tuned approach shows improvement over the zero-shot approach in terms of emotion classification accuracy. ")
        else:
            f.write("The fine-tuned approach does not show improvement over the zero-shot approach in terms of emotion classification accuracy. ")
        
        if mse_improvement > 0:
            f.write("It also achieves lower error in VAD prediction. ")
        else:
            f.write("However, it does not achieve lower error in VAD prediction. ")
        
        f.write("The performance improvement varies across different emotion classes, with some emotions benefiting more from fine-tuning than others.\n\n")
        
        f.write("## Conclusion\n\n")
        
        if accuracy_improvement > 0 and mse_improvement > 0:
            f.write("The fine-tuned approach outperforms the zero-shot approach in both emotion classification and VAD prediction, demonstrating the value of fine-tuning pre-trained language models for emotion recognition tasks.")
        elif accuracy_improvement > 0:
            f.write("The fine-tuned approach outperforms the zero-shot approach in emotion classification, but not in VAD prediction. This suggests that fine-tuning improves the model's ability to discriminate between emotions, but not necessarily its ability to predict VAD values accurately.")
        elif mse_improvement > 0:
            f.write("The fine-tuned approach outperforms the zero-shot approach in VAD prediction, but not in emotion classification. This suggests that fine-tuning improves the model's ability to predict VAD values accurately, but this does not translate to better emotion classification.")
        else:
            f.write("The fine-tuned approach does not outperform the zero-shot approach in either emotion classification or VAD prediction. This suggests that the zero-shot approach is already effective for this task, or that the fine-tuning process needs to be improved.")
    
    print(f"Comparison summary saved to {summary_path}")

def main():
    """Main function."""
    args = parse_args()
    
    # Load metrics
    zero_shot_metrics = load_metrics(args.zero_shot_dir, prefix='pipeline_')
    fine_tuned_metrics = load_metrics(args.fine_tuned_dir, prefix='pipeline_')
    
    if zero_shot_metrics is None or fine_tuned_metrics is None:
        print("Could not load metrics from one or both directories")
        return
    
    # Compare emotion accuracy
    compare_emotion_accuracy(zero_shot_metrics, fine_tuned_metrics, args.output_dir)
    
    # Compare VAD metrics
    compare_vad_metrics(zero_shot_metrics, fine_tuned_metrics, args.output_dir)
    
    # Compare emotion class performance
    compare_emotion_class_performance(zero_shot_metrics, fine_tuned_metrics, args.output_dir)
    
    # Create comparison summary
    create_comparison_summary(zero_shot_metrics, fine_tuned_metrics, args.output_dir)
    
    print(f"All comparisons saved to {args.output_dir}")

if __name__ == '__main__':
    main()
