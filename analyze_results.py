#!/usr/bin/env python3
"""
Script to analyze the downloaded experiment results
Generates summary statistics and comparisons of experiment performance
"""

import json
import os
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from collections import defaultdict

def load_experiment_data(results_dir="./emod_results"):
    """Load experiment summary data from JSON file"""
    summary_path = Path(results_dir) / "all_experiments_summary.json"
    
    if not summary_path.exists():
        print(f"Error: Could not find summary file at {summary_path}")
        return None
    
    try:
        with open(summary_path, 'r') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} experiment summaries")
        return data
    except Exception as e:
        print(f"Error loading experiment data: {e}")
        return None

def analyze_experiments(data):
    """Analyze experiment results and return summary statistics"""
    if not data:
        return None
    
    # Convert to pandas DataFrame for easier analysis
    df = pd.DataFrame(data)
    
    # Filter out experiments without final results
    successful_df = df[df['has_final_results'] == True].copy()
    
    print(f"Total experiments: {len(df)}")
    print(f"Experiments with final results: {len(successful_df)}")
    
    # Add additional columns for analysis
    successful_df['has_all_files'] = (
        successful_df['has_final_results'] & 
        successful_df['has_training_log'] & 
        successful_df['has_model']
    )
    
    # Count experiments by dataset
    dataset_counts = successful_df['dataset'].value_counts()
    print("\nExperiments by dataset:")
    for dataset, count in dataset_counts.items():
        print(f"  {dataset}: {count}")
    
    # Count experiments by type
    type_counts = successful_df['experiment_type'].value_counts()
    print("\nExperiments by type:")
    for exp_type, count in type_counts.items():
        print(f"  {exp_type}: {count}")
    
    # Count by model
    model_counts = successful_df['model_name'].value_counts()
    print(f"\nTotal models used: {len(model_counts)}")
    
    # Analyze multimodal experiments
    if 'multimodal' in successful_df['experiment_type'].values:
        multimodal_df = successful_df[successful_df['experiment_type'] == 'multimodal']
        
        # Count by audio feature
        if 'audio_feature' in multimodal_df.columns:
            audio_counts = multimodal_df['audio_feature'].value_counts()
            print("\nAudio features used:")
            for feature, count in audio_counts.items():
                print(f"  {feature}: {count}")
        
        # Count by fusion type
        if 'fusion_type' in multimodal_df.columns:
            fusion_counts = multimodal_df['fusion_type'].value_counts()
            print("\nFusion methods used:")
            for fusion, count in fusion_counts.items():
                print(f"  {fusion}: {count}")
    
    return successful_df

def compare_models(df, dataset=None):
    """Compare model performance across experiments"""
    if dataset:
        filtered_df = df[df['dataset'] == dataset].copy()
        print(f"\nAnalyzing model performance for dataset: {dataset}")
    else:
        filtered_df = df.copy()
        print("\nAnalyzing model performance across all datasets")
    
    if len(filtered_df) == 0:
        print("No matching experiments found.")
        return
    
    # Group by model name and get max validation accuracy
    model_perf = filtered_df.groupby('model_name')['best_val_accuracy'].agg(['mean', 'max', 'count'])
    model_perf = model_perf.sort_values('max', ascending=False)
    
    print("\nModel performance (sorted by best accuracy):")
    print(tabulate(model_perf, headers=['Mean Accuracy', 'Best Accuracy', 'Experiments'], tablefmt='pretty'))
    
    # For multimodal experiments, analyze by audio feature and fusion type
    if 'multimodal' in filtered_df['experiment_type'].values:
        multimodal_df = filtered_df[filtered_df['experiment_type'] == 'multimodal']
        
        if len(multimodal_df) > 0 and 'audio_feature' in multimodal_df.columns and 'fusion_type' in multimodal_df.columns:
            # Group by audio feature
            audio_perf = multimodal_df.groupby('audio_feature')['best_val_accuracy'].agg(['mean', 'max', 'count'])
            audio_perf = audio_perf.sort_values('max', ascending=False)
            
            print("\nPerformance by audio feature:")
            print(tabulate(audio_perf, headers=['Mean Accuracy', 'Best Accuracy', 'Experiments'], tablefmt='pretty'))
            
            # Group by fusion type
            fusion_perf = multimodal_df.groupby('fusion_type')['best_val_accuracy'].agg(['mean', 'max', 'count'])
            fusion_perf = fusion_perf.sort_values('max', ascending=False)
            
            print("\nPerformance by fusion method:")
            print(tabulate(fusion_perf, headers=['Mean Accuracy', 'Best Accuracy', 'Experiments'], tablefmt='pretty'))
            
            # Find best combination
            combo_perf = multimodal_df.groupby(['audio_feature', 'fusion_type'])['best_val_accuracy'].agg(['mean', 'max', 'count'])
            combo_perf = combo_perf.sort_values('max', ascending=False)
            
            print("\nBest audio feature + fusion combinations:")
            print(tabulate(combo_perf.head(5), headers=['Mean Accuracy', 'Best Accuracy', 'Experiments'], tablefmt='pretty'))

def identify_best_experiments(df, top_n=5):
    """Identify the best performing experiments"""
    if df is None or len(df) == 0:
        print("No experiment data available")
        return
    
    # Sort by validation accuracy
    best_exps = df.sort_values('best_val_accuracy', ascending=False).head(top_n)
    
    print(f"\nTop {top_n} experiments by validation accuracy:")
    for i, (_, exp) in enumerate(best_exps.iterrows()):
        print(f"{i+1}. {exp['experiment_id']}:")
        print(f"   - Model: {exp['model_name']}")
        print(f"   - Dataset: {exp['dataset']}")
        print(f"   - Type: {exp['experiment_type']}")
        print(f"   - Validation Accuracy: {exp['best_val_accuracy']:.4f}")
        
        # Print multimodal details if available
        if exp['experiment_type'] == 'multimodal' and 'audio_feature' in exp and 'fusion_type' in exp:
            print(f"   - Audio Feature: {exp['audio_feature']}")
            print(f"   - Fusion Method: {exp['fusion_type']}")
        
        print()
    
    # Group by dataset and find best for each
    datasets = df['dataset'].unique()
    print("\nBest experiment for each dataset:")
    
    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset]
        best_exp = dataset_df.loc[dataset_df['best_val_accuracy'].idxmax()]
        
        print(f"Dataset: {dataset}")
        print(f"   - Experiment: {best_exp['experiment_id']}")
        print(f"   - Model: {best_exp['model_name']}")
        print(f"   - Validation Accuracy: {best_exp['best_val_accuracy']:.4f}")
        
        if best_exp['experiment_type'] == 'multimodal' and 'audio_feature' in best_exp and 'fusion_type' in best_exp:
            print(f"   - Audio Feature: {best_exp['audio_feature']}")
            print(f"   - Fusion Method: {best_exp['fusion_type']}")
        
        print()

def plot_model_comparison(df, output_dir="./plots"):
    """Generate plots comparing model performance"""
    if df is None or len(df) == 0:
        print("No experiment data available for plotting")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Compare text models
    text_df = df[df['experiment_type'] == 'text'].copy()
    
    if len(text_df) > 0:
        plt.figure(figsize=(12, 6))
        model_perf = text_df.groupby('model_name')['best_val_accuracy'].max()
        model_perf = model_perf.sort_values(ascending=False)
        
        bars = plt.bar(model_perf.index, model_perf.values)
        plt.title('Best Validation Accuracy by Text Model')
        plt.ylabel('Validation Accuracy')
        plt.xlabel('Model')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.0)
        plt.tight_layout()
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.savefig(os.path.join(output_dir, "text_model_comparison.png"), dpi=300)
        print(f"Saved text model comparison plot to {output_dir}/text_model_comparison.png")
    
    # Compare multimodal configurations if available
    multimodal_df = df[df['experiment_type'] == 'multimodal'].copy()
    
    if len(multimodal_df) > 0 and 'audio_feature' in multimodal_df.columns and 'fusion_type' in multimodal_df.columns:
        # Compare audio features
        plt.figure(figsize=(10, 6))
        audio_perf = multimodal_df.groupby('audio_feature')['best_val_accuracy'].max()
        audio_perf = audio_perf.sort_values(ascending=False)
        
        bars = plt.bar(audio_perf.index, audio_perf.values)
        plt.title('Best Validation Accuracy by Audio Feature')
        plt.ylabel('Validation Accuracy')
        plt.xlabel('Audio Feature')
        plt.ylim(0, 1.0)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.savefig(os.path.join(output_dir, "audio_feature_comparison.png"), dpi=300)
        print(f"Saved audio feature comparison plot to {output_dir}/audio_feature_comparison.png")
        
        # Compare fusion methods
        plt.figure(figsize=(10, 6))
        fusion_perf = multimodal_df.groupby('fusion_type')['best_val_accuracy'].max()
        fusion_perf = fusion_perf.sort_values(ascending=False)
        
        bars = plt.bar(fusion_perf.index, fusion_perf.values)
        plt.title('Best Validation Accuracy by Fusion Method')
        plt.ylabel('Validation Accuracy')
        plt.xlabel('Fusion Method')
        plt.ylim(0, 1.0)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.savefig(os.path.join(output_dir, "fusion_method_comparison.png"), dpi=300)
        print(f"Saved fusion method comparison plot to {output_dir}/fusion_method_comparison.png")
        
        # Compare datasets
        if len(df['dataset'].unique()) > 1:
            plt.figure(figsize=(10, 6))
            dataset_perf = df.groupby('dataset')['best_val_accuracy'].max()
            
            bars = plt.bar(dataset_perf.index, dataset_perf.values)
            plt.title('Best Validation Accuracy by Dataset')
            plt.ylabel('Validation Accuracy')
            plt.xlabel('Dataset')
            plt.ylim(0, 1.0)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.4f}', ha='center', va='bottom')
            
            plt.savefig(os.path.join(output_dir, "dataset_comparison.png"), dpi=300)
            print(f"Saved dataset comparison plot to {output_dir}/dataset_comparison.png")

def main():
    parser = argparse.ArgumentParser(description="Analyze EMOD experiment results")
    parser.add_argument('--dir', type=str, default='./emod_results',
                        help='Directory containing experiment results')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Filter analysis to specific dataset')
    parser.add_argument('--top', type=int, default=5,
                        help='Number of top experiments to show')
    parser.add_argument('--plots', action='store_true',
                        help='Generate comparison plots')
    parser.add_argument('--plot-dir', type=str, default='./plots',
                        help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Load experiment data
    data = load_experiment_data(args.dir)
    
    if data:
        # Analyze experiments
        df = analyze_experiments(data)
        
        if df is not None:
            # Compare models
            compare_models(df, args.dataset)
            
            # Identify best experiments
            identify_best_experiments(df, args.top)
            
            # Generate plots if requested
            if args.plots:
                plot_model_comparison(df, args.plot_dir)

if __name__ == "__main__":
    main() 