#!/usr/bin/env python3
"""
Report Generator Module for EMOD

This module handles generating comprehensive reports from experiment results.
It supports both HTML and Markdown output formats.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# Default paths
RESULTS_DIR = "./results"
REPORTS_DIR = "./reports"

# Default markdown template
DEFAULT_MARKDOWN_TEMPLATE = """
# EMOD Experiment Results Report

Generated on {timestamp}

## Summary

- Total experiments analyzed: {experiment_count}
- Text-only models: {text_model_count}
- Multimodal models: {multimodal_count}

## Performance Overview

### Stage 1: VAD Prediction Performance

The following table shows the Mean Squared Error (MSE) for Valence, Arousal, and Dominance prediction across all models:

{all_vad_performance_table}

### Stage 2: Emotion Classification Performance

The following table shows the classification performance metrics across all models:

{all_classifier_performance_table}

## Performance Comparisons

### Top Performing Models

Ranked by lowest Mean Squared Error (MSE) on the Valence dimension:

{top_vad_models_table}

Ranked by highest Weighted F1 Score:

{top_classifier_models_table}

## Experiment Details

### Text-Only Models

{text_models_table}

### Multimodal Models

{multimodal_models_table}

## Visualizations

### VAD Prediction Performance Comparison
![VAD Prediction MSE](vad_performance_chart.png)

### Emotion Classification F1 Scores
![Classification F1 Scores](classification_f1_chart.png)

### Text vs Multimodal Performance
![Text vs Multimodal](text_vs_multimodal_chart.png)

### Training Curves

Training curves for all experiments are available in the `training_curves` directory.

## Detailed Results

Detailed results for each experiment are available in their respective directories within the `results` directory.

## Methodology

Experiments were conducted using the EMOD two-stage emotion recognition system:

1. **Stage 1**: Convert input (text and/or audio) to Valence-Arousal-Dominance (VAD) tuples
2. **Stage 2**: Classify emotions into four categories (happy, angry, sad, neutral)

For multimodal experiments, different fusion strategies were tested to combine text and audio features.

## Model Architectures

### Text Encoders
- RoBERTa
- BERT/DeBERTa
- DistilBERT
- XLNet
- ALBERT

### Audio Features
- MFCC
- Spectrogram
- Prosodic features
- wav2vec embeddings

### Fusion Strategies
- Early fusion
- Late fusion
- Hybrid fusion
- Attention-based fusion

### ML Classifiers
- Random Forest
- Gradient Boosting
- Support Vector Machines (SVM)
- Logistic Regression
- Multi-layer Perceptron (MLP)
"""

# Default HTML template (simplified header and footer shown here)
DEFAULT_HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>EMOD Experiment Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1, h2, h3 { color: #333; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .section { margin-bottom: 30px; }
        .viz-container { margin: 20px 0; text-align: center; }
        .viz-container img { max-width: 100%; height: auto; }
    </style>
</head>
<body>
    <h1>EMOD Experiment Results Report</h1>
    <p>Generated on {timestamp}</p>
    
    <div class="section">
        <h2>Summary</h2>
        <p>Total experiments analyzed: {experiment_count}</p>
        <p>Text-only models: {text_model_count}</p>
        <p>Multimodal models: {multimodal_count}</p>
    </div>
    
    <div class="section">
        <h2>Performance Overview</h2>
        
        <h3>Stage 1: VAD Prediction Performance</h3>
        {all_vad_performance_html}
        
        <h3>Stage 2: Emotion Classification Performance</h3>
        {all_classifier_performance_html}
    </div>
    
    <div class="section">
        <h2>Performance Comparisons</h2>
        
        <h3>Top Performing Models - VAD Prediction (by Valence MSE)</h3>
        {top_vad_models_html}
        
        <h3>Top Performing Models - Emotion Classification (by F1 Score)</h3>
        {top_classifier_models_html}
    </div>
    
    <div class="section">
        <h2>Experiment Details</h2>
        
        <h3>Text-Only Models</h3>
        {text_models_html}
        
        <h3>Multimodal Models</h3>
        {multimodal_models_html}
    </div>
    
    <div class="section">
        <h2>Visualizations</h2>
        
        <div class="viz-container">
            <h3>VAD Prediction Performance Comparison</h3>
            <img src="vad_performance_chart.png" alt="VAD Prediction MSE">
        </div>
        
        <div class="viz-container">
            <h3>Emotion Classification F1 Scores</h3>
            <img src="classification_f1_chart.png" alt="Classification F1 Scores">
        </div>
        
        <div class="viz-container">
            <h3>Text vs Multimodal Performance</h3>
            <img src="text_vs_multimodal_chart.png" alt="Text vs Multimodal">
        </div>
        
        <p>Training curves are available in the <a href="training_curves/">training_curves</a> directory.</p>
    </div>
</body>
</html>
"""

def collect_experiment_data(results_dir: str = RESULTS_DIR) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Collect experiment data from the results directory
    
    Returns:
        Tuple of (experiment data, text model count, multimodal count)
    """
    # Try to load experiment summary first
    summary_path = os.path.join(results_dir, "experiment_summary.csv")
    
    if os.path.exists(summary_path):
        try:
            df = pd.read_csv(summary_path)
            
            # Convert DataFrame to list of dictionaries
            experiment_data = df.to_dict(orient='records')
            
            # Count types
            text_count = sum(1 for exp in experiment_data if exp.get('Type') == 'Text-only')
            multimodal_count = sum(1 for exp in experiment_data if exp.get('Type') == 'Multimodal')
            
            return experiment_data, text_count, multimodal_count
        except Exception as e:
            print(f"Error loading experiment summary: {e}")
            print("Falling back to loading individual experiment results...")
    
    # Load comparison tables next
    text_table_path = os.path.join(REPORTS_DIR, "text_model_comparison.csv")
    multimodal_table_path = os.path.join(REPORTS_DIR, "multimodal_comparison.csv")
    
    experiment_data = []
    
    # Load text model data
    if os.path.exists(text_table_path):
        try:
            text_df = pd.read_csv(text_table_path)
            text_records = text_df.to_dict(orient='records')
            for record in text_records:
                record['Type'] = 'Text-only'
            experiment_data.extend(text_records)
        except Exception as e:
            print(f"Error loading text model comparison data: {e}")
    
    # Load multimodal data
    if os.path.exists(multimodal_table_path):
        try:
            multimodal_df = pd.read_csv(multimodal_table_path)
            multimodal_records = multimodal_df.to_dict(orient='records')
            for record in multimodal_records:
                record['Type'] = 'Multimodal'
            experiment_data.extend(multimodal_records)
        except Exception as e:
            print(f"Error loading multimodal comparison data: {e}")
    
    # Count types
    text_count = sum(1 for exp in experiment_data if exp.get('Type') == 'Text-only')
    multimodal_count = sum(1 for exp in experiment_data if exp.get('Type') == 'Multimodal')
    
    return experiment_data, text_count, multimodal_count

def format_dataframe_as_markdown(df: pd.DataFrame) -> str:
    """Format a DataFrame as a markdown table"""
    if df.empty:
        return "No data available"
    
    # Convert DataFrame to markdown
    markdown_table = df.to_markdown(index=False)
    return markdown_table

def format_dataframe_as_html(df: pd.DataFrame) -> str:
    """Format a DataFrame as an HTML table"""
    if df.empty:
        return "<p>No data available</p>"
    
    # Convert DataFrame to HTML
    html_table = df.to_html(index=False)
    return html_table

def select_top_vad_models(experiment_data: List[Dict[str, Any]], top_n: int = 5) -> pd.DataFrame:
    """Select top models based on Valence MSE"""
    # Filter experiments with Valence MSE metric
    filtered_data = [d for d in experiment_data if 'valence_mse' in d and d['valence_mse'] is not None]
    
    # Sort by Valence MSE (lower is better)
    sorted_data = sorted(filtered_data, key=lambda x: x['valence_mse'])
    top_data = sorted_data[:top_n]
    
    # Create DataFrame with selected columns
    columns = ['text_model', 'Type']
    
    # Add multimodal columns if present
    if any(d.get('Type') == 'Multimodal' or d.get('experiment_type') == 'multimodal' for d in top_data):
        if 'audio_feature' in top_data[0]:
            columns.extend(['audio_feature', 'fusion_type'])
        else:
            columns.extend(['Audio', 'Fusion'])
    
    # Add metric columns
    metric_columns = ['valence_mse', 'arousal_mse', 'dominance_mse']
    available_metrics = [col for col in metric_columns if any(col in d for d in top_data)]
    columns.extend(available_metrics)
    
    # Create DataFrame with available columns
    result_data = []
    for d in top_data:
        # Normalize column names from different sources
        normalized = {}
        for col in columns:
            if col in d:
                normalized[col] = d[col]
            elif col == 'Type' and 'experiment_type' in d:
                normalized[col] = 'Multimodal' if d['experiment_type'] == 'multimodal' else 'Text-only'
            elif col == 'Audio' and 'audio_feature' in d:
                normalized[col] = d['audio_feature']
            elif col == 'Fusion' and 'fusion_type' in d:
                normalized[col] = d['fusion_type']
        
        result_data.append(normalized)
    
    df = pd.DataFrame(result_data)
    
    # Rename columns for better readability
    column_renames = {
        'text_model': 'Text Model',
        'audio_feature': 'Audio Feature',
        'fusion_type': 'Fusion Type',
        'valence_mse': 'Valence MSE',
        'arousal_mse': 'Arousal MSE',
        'dominance_mse': 'Dominance MSE'
    }
    df = df.rename(columns={col: column_renames.get(col, col) for col in df.columns})
    
    return df

def select_top_classifier_models(experiment_data: List[Dict[str, Any]], top_n: int = 5) -> pd.DataFrame:
    """Select top models based on classifier F1 score"""
    # Filter experiments with classifier F1 metric
    f1_key = 'best_classifier_f1' if 'best_classifier_f1' in experiment_data[0] else 'Best F1'
    filtered_data = [d for d in experiment_data if f1_key in d and d[f1_key] is not None]
    
    # Sort by F1 score (higher is better)
    sorted_data = sorted(filtered_data, key=lambda x: x[f1_key], reverse=True)
    top_data = sorted_data[:top_n]
    
    # Create DataFrame with selected columns
    columns = ['text_model', 'Type']
    
    # Add multimodal columns if present
    if any(d.get('Type') == 'Multimodal' or d.get('experiment_type') == 'multimodal' for d in top_data):
        if 'audio_feature' in top_data[0]:
            columns.extend(['audio_feature', 'fusion_type'])
        else:
            columns.extend(['Audio', 'Fusion'])
    
    # Add classifier columns
    classifier_keys = {
        'best_classifier': 'Best Classifier',
        'Best Classifier': 'Best Classifier',
        'best_classifier_accuracy': 'Accuracy',
        'Best Accuracy': 'Accuracy',
        'best_classifier_f1': 'F1 Score',
        'Best F1': 'F1 Score'
    }
    
    for key, display_name in classifier_keys.items():
        if any(key in d for d in top_data):
            columns.append(key)
    
    # Create DataFrame with available columns
    result_data = []
    for d in top_data:
        # Normalize column names from different sources
        normalized = {}
        for col in columns:
            if col in d:
                normalized[col] = d[col]
            elif col == 'Type' and 'experiment_type' in d:
                normalized[col] = 'Multimodal' if d['experiment_type'] == 'multimodal' else 'Text-only'
            elif col == 'Audio' and 'audio_feature' in d:
                normalized[col] = d['audio_feature']
            elif col == 'Fusion' and 'fusion_type' in d:
                normalized[col] = d['fusion_type']
        
        result_data.append(normalized)
    
    df = pd.DataFrame(result_data)
    
    # Rename columns for better readability
    column_renames = {
        'text_model': 'Text Model',
        'audio_feature': 'Audio Feature',
        'fusion_type': 'Fusion Type',
        'best_classifier': 'Best Classifier',
        'best_classifier_accuracy': 'Accuracy',
        'best_classifier_f1': 'F1 Score'
    }
    df = df.rename(columns={col: column_renames.get(col, col) for col in df.columns})
    
    return df

def format_model_tables(experiment_data: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Format experiment data into separate tables for text and multimodal models"""
    # Split by type
    text_data = [d for d in experiment_data if d.get('Type') == 'Text-only' or d.get('experiment_type') == 'text']
    multimodal_data = [d for d in experiment_data if d.get('Type') == 'Multimodal' or d.get('experiment_type') == 'multimodal']
    
    # Create DataFrames
    text_df = pd.DataFrame(text_data) if text_data else pd.DataFrame()
    multimodal_df = pd.DataFrame(multimodal_data) if multimodal_data else pd.DataFrame()
    
    # Select and rename columns for text models
    if not text_df.empty:
        # Select important columns
        text_columns = ['text_model', 'valence_mse', 'arousal_mse', 'dominance_mse', 
                        'best_classifier', 'best_classifier_accuracy', 'best_classifier_f1']
        text_columns = [col for col in text_columns if col in text_df.columns]
        
        # Alternative column names
        for alt_col, col in [('Text Model', 'text_model'), ('Model', 'text_model'),
                            ('Best Classifier', 'best_classifier'),
                            ('Best Accuracy', 'best_classifier_accuracy'),
                            ('Best F1', 'best_classifier_f1')]:
            if alt_col in text_df.columns and col not in text_columns:
                text_columns.append(alt_col)
        
        text_df = text_df[text_columns]
        
        # Rename columns
        column_renames = {
            'text_model': 'Text Model',
            'valence_mse': 'Valence MSE',
            'arousal_mse': 'Arousal MSE',
            'dominance_mse': 'Dominance MSE',
            'best_classifier': 'Best Classifier',
            'best_classifier_accuracy': 'Accuracy',
            'best_classifier_f1': 'F1 Score'
        }
        text_df = text_df.rename(columns={col: column_renames.get(col, col) for col in text_df.columns})
    
    # Select and rename columns for multimodal models
    if not multimodal_df.empty:
        # Select important columns
        multimodal_columns = ['text_model', 'audio_feature', 'fusion_type', 
                             'valence_mse', 'arousal_mse', 'dominance_mse',
                             'best_classifier', 'best_classifier_accuracy', 'best_classifier_f1']
        multimodal_columns = [col for col in multimodal_columns if col in multimodal_df.columns]
        
        # Alternative column names
        for alt_col, col in [('Text Model', 'text_model'), ('Model', 'text_model'),
                            ('Audio', 'audio_feature'), ('Audio Feature', 'audio_feature'),
                            ('Fusion', 'fusion_type'), ('Fusion Type', 'fusion_type'),
                            ('Best Classifier', 'best_classifier'),
                            ('Best Accuracy', 'best_classifier_accuracy'),
                            ('Best F1', 'best_classifier_f1')]:
            if alt_col in multimodal_df.columns and col not in multimodal_columns:
                multimodal_columns.append(alt_col)
        
        multimodal_df = multimodal_df[multimodal_columns]
        
        # Rename columns
        column_renames = {
            'text_model': 'Text Model',
            'audio_feature': 'Audio Feature',
            'fusion_type': 'Fusion Type',
            'valence_mse': 'Valence MSE',
            'arousal_mse': 'Arousal MSE',
            'dominance_mse': 'Dominance MSE',
            'best_classifier': 'Best Classifier',
            'best_classifier_accuracy': 'Accuracy',
            'best_classifier_f1': 'F1 Score'
        }
        multimodal_df = multimodal_df.rename(columns={col: column_renames.get(col, col) for col in multimodal_df.columns})
    
    return text_df, multimodal_df

def generate_all_vad_performance_table(experiment_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Generate a table with VAD prediction performance for all models"""
    # Select models with VAD metrics
    models_with_vad = []
    
    for exp in experiment_data:
        model_info = {
            'Type': 'Text-only' if exp.get('Type') == 'Text-only' or exp.get('experiment_type') == 'text' else 'Multimodal',
            'Model': exp.get('text_model') or exp.get('Model', '')
        }
        
        # Add multimodal info if available
        if model_info['Type'] == 'Multimodal':
            model_info['Audio'] = exp.get('audio_feature') or exp.get('Audio', '')
            model_info['Fusion'] = exp.get('fusion_type') or exp.get('Fusion', '')
        
        # Add VAD metrics if available
        vad_metrics = False
        for dim in ['valence', 'arousal', 'dominance']:
            mse_key = f'{dim}_mse'
            r2_key = f'{dim}_r2'
            
            if mse_key in exp:
                model_info[f'{dim.title()} MSE'] = exp[mse_key]
                vad_metrics = True
            
            if r2_key in exp:
                model_info[f'{dim.title()} R²'] = exp[r2_key]
        
        # Only add if VAD metrics found
        if vad_metrics:
            models_with_vad.append(model_info)
    
    # Convert to DataFrame
    if not models_with_vad:
        return pd.DataFrame()
    
    df = pd.DataFrame(models_with_vad)
    
    # Ensure consistent columns
    required_columns = ['Type', 'Model']
    metric_columns = ['Valence MSE', 'Arousal MSE', 'Dominance MSE']
    
    for col in required_columns + metric_columns:
        if col not in df.columns:
            df[col] = None
    
    # Return sorted by overall performance (average MSE)
    df['Avg MSE'] = df[metric_columns].mean(axis=1)
    return df.sort_values('Avg MSE').drop('Avg MSE', axis=1)

def generate_all_classifier_performance_table(experiment_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Generate a table with classification performance for all models"""
    # Select models with classifier metrics
    models_with_clf = []
    
    for exp in experiment_data:
        model_info = {
            'Type': 'Text-only' if exp.get('Type') == 'Text-only' or exp.get('experiment_type') == 'text' else 'Multimodal',
            'Model': exp.get('text_model') or exp.get('Model', '')
        }
        
        # Add multimodal info if available
        if model_info['Type'] == 'Multimodal':
            model_info['Audio'] = exp.get('audio_feature') or exp.get('Audio', '')
            model_info['Fusion'] = exp.get('fusion_type') or exp.get('Fusion', '')
        
        # Add best classifier info if available
        clf_metrics = False
        
        # Try different column naming conventions
        best_clf_cols = [
            ('best_classifier', 'best_classifier_accuracy', 'best_classifier_f1'),
            ('Best Classifier', 'Best Accuracy', 'Best F1')
        ]
        
        for clf_col, acc_col, f1_col in best_clf_cols:
            if clf_col in exp and exp[clf_col]:
                model_info['Best Classifier'] = exp[clf_col]
                model_info['Accuracy'] = exp.get(acc_col)
                model_info['F1 Score'] = exp.get(f1_col)
                clf_metrics = True
                break
        
        # Only add if classifier metrics found
        if clf_metrics:
            models_with_clf.append(model_info)
    
    # Convert to DataFrame
    if not models_with_clf:
        return pd.DataFrame()
    
    df = pd.DataFrame(models_with_clf)
    
    # Ensure consistent columns
    required_columns = ['Type', 'Model', 'Best Classifier', 'Accuracy', 'F1 Score']
    
    for col in required_columns:
        if col not in df.columns:
            df[col] = None
    
    # Return sorted by F1 score
    return df.sort_values('F1 Score', ascending=False)

def generate_performance_visualizations(experiment_data: List[Dict[str, Any]], output_dir: str = REPORTS_DIR) -> Dict[str, str]:
    """Generate performance visualization charts"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualization files
    viz_files = {}
    
    # 1. VAD Prediction Performance Chart
    vad_data = generate_all_vad_performance_table(experiment_data)
    if not vad_data.empty:
        plt.figure(figsize=(12, 6))
        
        # Prepare data for plotting
        plot_data = []
        
        for _, row in vad_data.iterrows():
            model_name = row['Model']
            if row['Type'] == 'Multimodal':
                model_name = f"{model_name}-{row.get('Audio', '')}-{row.get('Fusion', '')}"
            
            for dim in ['Valence MSE', 'Arousal MSE', 'Dominance MSE']:
                if pd.notnull(row[dim]):
                    plot_data.append({
                        'Model': model_name,
                        'Dimension': dim.split()[0], 
                        'MSE': row[dim]
                    })
        
        if plot_data:
            plot_df = pd.DataFrame(plot_data)
            
            # Create bar plot with Seaborn
            ax = sns.barplot(x='Model', y='MSE', hue='Dimension', data=plot_df)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.title('VAD Prediction Mean Squared Error by Model')
            
            # Save the figure
            vad_chart_path = os.path.join(output_dir, 'vad_performance_chart.png')
            plt.savefig(vad_chart_path)
            plt.close()
            
            viz_files['vad_chart'] = vad_chart_path
    
    # 2. Emotion Classification F1 Score Chart
    clf_data = generate_all_classifier_performance_table(experiment_data)
    if not clf_data.empty:
        plt.figure(figsize=(12, 6))
        
        # Prepare data for plotting
        plot_data = []
        
        for _, row in clf_data.iterrows():
            model_name = row['Model']
            if row['Type'] == 'Multimodal':
                model_name = f"{model_name}-{row.get('Audio', '')}-{row.get('Fusion', '')}"
            
            if pd.notnull(row['F1 Score']):
                plot_data.append({
                    'Model': model_name,
                    'Classifier': row['Best Classifier'],
                    'F1 Score': row['F1 Score']
                })
        
        if plot_data:
            plot_df = pd.DataFrame(plot_data)
            
            # Create bar plot with Seaborn
            ax = sns.barplot(x='Model', y='F1 Score', hue='Classifier', data=plot_df)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.title('Emotion Classification F1 Scores by Model')
            
            # Save the figure
            f1_chart_path = os.path.join(output_dir, 'classification_f1_chart.png')
            plt.savefig(f1_chart_path)
            plt.close()
            
            viz_files['f1_chart'] = f1_chart_path
    
    # 3. Text vs Multimodal Performance Comparison
    if not clf_data.empty:
        plt.figure(figsize=(10, 6))
        
        # Group by type and calculate mean F1 score
        text_vs_mm = clf_data.groupby('Type')['F1 Score'].agg(['mean', 'std']).reset_index()
        
        if not text_vs_mm.empty:
            # Create bar plot
            ax = sns.barplot(x='Type', y='mean', data=text_vs_mm)
            
            # Add error bars
            if 'std' in text_vs_mm.columns:
                x = np.arange(len(text_vs_mm))
                plt.errorbar(x=x, y=text_vs_mm['mean'], yerr=text_vs_mm['std'], fmt='none', c='black', capsize=5)
            
            plt.ylabel('Average F1 Score')
            plt.title('Text-only vs Multimodal Performance Comparison')
            
            # Save the figure
            compare_chart_path = os.path.join(output_dir, 'text_vs_multimodal_chart.png')
            plt.savefig(compare_chart_path)
            plt.close()
            
            viz_files['compare_chart'] = compare_chart_path
    
    return viz_files

def generate_markdown_report(
    experiment_data: List[Dict[str, Any]], 
    text_count: int, 
    multimodal_count: int,
    template: Optional[str] = None
) -> str:
    """Generate a markdown report"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get template
    template_content = DEFAULT_MARKDOWN_TEMPLATE
    if template and os.path.exists(template):
        with open(template, 'r') as f:
            template_content = f.read()
    
    # Generate tables
    top_vad_df = select_top_vad_models(experiment_data)
    top_clf_df = select_top_classifier_models(experiment_data)
    text_df, multimodal_df = format_model_tables(experiment_data)
    
    # Generate all performance tables
    all_vad_performance_df = generate_all_vad_performance_table(experiment_data)
    all_classifier_performance_df = generate_all_classifier_performance_table(experiment_data)
    
    # Format tables as markdown
    top_vad_table = format_dataframe_as_markdown(top_vad_df)
    top_clf_table = format_dataframe_as_markdown(top_clf_df)
    text_table = format_dataframe_as_markdown(text_df)
    multimodal_table = format_dataframe_as_markdown(multimodal_df)
    all_vad_performance_table = format_dataframe_as_markdown(all_vad_performance_df)
    all_classifier_performance_table = format_dataframe_as_markdown(all_classifier_performance_df)
    
    # Fill template
    report = template_content.format(
        timestamp=timestamp,
        experiment_count=len(experiment_data),
        text_model_count=text_count,
        multimodal_count=multimodal_count,
        top_vad_models_table=top_vad_table,
        top_classifier_models_table=top_clf_table,
        text_models_table=text_table,
        multimodal_models_table=multimodal_table,
        all_vad_performance_table=all_vad_performance_table,
        all_classifier_performance_table=all_classifier_performance_table
    )
    
    return report

def generate_html_report(
    experiment_data: List[Dict[str, Any]], 
    text_count: int, 
    multimodal_count: int,
    template: Optional[str] = None
) -> str:
    """Generate an HTML report"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get template
    template_content = DEFAULT_HTML_TEMPLATE
    if template and os.path.exists(template):
        with open(template, 'r') as f:
            template_content = f.read()
    
    # Generate tables
    top_vad_df = select_top_vad_models(experiment_data)
    top_clf_df = select_top_classifier_models(experiment_data)
    text_df, multimodal_df = format_model_tables(experiment_data)
    
    # Generate all performance tables
    all_vad_performance_df = generate_all_vad_performance_table(experiment_data)
    all_classifier_performance_df = generate_all_classifier_performance_table(experiment_data)
    
    # Format tables as HTML
    top_vad_html = format_dataframe_as_html(top_vad_df)
    top_clf_html = format_dataframe_as_html(top_clf_df)
    text_html = format_dataframe_as_html(text_df)
    multimodal_html = format_dataframe_as_html(multimodal_df)
    all_vad_performance_html = format_dataframe_as_html(all_vad_performance_df)
    all_classifier_performance_html = format_dataframe_as_html(all_classifier_performance_df)
    
    # Fill template
    report = template_content.format(
        timestamp=timestamp,
        experiment_count=len(experiment_data),
        text_model_count=text_count,
        multimodal_count=multimodal_count,
        top_vad_models_html=top_vad_html,
        top_classifier_models_html=top_clf_html,
        text_models_html=text_html,
        multimodal_models_html=multimodal_html,
        all_vad_performance_html=all_vad_performance_html,
        all_classifier_performance_html=all_classifier_performance_html
    )
    
    return report

def generate_report(
    results_dir: str = RESULTS_DIR,
    output_dir: str = REPORTS_DIR,
    template: Optional[str] = None,
    format: str = "markdown"
) -> Optional[str]:
    """
    Generate a comprehensive report from experiment results
    
    Args:
        results_dir: Directory containing experiment results
        output_dir: Directory to save report
        template: Custom template file (optional)
        format: Output format ('markdown' or 'html')
        
    Returns:
        str: Path to generated report, or None if failed
    """
    # Ensure directories exist
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect experiment data
    experiment_data, text_count, multimodal_count = collect_experiment_data(results_dir)
    
    if not experiment_data:
        print("No experiment data found to generate report")
        return None
    
    # Generate visualizations
    generate_performance_visualizations(experiment_data, output_dir)
    
    # Generate report
    if format.lower() == "markdown":
        report_content = generate_markdown_report(
            experiment_data=experiment_data,
            text_count=text_count,
            multimodal_count=multimodal_count,
            template=template
        )
        report_path = os.path.join(output_dir, "experiment_report.md")
    else:  # HTML format
        report_content = generate_html_report(
            experiment_data=experiment_data,
            text_count=text_count,
            multimodal_count=multimodal_count,
            template=template
        )
        report_path = os.path.join(output_dir, "experiment_report.html")
    
    # Write report to file
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print(f"Report generated at {report_path}")
    return report_path

if __name__ == "__main__":
    # This can be run directly for testing
    format_arg = "markdown"
    if len(sys.argv) > 1:
        format_arg = sys.argv[1]
    
    generate_report(format=format_arg) 