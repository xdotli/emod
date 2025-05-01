"""
Visualization utilities for the EMOD project.

This module provides functions for visualizing results from the emotion recognition system.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import os

def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(10, 8), 
                          output_path=None, title='Confusion Matrix'):
    """
    Plot a confusion matrix.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        class_names (list, optional): List of class names
        figsize (tuple): Figure size
        output_path (str, optional): Path to save the figure
        title (str): Title of the plot
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize the confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Set labels
    if class_names is None:
        class_names = sorted(set(y_true))
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotate the tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    fmt = '.2f'
    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black")
    
    fig.tight_layout()
    
    # Save if path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
    
    return fig

def plot_vad_distributions(vad_values, emotion_labels=None, figsize=(18, 6), 
                          output_path=None, title='VAD Distributions'):
    """
    Plot distributions of Valence, Arousal, and Dominance values.
    
    Args:
        vad_values (ndarray): VAD values (shape: [n_samples, 3])
        emotion_labels (array-like, optional): Emotion labels for coloring
        figsize (tuple): Figure size
        output_path (str, optional): Path to save the figure
        title (str): Title of the plot
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Valence': vad_values[:, 0],
        'Arousal': vad_values[:, 1],
        'Dominance': vad_values[:, 2],
    })
    
    if emotion_labels is not None:
        df['Emotion'] = emotion_labels
        
        # Plot distributions
        for i, dim in enumerate(['Valence', 'Arousal', 'Dominance']):
            sns.violinplot(x='Emotion', y=dim, data=df, ax=axes[i])
            axes[i].set_title(f'{dim} by Emotion')
            axes[i].set_xlabel('Emotion')
            axes[i].set_ylabel(dim)
    else:
        # Plot distributions without emotion grouping
        for i, dim in enumerate(['Valence', 'Arousal', 'Dominance']):
            sns.histplot(df[dim], kde=True, ax=axes[i])
            axes[i].set_title(f'{dim} Distribution')
            axes[i].set_xlabel(dim)
            axes[i].set_ylabel('Count')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save if path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
    
    return fig

def plot_vad_2d(vad_values, emotion_labels=None, figsize=(12, 10), 
                output_path=None, title='VAD 2D Projections'):
    """
    Plot 2D projections of VAD values.
    
    Args:
        vad_values (ndarray): VAD values (shape: [n_samples, 3])
        emotion_labels (array-like, optional): Emotion labels for coloring
        figsize (tuple): Figure size
        output_path (str, optional): Path to save the figure
        title (str): Title of the plot
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Valence': vad_values[:, 0],
        'Arousal': vad_values[:, 1],
        'Dominance': vad_values[:, 2],
    })
    
    # Define plot pairs
    pairs = [
        ('Valence', 'Arousal'),
        ('Valence', 'Dominance'),
        ('Arousal', 'Dominance')
    ]
    
    if emotion_labels is not None:
        df['Emotion'] = emotion_labels
        
        # Plot scatter plots with hue
        for i, (x, y) in enumerate(pairs):
            sns.scatterplot(x=x, y=y, hue='Emotion', data=df, ax=axes[i])
            axes[i].set_title(f'{x} vs {y}')
    else:
        # Plot scatter plots without hue
        for i, (x, y) in enumerate(pairs):
            sns.scatterplot(x=x, y=y, data=df, ax=axes[i])
            axes[i].set_title(f'{x} vs {y}')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save if path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
    
    return fig

def plot_metrics_comparison(text_metrics, multimodal_metrics, metric_names, 
                           figsize=(12, 6), output_path=None, 
                           title='Text vs Multimodal Metrics Comparison'):
    """
    Compare metrics between text-only and multimodal approaches.
    
    Args:
        text_metrics (dict): Metrics from text-only approach
        multimodal_metrics (dict): Metrics from multimodal approach
        metric_names (list): Names of metrics to compare
        figsize (tuple): Figure size
        output_path (str, optional): Path to save the figure
        title (str): Title of the plot
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data
    text_values = [text_metrics[name] for name in metric_names]
    multimodal_values = [multimodal_metrics[name] for name in metric_names]
    
    # Set x positions
    x = np.arange(len(metric_names))
    width = 0.35
    
    # Plot bars
    ax.bar(x - width/2, text_values, width, label='Text-only')
    ax.bar(x + width/2, multimodal_values, width, label='Multimodal')
    
    # Add percentage improvement
    for i, (text_val, mm_val) in enumerate(zip(text_values, multimodal_values)):
        improvement = (mm_val - text_val) / text_val * 100
        if improvement > 0:
            plt.text(i, max(text_val, mm_val) * 1.05, 
                    f"{improvement:.1f}%↑", 
                    ha='center', fontsize=10, color='green')
        else:
            plt.text(i, max(text_val, mm_val) * 1.05, 
                    f"{-improvement:.1f}%↓", 
                    ha='center', fontsize=10, color='red')
    
    # Add labels and title
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()
    
    plt.tight_layout()
    
    # Save if path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
    
    return fig

def plot_learning_curves(train_losses, val_losses, figsize=(10, 6), 
                        output_path=None, title='Learning Curves'):
    """
    Plot learning curves from training history.
    
    Args:
        train_losses (list): Training losses
        val_losses (list): Validation losses
        figsize (tuple): Figure size
        output_path (str, optional): Path to save the figure
        title (str): Title of the plot
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    # Add labels and title
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    
    # Save if path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
    
    return fig

def create_results_dashboard(metrics, vad_values, y_true, y_pred, 
                            output_dir, prefix=''):
    """
    Create a comprehensive dashboard of visualizations.
    
    Args:
        metrics (dict): Evaluation metrics
        vad_values (ndarray): VAD values
        y_true (array-like): True emotion labels
        y_pred (array-like): Predicted emotion labels
        output_dir (str): Directory to save figures
        prefix (str): Prefix for file names
        
    Returns:
        dict: Paths to the generated figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot confusion matrix
    cm_path = os.path.join(output_dir, f'{prefix}confusion_matrix.png')
    plot_confusion_matrix(y_true, y_pred, output_path=cm_path)
    
    # Plot VAD distributions
    vad_dist_path = os.path.join(output_dir, f'{prefix}vad_distributions.png')
    plot_vad_distributions(vad_values, y_true, output_path=vad_dist_path)
    
    # Plot VAD 2D projections
    vad_2d_path = os.path.join(output_dir, f'{prefix}vad_2d.png')
    plot_vad_2d(vad_values, y_true, output_path=vad_2d_path)
    
    # Create summary report
    report_path = os.path.join(output_dir, f'{prefix}summary.txt')
    with open(report_path, 'w') as f:
        f.write("EMOD Emotion Recognition Results\n")
        f.write("================================\n\n")
        
        f.write("Emotion Classification Metrics:\n")
        f.write(f"  Accuracy: {metrics.get('accuracy', 'N/A'):.4f}\n")
        f.write(f"  Weighted F1: {metrics.get('f1_weighted', 'N/A'):.4f}\n")
        f.write(f"  Macro F1: {metrics.get('f1_macro', 'N/A'):.4f}\n\n")
        
        if 'classification_report' in metrics:
            f.write("Classification Report:\n")
            for cls, values in metrics['classification_report'].items():
                if isinstance(values, dict):
                    f.write(f"  {cls}:\n")
                    for metric, value in values.items():
                        f.write(f"    {metric}: {value:.4f}\n")
                    f.write("\n")
        
        if 'vad_metrics' in metrics:
            f.write("VAD Prediction Metrics:\n")
            vad_dims = ['Valence', 'Arousal', 'Dominance']
            for i, dim in enumerate(vad_dims):
                f.write(f"  {dim}:\n")
                for metric in ['mse', 'rmse', 'mae', 'r2']:
                    if metric in metrics['vad_metrics']:
                        f.write(f"    {metric.upper()}: {metrics['vad_metrics'][metric][i]:.4f}\n")
                f.write("\n")
    
    return {
        'confusion_matrix': cm_path,
        'vad_distributions': vad_dist_path,
        'vad_2d': vad_2d_path,
        'summary_report': report_path
    } 