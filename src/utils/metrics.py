"""
Evaluation metrics for emotion recognition.
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def calculate_vad_metrics(true_vad, pred_vad):
    """
    Calculate metrics for VAD prediction.
    
    Args:
        true_vad (np.ndarray): Array of true VAD values
        pred_vad (np.ndarray): Array of predicted VAD values
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    mse = np.mean((true_vad - pred_vad) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(true_vad - pred_vad))
    
    # Calculate metrics for each dimension
    dim_metrics = {}
    for i, dim in enumerate(['valence', 'arousal', 'dominance']):
        dim_mse = np.mean((true_vad[:, i] - pred_vad[:, i]) ** 2)
        dim_rmse = np.sqrt(dim_mse)
        dim_mae = np.mean(np.abs(true_vad[:, i] - pred_vad[:, i]))
        
        dim_metrics[dim] = {
            'mse': dim_mse,
            'rmse': dim_rmse,
            'mae': dim_mae
        }
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'dim_metrics': dim_metrics
    }

def calculate_emotion_metrics(true_emotions, pred_emotions, idx_to_emotion=None):
    """
    Calculate metrics for emotion prediction.
    
    Args:
        true_emotions (np.ndarray): Array of true emotion indices
        pred_emotions (np.ndarray): Array of predicted emotion indices
        idx_to_emotion (dict): Mapping from indices to emotion labels
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    accuracy = accuracy_score(true_emotions, pred_emotions)
    f1_macro = f1_score(true_emotions, pred_emotions, average='macro')
    f1_weighted = f1_score(true_emotions, pred_emotions, average='weighted')
    conf_matrix = confusion_matrix(true_emotions, pred_emotions)
    
    # Convert to emotion labels if mapping is provided
    if idx_to_emotion:
        true_labels = [idx_to_emotion[idx] for idx in true_emotions]
        pred_labels = [idx_to_emotion[idx] for idx in pred_emotions]
        class_report = classification_report(true_labels, pred_labels, output_dict=True)
    else:
        class_report = classification_report(true_emotions, pred_emotions, output_dict=True)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
    }

def log_metrics(metrics, output_dir, prefix=''):
    """
    Log metrics to a file.
    
    Args:
        metrics (dict): Dictionary containing evaluation metrics
        output_dir (str): Directory to save the metrics
        prefix (str): Prefix for the output file name
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a formatted string of metrics
    metrics_str = []
    
    if 'vad_metrics' in metrics:
        vad_metrics = metrics['vad_metrics']
        metrics_str.append(f"VAD Metrics:")
        metrics_str.append(f"  MSE: {vad_metrics['mse']:.4f}")
        metrics_str.append(f"  RMSE: {vad_metrics['rmse']:.4f}")
        metrics_str.append(f"  MAE: {vad_metrics['mae']:.4f}")
        
        if 'dim_metrics' in vad_metrics:
            dim_metrics = vad_metrics['dim_metrics']
            for dim, metrics_dict in dim_metrics.items():
                metrics_str.append(f"  {dim.capitalize()}:")
                metrics_str.append(f"    MSE: {metrics_dict['mse']:.4f}")
                metrics_str.append(f"    RMSE: {metrics_dict['rmse']:.4f}")
                metrics_str.append(f"    MAE: {metrics_dict['mae']:.4f}")
    
    if 'emotion_metrics' in metrics:
        emotion_metrics = metrics['emotion_metrics']
        metrics_str.append(f"Emotion Metrics:")
        metrics_str.append(f"  Accuracy: {emotion_metrics['accuracy']:.4f}")
        metrics_str.append(f"  F1 (macro): {emotion_metrics['f1_macro']:.4f}")
        metrics_str.append(f"  F1 (weighted): {emotion_metrics['f1_weighted']:.4f}")
        
        if 'classification_report' in emotion_metrics:
            class_report = emotion_metrics['classification_report']
            metrics_str.append(f"  Classification Report:")
            
            for class_name, class_metrics in class_report.items():
                if isinstance(class_metrics, dict):
                    metrics_str.append(f"    {class_name}:")
                    metrics_str.append(f"      Precision: {class_metrics['precision']:.4f}")
                    metrics_str.append(f"      Recall: {class_metrics['recall']:.4f}")
                    metrics_str.append(f"      F1: {class_metrics['f1-score']:.4f}")
                    metrics_str.append(f"      Support: {class_metrics['support']}")
    
    # Write metrics to file
    output_path = os.path.join(output_dir, f"{prefix}metrics.txt")
    with open(output_path, 'w') as f:
        f.write('\n'.join(metrics_str))
    
    # Also save as JSON for easier parsing
    import json
    json_path = os.path.join(output_dir, f"{prefix}metrics.json")
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
