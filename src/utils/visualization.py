"""
Visualization utilities for emotion recognition.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

def plot_confusion_matrix(true_labels, pred_labels, output_path=None):
    """
    Plot confusion matrix.
    
    Args:
        true_labels (list): List of true emotion labels
        pred_labels (list): List of predicted emotion labels
        output_path (str): Path to save the plot
    """
    # Create confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Normalize by row (true labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Get unique labels
    unique_labels = sorted(set(true_labels) | set(pred_labels))
    
    # Create figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues',
        xticklabels=unique_labels,
        yticklabels=unique_labels
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save or show the plot
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_vad_distribution(true_vad, pred_vad, output_dir=None):
    """
    Plot distribution of VAD values.
    
    Args:
        true_vad (np.ndarray): Array of true VAD values
        pred_vad (np.ndarray): Array of predicted VAD values
        output_dir (str): Directory to save the plots
    """
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
        
        # Save or show the plot
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'{dim.lower()}_distribution.png')
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    # Plot scatter plots for each pair of dimensions
    dim_pairs = [(0, 1), (0, 2), (1, 2)]
    pair_names = [('Valence', 'Arousal'), ('Valence', 'Dominance'), ('Arousal', 'Dominance')]
    
    for (i, j), (dim_i, dim_j) in zip(dim_pairs, pair_names):
        plt.figure(figsize=(10, 8))
        
        # Plot scatter points
        plt.scatter(true_vad[:, i], true_vad[:, j], alpha=0.5, label='True')
        plt.scatter(pred_vad[:, i], pred_vad[:, j], alpha=0.5, label='Predicted')
        
        plt.xlabel(dim_i)
        plt.ylabel(dim_j)
        plt.title(f'{dim_i} vs {dim_j}')
        plt.legend()
        
        # Save or show the plot
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'{dim_i.lower()}_{dim_j.lower()}_scatter.png')
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def plot_vad_by_emotion(vad_values, emotion_labels, output_dir=None):
    """
    Plot VAD values grouped by emotion.
    
    Args:
        vad_values (np.ndarray): Array of VAD values
        emotion_labels (list): List of emotion labels
        output_dir (str): Directory to save the plots
    """
    # Create DataFrame
    df = pd.DataFrame({
        'Valence': vad_values[:, 0],
        'Arousal': vad_values[:, 1],
        'Dominance': vad_values[:, 2],
        'Emotion': emotion_labels
    })
    
    # Plot boxplots for each dimension
    dim_names = ['Valence', 'Arousal', 'Dominance']
    
    for dim in dim_names:
        plt.figure(figsize=(12, 6))
        
        # Create boxplot
        sns.boxplot(x='Emotion', y=dim, data=df)
        
        plt.xlabel('Emotion')
        plt.ylabel(dim)
        plt.title(f'{dim} by Emotion')
        plt.xticks(rotation=45)
        
        # Save or show the plot
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'{dim.lower()}_by_emotion.png')
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    # Plot 3D scatter plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get unique emotions
    unique_emotions = sorted(set(emotion_labels))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_emotions)))
    
    for emotion, color in zip(unique_emotions, colors):
        mask = df['Emotion'] == emotion
        ax.scatter(
            df.loc[mask, 'Valence'],
            df.loc[mask, 'Arousal'],
            df.loc[mask, 'Dominance'],
            color=color,
            label=emotion,
            alpha=0.7
        )
    
    ax.set_xlabel('Valence')
    ax.set_ylabel('Arousal')
    ax.set_zlabel('Dominance')
    ax.set_title('VAD Space by Emotion')
    ax.legend()
    
    # Save or show the plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'vad_3d_by_emotion.png')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_tsne_vad(vad_values, emotion_labels, output_path=None):
    """
    Plot t-SNE visualization of VAD values colored by emotion.
    
    Args:
        vad_values (np.ndarray): Array of VAD values
        emotion_labels (list): List of emotion labels
        output_path (str): Path to save the plot
    """
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    vad_tsne = tsne.fit_transform(vad_values)
    
    # Create DataFrame
    df = pd.DataFrame({
        'x': vad_tsne[:, 0],
        'y': vad_tsne[:, 1],
        'Emotion': emotion_labels
    })
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Get unique emotions
    unique_emotions = sorted(set(emotion_labels))
    
    # Create scatter plot
    for emotion in unique_emotions:
        mask = df['Emotion'] == emotion
        plt.scatter(
            df.loc[mask, 'x'],
            df.loc[mask, 'y'],
            label=emotion,
            alpha=0.7
        )
    
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.title('t-SNE Visualization of VAD Values')
    plt.legend()
    
    # Save or show the plot
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
