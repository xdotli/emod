"""
Visualization utilities for emotion recognition.
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from .vad_emotion_mapping import get_emotion_color

def plot_vad_distribution(vad_values, emotions=None, title='VAD Distribution'):
    """
    Plot 3D distribution of VAD values.
    
    Args:
        vad_values: Array of VAD values (valence, arousal, dominance)
        emotions: Array of emotion labels (optional)
        title: Plot title
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract VAD components
    valence = vad_values[:, 0]
    arousal = vad_values[:, 1]
    dominance = vad_values[:, 2]
    
    if emotions is not None:
        # Color points by emotion
        colors = [get_emotion_color(emotion) for emotion in emotions]
        scatter = ax.scatter(valence, arousal, dominance, c=colors, alpha=0.7)
        
        # Add legend
        unique_emotions = np.unique(emotions)
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=get_emotion_color(emotion), 
                          markersize=10, label=emotion) 
                          for emotion in unique_emotions]
        ax.legend(handles=legend_elements)
    else:
        # Use default coloring
        scatter = ax.scatter(valence, arousal, dominance, alpha=0.7)
    
    ax.set_xlabel('Valence')
    ax.set_ylabel('Arousal')
    ax.set_zlabel('Dominance')
    ax.set_title(title)
    
    # Set axis limits
    ax.set_xlim(1, 5)
    ax.set_ylim(1, 5)
    ax.set_zlim(1, 5)
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix(cm, class_names, title='Confusion Matrix'):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot confusion matrix
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    
    # Add labels
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    plt.tight_layout()
    
    return fig

def plot_vad_predictions(true_vad, pred_vad, title='VAD Predictions'):
    """
    Plot true vs predicted VAD values.
    
    Args:
        true_vad: Array of true VAD values
        pred_vad: Array of predicted VAD values
        title: Plot title
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    vad_labels = ['Valence', 'Arousal', 'Dominance']
    
    for i, ax in enumerate(axes):
        ax.scatter(true_vad[:, i], pred_vad[:, i], alpha=0.5)
        
        # Add diagonal line (perfect predictions)
        min_val = min(true_vad[:, i].min(), pred_vad[:, i].min())
        max_val = max(true_vad[:, i].max(), pred_vad[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        ax.set_xlabel(f'True {vad_labels[i]}')
        ax.set_ylabel(f'Predicted {vad_labels[i]}')
        ax.set_title(f'{vad_labels[i]} Predictions')
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig

def plot_emotion_distribution(emotions, title='Emotion Distribution'):
    """
    Plot distribution of emotions.
    
    Args:
        emotions: Array of emotion labels
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Count emotions
    unique_emotions, counts = np.unique(emotions, return_counts=True)
    
    # Sort by count
    sort_idx = np.argsort(counts)[::-1]
    unique_emotions = unique_emotions[sort_idx]
    counts = counts[sort_idx]
    
    # Plot bar chart
    bars = ax.bar(unique_emotions, counts)
    
    # Color bars by emotion
    for i, bar in enumerate(bars):
        bar.set_color(get_emotion_color(unique_emotions[i]))
    
    ax.set_xlabel('Emotion')
    ax.set_ylabel('Count')
    ax.set_title(title)
    
    # Add count labels
    for i, count in enumerate(counts):
        ax.text(i, count + 0.1, str(count), ha='center')
    
    plt.tight_layout()
    
    return fig
