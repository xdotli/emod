#!/usr/bin/env python3
"""
Test script to analyze the rule-based VAD-to-emotion mapping.

This script analyzes the rule-based VAD-to-emotion mapping and creates
visualizations to show the distribution of emotions in VAD space.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from process_vad import vad_to_emotion

def analyze_vad_mapping():
    """Analyze the rule-based VAD-to-emotion mapping."""
    # Create a grid of VAD values
    valence = np.linspace(-1, 1, 20)
    arousal = np.linspace(-1, 1, 20)
    dominance = np.linspace(-1, 1, 20)
    
    # Generate emotion mappings for various combinations
    data = []
    
    # 1. Valence-Arousal plane with fixed dominance levels
    for d in [-0.8, -0.4, 0, 0.4, 0.8]:
        for v in valence:
            for a in arousal:
                emotion = vad_to_emotion(v, a, d)
                data.append({
                    'valence': v,
                    'arousal': a,
                    'dominance': d,
                    'emotion': emotion
                })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Count occurrences of each emotion
    emotion_counts = df['emotion'].value_counts()
    print("Emotion distribution in VAD space:")
    print(tabulate(
        [[emotion, count, f"{count/len(df)*100:.2f}%"] for emotion, count in emotion_counts.items()],
        headers=["Emotion", "Count", "Percentage"],
        tablefmt="grid"
    ))
    
    return df

def visualize_vad_mapping(df):
    """Create visualizations for the VAD-to-emotion mapping."""
    # Create 2D plots for different dominance levels
    dominance_levels = sorted(df['dominance'].unique())
    
    # Define colors for emotions
    emotion_colors = {
        'happy': 'yellow',
        'excited': 'orange',
        'content': 'green',
        'relaxed': 'lightgreen',
        'angry': 'red',
        'fearful': 'purple',
        'disgusted': 'brown',
        'sad': 'blue'
    }
    
    # Create subplots for each dominance level
    fig, axes = plt.subplots(1, len(dominance_levels), figsize=(20, 5), sharex=True, sharey=True)
    
    # Set a common title
    fig.suptitle('VAD-to-Emotion Mapping (Valence-Arousal planes at different Dominance levels)', fontsize=16)
    
    for i, d in enumerate(dominance_levels):
        # Get data for this dominance level
        subset = df[df['dominance'] == d]
        
        # Create a pivot table for the 2D grid
        pivot = subset.pivot_table(
            index='arousal', 
            columns='valence', 
            values='emotion', 
            aggfunc=lambda x: x.iloc[0]
        )
        
        # Plot as scatter points
        for emotion in subset['emotion'].unique():
            emotion_subset = subset[subset['emotion'] == emotion]
            axes[i].scatter(
                emotion_subset['valence'],
                emotion_subset['arousal'],
                c=emotion_colors.get(emotion, 'gray'),
                label=emotion if i == 0 else None,  # Only add label in first plot
                alpha=0.7,
                s=50
            )
        
        # Add grid lines at quadrant boundaries
        axes[i].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[i].axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Set labels
        axes[i].set_xlabel('Valence')
        axes[i].set_ylabel('Arousal' if i == 0 else '')
        axes[i].set_title(f'Dominance = {d:.1f}')
        
        # Set axis limits
        axes[i].set_xlim(-1.1, 1.1)
        axes[i].set_ylim(-1.1, 1.1)
    
    # Add a legend outside the plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=4)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig('vad_emotion_mapping.png')
    print("Visualization saved as 'vad_emotion_mapping.png'")
    
    # Create a 3D visualization
    try:
        # Sample the data to reduce points for 3D plot
        sample_size = min(2000, len(df))
        df_sample = df.sample(sample_size, random_state=42)
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each emotion category
        for emotion in df_sample['emotion'].unique():
            subset = df_sample[df_sample['emotion'] == emotion]
            ax.scatter(
                subset['valence'],
                subset['arousal'],
                subset['dominance'],
                c=emotion_colors.get(emotion, 'gray'),
                label=emotion,
                alpha=0.7
            )
        
        # Add grid planes at zero
        valence_grid, arousal_grid = np.meshgrid([-1, 1], [-1, 1])
        dominance_zeros = np.zeros_like(valence_grid)
        ax.plot_surface(valence_grid, arousal_grid, dominance_zeros, alpha=0.1, color='gray')
        
        valence_grid, dominance_grid = np.meshgrid([-1, 1], [-1, 1])
        arousal_zeros = np.zeros_like(valence_grid)
        ax.plot_surface(valence_grid, arousal_zeros, dominance_grid, alpha=0.1, color='gray')
        
        arousal_grid, dominance_grid = np.meshgrid([-1, 1], [-1, 1])
        valence_zeros = np.zeros_like(arousal_grid)
        ax.plot_surface(valence_zeros, arousal_grid, dominance_grid, alpha=0.1, color='gray')
        
        # Set labels and title
        ax.set_xlabel('Valence')
        ax.set_ylabel('Arousal')
        ax.set_zlabel('Dominance')
        ax.set_title('3D Distribution of Emotions in VAD Space')
        
        # Set limits
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        
        # Add legend
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('vad_emotion_mapping_3d.png')
        print("3D visualization saved as 'vad_emotion_mapping_3d.png'")
    except Exception as e:
        print(f"Error creating 3D visualization: {e}")

def validate_test_cases():
    """Test the rule-based VAD-to-emotion mapping with specific test cases."""
    # Define test VAD values that cover different regions of the VAD space
    test_vad_values = [
        # Standard emotions
        [0.8, 0.7, 0.6],    # Typical happy
        [-0.8, -0.5, -0.6],  # Typical sad
        [-0.6, 0.8, 0.7],   # Typical angry
        [0.0, 0.0, 0.0],    # Neutral
        
        # Edge cases
        [0.1, 0.1, 0.1],    # Slight positive
        [-0.1, -0.1, -0.1],  # Slight negative
        [0.9, -0.9, 0.9],   # High valence, low arousal, high dominance
        [-0.9, 0.9, -0.9],  # Low valence, high arousal, low dominance
        
        # Corner cases
        [1.0, 1.0, 1.0],    # Maximum positive
        [-1.0, -1.0, -1.0],  # Maximum negative
        [1.0, -1.0, -1.0],  # Mixed extremes
        [-1.0, 1.0, 1.0]    # Mixed extremes
    ]
    
    # Get predictions
    predictions = []
    for vad in test_vad_values:
        valence, arousal, dominance = vad
        emotion = vad_to_emotion(valence, arousal, dominance)
        predictions.append({
            'valence': valence,
            'arousal': arousal,
            'dominance': dominance,
            'emotion': emotion
        })
    
    # Print results
    print("\nRule-based VAD-to-Emotion Results for Test Cases:")
    print(tabulate(
        [[i+1, f"[{p['valence']:.1f}, {p['arousal']:.1f}, {p['dominance']:.1f}]", p['emotion']] 
         for i, p in enumerate(predictions)],
        headers=["#", "VAD Values", "Predicted Emotion"],
        tablefmt="grid"
    ))
    
    return predictions

def main():
    """Analyze the VAD-to-emotion mapping and create visualizations."""
    print("=" * 80)
    print("ANALYZING VAD-TO-EMOTION MAPPING")
    print("=" * 80)
    
    # Analyze the mapping
    df = analyze_vad_mapping()
    
    # Validate with test cases
    test_results = validate_test_cases()
    
    # Visualize the mapping
    visualize_vad_mapping(df)
    
    print("\nRule-based VAD-to-Emotion Mapping Analysis:")
    print("  - The rule-based approach uses quadrants in the VAD space to determine emotions")
    print("  - Emotions are distributed based on specific regions of the VAD space")
    print("  - Valence, Arousal, and Dominance all affect the final emotion prediction")
    print("  - The visualization shows how emotions are distributed across the VAD space")
    
    print("\nResults saved to 'vad_emotion_mapping.png' and 'vad_emotion_mapping_3d.png'")
    
    print("=" * 80)

if __name__ == "__main__":
    main() 