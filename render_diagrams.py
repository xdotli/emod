#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from matplotlib.path import Path
import re
import networkx as nx

# Create output directory
os.makedirs("CS297-298-Xiangyi-Report/Figures", exist_ok=True)

# Function to extract diagram titles from markdown
def extract_titles_from_markdown(file_path):
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        titles = re.findall(r'## \d+\. (.*)', content)
        return titles
    except Exception as e:
        print(f"Error extracting titles: {e}")
        return [
            "High-Level System Architecture",
            "Text Model Architecture Detail",
            "Fusion Strategies Comparison",
            "MFCC Feature Extraction Pipeline",
            "Experiment Execution Framework"
        ]

# Get titles from the markdown file
default_titles = [
    "High-Level System Architecture",
    "Text Model Architecture Detail",
    "Fusion Strategies Comparison",
    "MFCC Feature Extraction Pipeline",
    "Experiment Execution Framework"
]

try:
    titles = extract_titles_from_markdown('model_architecture_diagrams.md')
    if not titles or len(titles) < 5:
        titles = default_titles
except:
    titles = default_titles

# 1. System Architecture
def create_system_architecture():
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Define the boxes
    boxes = [
        {"label": "Audio Signal", "position": (0.25, 0.9), "size": (0.2, 0.1)},
        {"label": "Text Transcript", "position": (0.75, 0.9), "size": (0.2, 0.1)},
        {"label": "Audio Feature\nExtraction", "position": (0.25, 0.75), "size": (0.2, 0.1)},
        {"label": "Text\nTokenization", "position": (0.75, 0.75), "size": (0.2, 0.1)},
        {"label": "Audio Model", "position": (0.25, 0.6), "size": (0.2, 0.1)},
        {"label": "Transformer\nModel", "position": (0.75, 0.6), "size": (0.2, 0.1)},
        {"label": "Early Fusion", "position": (0.25, 0.4), "size": (0.2, 0.1)},
        {"label": "Late Fusion", "position": (0.5, 0.4), "size": (0.2, 0.1)},
        {"label": "Hybrid Fusion", "position": (0.75, 0.4), "size": (0.2, 0.1)},
        {"label": "Emotion Prediction", "position": (0.5, 0.2), "size": (0.3, 0.1)}
    ]
    
    # Add boxes
    for box in boxes:
        rect = patches.Rectangle(
            box["position"], box["size"][0], box["size"][1], 
            linewidth=1, edgecolor='black', facecolor='lightblue', alpha=0.7
        )
        ax.add_patch(rect)
        ax.text(
            box["position"][0] + box["size"][0]/2, 
            box["position"][1] + box["size"][1]/2,
            box["label"], 
            ha='center', va='center'
        )
    
    # Draw arrows
    arrows = [
        {"start": (0.25, 0.9), "end": (0.25, 0.85)},
        {"start": (0.75, 0.9), "end": (0.75, 0.85)},
        {"start": (0.25, 0.75), "end": (0.25, 0.7)},
        {"start": (0.75, 0.75), "end": (0.75, 0.7)},
        {"start": (0.25, 0.6), "end": (0.25, 0.5)},
        {"start": (0.75, 0.6), "end": (0.75, 0.5)},
        {"start": (0.25, 0.5), "end": (0.25, 0.4)},
        {"start": (0.75, 0.5), "end": (0.75, 0.4)},
        {"start": (0.25, 0.4), "end": (0.5, 0.3)},
        {"start": (0.5, 0.4), "end": (0.5, 0.3)},
        {"start": (0.75, 0.4), "end": (0.5, 0.3)},
        {"start": (0.5, 0.3), "end": (0.5, 0.2)}
    ]
    
    for arrow in arrows:
        ax.arrow(
            arrow["start"][0] + 0.1, arrow["start"][1], 
            0, arrow["end"][1] - arrow["start"][1],
            head_width=0.02, head_length=0.02, 
            fc='black', ec='black', linewidth=1
        )
    
    # Add section labels
    ax.text(0.5, 0.95, "Input Data", fontsize=14, ha='center', fontweight='bold')
    ax.text(0.5, 0.68, "Stage 1: Modality-Specific Processing", fontsize=14, ha='center', fontweight='bold')
    ax.text(0.5, 0.48, "Stage 2: Multimodal Fusion", fontsize=14, ha='center', fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.title(titles[0], fontsize=18, pad=20)
    plt.tight_layout()
    plt.savefig("CS297-298-Xiangyi-Report/Figures/system_architecture.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_improved_system_architecture(output_path='CS297-298-Xiangyi-Report/Figures/system_architecture_fixed.png'):
    """
    Generate an improved version of the system architecture diagram with better alignment of labels.
    Specifically fixes the "Input Data" label positioning.
    """
    plt.figure(figsize=(10, 8), dpi=300)
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes with proper positioning to fix label alignment issues
    # Input layer
    G.add_node("Input Data", pos=(0.5, 1.0))
    G.add_node("Audio Signal", pos=(0.25, 0.9))
    G.add_node("Text Transcript", pos=(0.75, 0.9))
    
    # Processing layer
    G.add_node("Audio Feature\nExtraction", pos=(0.25, 0.7))
    G.add_node("Text\nTokenization", pos=(0.75, 0.7))
    
    # Model layer
    G.add_node("Audio Model", pos=(0.25, 0.5))
    G.add_node("Transformer\nModel", pos=(0.75, 0.5))
    
    # Fusion layer header
    G.add_node("Stage 2: Multimodal Fusion", pos=(0.5, 0.4))
    
    # Fusion methods
    G.add_node("Early Fusion", pos=(0.25, 0.3))
    G.add_node("Late Fusion", pos=(0.5, 0.3))
    G.add_node("Hybrid Fusion", pos=(0.75, 0.3))
    
    # Output
    G.add_node("Emotion Prediction", pos=(0.5, 0.1))
    
    # Stage label
    G.add_node("Stage 1: Modality-Specific Processing", pos=(0.5, 0.6))
    
    # Add edges
    # Input connections
    G.add_edge("Input Data", "Audio Signal")
    G.add_edge("Input Data", "Text Transcript")
    
    # Audio path
    G.add_edge("Audio Signal", "Audio Feature\nExtraction")
    G.add_edge("Audio Feature\nExtraction", "Audio Model")
    G.add_edge("Audio Model", "Early Fusion")
    G.add_edge("Audio Model", "Late Fusion")
    G.add_edge("Audio Model", "Hybrid Fusion")
    
    # Text path
    G.add_edge("Text Transcript", "Text\nTokenization")
    G.add_edge("Text\nTokenization", "Transformer\nModel")
    G.add_edge("Transformer\nModel", "Early Fusion")
    G.add_edge("Transformer\nModel", "Late Fusion")
    G.add_edge("Transformer\nModel", "Hybrid Fusion")
    
    # Output connections
    G.add_edge("Early Fusion", "Emotion Prediction")
    G.add_edge("Late Fusion", "Emotion Prediction")
    G.add_edge("Hybrid Fusion", "Emotion Prediction")
    
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw the graph with a light blue theme
    node_sizes = {
        "Input Data": 2000,
        "Stage 1: Modality-Specific Processing": 3000,
        "Stage 2: Multimodal Fusion": 3000
    }
    
    default_size = 2000
    node_sizes = {node: node_sizes.get(node, default_size) for node in G.nodes()}
    
    # Draw non-label nodes
    regular_nodes = [n for n in G.nodes() if n not in ["Stage 1: Modality-Specific Processing", "Stage 2: Multimodal Fusion"]]
    nx.draw_networkx_nodes(
        G, pos, 
        nodelist=regular_nodes,
        node_size=[node_sizes[n] for n in regular_nodes],
        node_color='lightblue', 
        edgecolors='black',
        linewidths=1
    )
    
    # Draw stage label nodes with different style - transparent with no borders
    stage_nodes = ["Stage 1: Modality-Specific Processing", "Stage 2: Multimodal Fusion"]
    nx.draw_networkx_nodes(
        G, pos, 
        nodelist=stage_nodes,
        node_size=[node_sizes[n] for n in stage_nodes],
        node_color='none',  # Transparent
        edgecolors='none'  # No border
    )
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='black', width=1.5, arrowsize=15)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    
    # Remove axis
    plt.axis('off')
    
    # Add title
    plt.title("High-Level System Architecture", fontsize=14)
    
    # Save with high quality
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"System architecture diagram saved to {output_path}")
    return output_path

# 2. Text Model Architecture
def create_text_model_architecture():
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define boxes
    boxes = [
        {"label": "Text Input", "position": (0.5, 0.9), "size": (0.2, 0.08)},
        {"label": "Tokenizer", "position": (0.5, 0.78), "size": (0.2, 0.08)},
        {"label": "Token + Position\nEmbedding", "position": (0.5, 0.66), "size": (0.3, 0.08)},
        {"label": "Self-Attention Layer", "position": (0.5, 0.54), "size": (0.3, 0.08)},
        {"label": "Feed-Forward Network", "position": (0.5, 0.42), "size": (0.3, 0.08)},
        {"label": "Layer Normalization", "position": (0.5, 0.3), "size": (0.3, 0.08)},
        {"label": "CLS Token\nRepresentation", "position": (0.5, 0.18), "size": (0.25, 0.08)},
        {"label": "Classification Head", "position": (0.5, 0.06), "size": (0.25, 0.08)}
    ]
    
    # Add boxes and labels
    for i, box in enumerate(boxes):
        if 2 <= i <= 5:  # Transformer encoder components
            color = 'lightgreen'
            alpha = 0.6
        else:
            color = 'lightblue'
            alpha = 0.7
            
        rect = patches.Rectangle(
            (box["position"][0] - box["size"][0]/2, box["position"][1]), 
            box["size"][0], box["size"][1], 
            linewidth=1, edgecolor='black', facecolor=color, alpha=alpha
        )
        ax.add_patch(rect)
        ax.text(
            box["position"][0], 
            box["position"][1] + box["size"][1]/2,
            box["label"], 
            ha='center', va='center'
        )
    
    # Draw arrows
    for i in range(len(boxes) - 1):
        start_y = boxes[i]["position"][1]
        end_y = boxes[i+1]["position"][1] + boxes[i+1]["size"][1]
        ax.arrow(
            boxes[i]["position"][0], start_y, 
            0, end_y - start_y - 0.08,
            head_width=0.02, head_length=0.01, 
            fc='black', ec='black', linewidth=1
        )
    
    # Add Transformer Encoder box
    encoder_rect = patches.Rectangle(
        (0.33, 0.28), 0.34, 0.48, 
        linewidth=2, edgecolor='darkgreen', facecolor='none', linestyle='--'
    )
    ax.add_patch(encoder_rect)
    ax.text(0.26, 0.52, "Transformer\nEncoder", fontsize=12, ha='right', fontweight='bold')
    
    # Add recursive arrow to show multiple layers
    ax.annotate(
        "Next Layer", 
        xy=(0.68, 0.34), xytext=(0.8, 0.42),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3")
    )
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.title(titles[1], fontsize=18, pad=20)
    plt.tight_layout()
    plt.savefig("CS297-298-Xiangyi-Report/Figures/text_model_architecture.png", dpi=300, bbox_inches='tight')
    plt.close()

# 3. Fusion Strategies
def create_fusion_strategies():
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Set up positions
    early_x, late_x, hybrid_x = 0.25, 0.5, 0.75
    start_y = 0.9
    box_width, box_height = 0.2, 0.08
    
    # Create colored backgrounds for each fusion type
    early_bg = patches.Rectangle((0.15, 0.55), 0.2, 0.3, linewidth=1, edgecolor='green', 
                                facecolor='lightgreen', alpha=0.3)
    late_bg = patches.Rectangle((0.4, 0.55), 0.2, 0.3, linewidth=1, edgecolor='blue', 
                               facecolor='lightblue', alpha=0.3)
    hybrid_bg = patches.Rectangle((0.65, 0.55), 0.2, 0.3, linewidth=1, edgecolor='orange', 
                                 facecolor='lightyellow', alpha=0.3)
    
    ax.add_patch(early_bg)
    ax.add_patch(late_bg)
    ax.add_patch(hybrid_bg)
    
    # Add fusion type labels
    ax.text(early_x, 0.88, "Early Fusion", fontsize=14, ha='center', fontweight='bold')
    ax.text(late_x, 0.88, "Late Fusion", fontsize=14, ha='center', fontweight='bold')
    ax.text(hybrid_x, 0.88, "Hybrid Fusion", fontsize=14, ha='center', fontweight='bold')
    
    # Input boxes
    text_box = patches.Rectangle((0.3, start_y), box_width, box_height, 
                                linewidth=1, edgecolor='black', facecolor='lightblue', alpha=0.7)
    audio_box = patches.Rectangle((0.7, start_y), box_width, box_height, 
                                 linewidth=1, edgecolor='black', facecolor='lightblue', alpha=0.7)
    ax.add_patch(text_box)
    ax.add_patch(audio_box)
    ax.text(0.4, start_y + box_height/2, "Text Features", ha='center', va='center')
    ax.text(0.8, start_y + box_height/2, "Audio Features", ha='center', va='center')
    
    # Early fusion boxes
    ef_concat = patches.Rectangle((early_x - box_width/2, 0.8), box_width, box_height, 
                                 linewidth=1, edgecolor='black', facecolor='lightgreen', alpha=0.7)
    ef_process = patches.Rectangle((early_x - box_width/2, 0.7), box_width, box_height, 
                                  linewidth=1, edgecolor='black', facecolor='lightgreen', alpha=0.7)
    ef_output = patches.Rectangle((early_x - box_width/2, 0.6), box_width, box_height, 
                                 linewidth=1, edgecolor='black', facecolor='lightgreen', alpha=0.7)
    
    ax.add_patch(ef_concat)
    ax.add_patch(ef_process)
    ax.add_patch(ef_output)
    ax.text(early_x, 0.84, "Feature\nConcatenation", ha='center', va='center')
    ax.text(early_x, 0.74, "Joint\nProcessing", ha='center', va='center')
    ax.text(early_x, 0.64, "Prediction", ha='center', va='center')
    
    # Late fusion boxes
    lf_text = patches.Rectangle((late_x - box_width/2, 0.8), box_width, box_height, 
                               linewidth=1, edgecolor='black', facecolor='lightblue', alpha=0.7)
    lf_audio = patches.Rectangle((late_x - box_width/2, 0.7), box_width, box_height, 
                                linewidth=1, edgecolor='black', facecolor='lightblue', alpha=0.7)
    lf_combine = patches.Rectangle((late_x - box_width/2, 0.6), box_width, box_height, 
                                  linewidth=1, edgecolor='black', facecolor='lightblue', alpha=0.7)
    
    ax.add_patch(lf_text)
    ax.add_patch(lf_audio)
    ax.add_patch(lf_combine)
    ax.text(late_x, 0.84, "Text Model", ha='center', va='center')
    ax.text(late_x, 0.74, "Audio Model", ha='center', va='center')
    ax.text(late_x, 0.64, "Decision\nCombination", ha='center', va='center')
    
    # Hybrid fusion boxes
    hf_text = patches.Rectangle((hybrid_x - box_width/2, 0.8), box_width, box_height, 
                               linewidth=1, edgecolor='black', facecolor='lightyellow', alpha=0.7)
    hf_audio = patches.Rectangle((hybrid_x - box_width/2, 0.7), box_width, box_height, 
                                linewidth=1, edgecolor='black', facecolor='lightyellow', alpha=0.7)
    hf_combine = patches.Rectangle((hybrid_x - box_width/2, 0.6), box_width, box_height, 
                                  linewidth=1, edgecolor='black', facecolor='lightyellow', alpha=0.7)
    
    ax.add_patch(hf_text)
    ax.add_patch(hf_audio)
    ax.add_patch(hf_combine)
    ax.text(hybrid_x, 0.84, "Intermediate\nText Repr.", ha='center', va='center')
    ax.text(hybrid_x, 0.74, "Intermediate\nAudio Repr.", ha='center', va='center')
    ax.text(hybrid_x, 0.64, "Feature\nCombination", ha='center', va='center')
    
    # Final output boxes
    ef_final = patches.Rectangle((early_x - box_width/2, 0.45), box_width, box_height, 
                                linewidth=1, edgecolor='black', facecolor='lightcoral', alpha=0.7)
    lf_final = patches.Rectangle((late_x - box_width/2, 0.45), box_width, box_height, 
                                linewidth=1, edgecolor='black', facecolor='lightcoral', alpha=0.7)
    hf_final = patches.Rectangle((hybrid_x - box_width/2, 0.45), box_width, box_height, 
                                linewidth=1, edgecolor='black', facecolor='lightcoral', alpha=0.7)
    
    ax.add_patch(ef_final)
    ax.add_patch(lf_final)
    ax.add_patch(hf_final)
    ax.text(early_x, 0.49, "Emotion\nPrediction", ha='center', va='center')
    ax.text(late_x, 0.49, "Emotion\nPrediction", ha='center', va='center')
    ax.text(hybrid_x, 0.49, "Emotion\nPrediction", ha='center', va='center')
    
    # Draw connecting arrows
    # Early fusion arrows
    ax.arrow(0.4, 0.94, -0.1, -0.1, head_width=0.02, head_length=0.02, 
             fc='black', ec='black', linewidth=1)
    ax.arrow(0.8, 0.94, -0.5, -0.1, head_width=0.02, head_length=0.02, 
             fc='black', ec='black', linewidth=1)
    ax.arrow(early_x, 0.8, 0, -0.02, head_width=0.02, head_length=0.02, 
             fc='black', ec='black', linewidth=1)
    ax.arrow(early_x, 0.7, 0, -0.02, head_width=0.02, head_length=0.02, 
             fc='black', ec='black', linewidth=1)
    ax.arrow(early_x, 0.6, 0, -0.07, head_width=0.02, head_length=0.02, 
             fc='black', ec='black', linewidth=1)
    
    # Late fusion arrows
    ax.arrow(0.4, 0.94, 0.1, -0.1, head_width=0.02, head_length=0.02, 
             fc='black', ec='black', linewidth=1)
    ax.arrow(0.8, 0.94, -0.3, -0.1, head_width=0.02, head_length=0.02, 
             fc='black', ec='black', linewidth=1)
    ax.arrow(late_x, 0.8, 0, -0.02, head_width=0.02, head_length=0.02, 
             fc='black', ec='black', linewidth=1)
    ax.arrow(late_x, 0.7, 0, -0.02, head_width=0.02, head_length=0.02, 
             fc='black', ec='black', linewidth=1)
    ax.arrow(late_x, 0.6, 0, -0.07, head_width=0.02, head_length=0.02, 
             fc='black', ec='black', linewidth=1)
    
    # Hybrid fusion arrows
    ax.arrow(0.4, 0.94, 0.35, -0.1, head_width=0.02, head_length=0.02, 
             fc='black', ec='black', linewidth=1)
    ax.arrow(0.8, 0.94, -0.05, -0.1, head_width=0.02, head_length=0.02, 
             fc='black', ec='black', linewidth=1)
    ax.arrow(hybrid_x, 0.8, 0, -0.02, head_width=0.02, head_length=0.02, 
             fc='black', ec='black', linewidth=1)
    ax.arrow(hybrid_x, 0.7, 0, -0.02, head_width=0.02, head_length=0.02, 
             fc='black', ec='black', linewidth=1)
    ax.arrow(hybrid_x, 0.6, 0, -0.07, head_width=0.02, head_length=0.02, 
             fc='black', ec='black', linewidth=1)
    
    # Add summary observations
    ax.text(0.5, 0.34, "Fusion Strategy Comparison", fontsize=16, ha='center', fontweight='bold')
    ax.text(0.5, 0.28, "• Early fusion: Joint learning but computationally expensive", 
            fontsize=10, ha='center')
    ax.text(0.5, 0.24, "• Late fusion: Modular and flexible but limited interaction", 
            fontsize=10, ha='center')
    ax.text(0.5, 0.2, "• Hybrid fusion: Balance between joint learning and modularity", 
            fontsize=10, ha='center')
    ax.text(0.5, 0.16, "• Hybrid fusion achieved highest performance (91.74%)", 
            fontsize=10, ha='center', fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.title(titles[2], fontsize=18, pad=20)
    plt.tight_layout()
    plt.savefig("CS297-298-Xiangyi-Report/Figures/fusion_strategies_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

# 4. MFCC Pipeline
def create_mfcc_pipeline():
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Define the processing steps
    steps = [
        "Raw Audio\nSignal",
        "Pre-emphasis",
        "Framing\n(25ms windows)",
        "Hamming\nWindow",
        "Fast Fourier\nTransform",
        "Power\nSpectrum",
        "Mel Filter Bank\n(40 filters)",
        "Log\nCompression",
        "DCT",
        "MFCC\nFeatures",
        "Delta &\nDelta-Delta",
        "Z-score\nNormalization",
        "Final Feature\nVector"
    ]
    
    # Define node positions
    xs = np.linspace(0.05, 0.95, len(steps))
    y = 0.5
    
    # Add boxes and labels
    for i, (x, step) in enumerate(zip(xs, steps)):
        if i == 0:  # Input
            color = 'lightgreen'
        elif i == len(steps) - 1:  # Output
            color = 'lightcoral'
        elif i == 6:  # Mel Filter Bank
            color = 'lightblue'
        elif i == 9:  # MFCC Features
            color = 'lightyellow'
        else:
            color = 'white'
            
        box = patches.Rectangle((x - 0.03, y - 0.15), 0.06, 0.3, 
                              linewidth=1, edgecolor='black', facecolor=color, alpha=0.7)
        ax.add_patch(box)
        ax.text(x, y, step, ha='center', va='center', fontsize=9)
        
        # Add connecting arrows
        if i < len(steps) - 1:
            ax.arrow(x + 0.03, y, (xs[i+1] - x) - 0.06, 0, 
                    head_width=0.02, head_length=0.01, fc='black', ec='black', linewidth=1)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.title(titles[3], fontsize=18, pad=20)
    plt.tight_layout()
    plt.savefig("CS297-298-Xiangyi-Report/Figures/mfcc_pipeline.png", dpi=300, bbox_inches='tight')
    plt.close()

# 5. Experiment Framework
def create_experiment_framework():
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Define regions
    config_rect = patches.Rectangle((0.1, 0.7), 0.3, 0.2, linewidth=1, 
                                   edgecolor='black', facecolor='lightyellow', alpha=0.4)
    cloud_rect = patches.Rectangle((0.5, 0.7), 0.3, 0.2, linewidth=1, 
                                  edgecolor='black', facecolor='lightgreen', alpha=0.4)
    pipe_rect = patches.Rectangle((0.3, 0.3), 0.3, 0.3, linewidth=1, 
                                 edgecolor='black', facecolor='lightblue', alpha=0.4)
    
    ax.add_patch(config_rect)
    ax.add_patch(cloud_rect)
    ax.add_patch(pipe_rect)
    
    # Add region labels
    ax.text(0.25, 0.92, "Experiment Configuration", fontsize=14, ha='center', fontweight='bold')
    ax.text(0.65, 0.92, "Modal Cloud Infrastructure", fontsize=14, ha='center', fontweight='bold')
    ax.text(0.45, 0.62, "Experiment Pipeline", fontsize=14, ha='center', fontweight='bold')
    
    # Configuration components
    config_items = [
        {"label": "Experiment\nParameters", "position": (0.15, 0.85)},
        {"label": "Model\nArchitectures", "position": (0.25, 0.85)},
        {"label": "Audio Feature\nTypes", "position": (0.35, 0.85)},
        {"label": "Fusion\nStrategies", "position": (0.15, 0.75)},
        {"label": "Datasets", "position": (0.35, 0.75)}
    ]
    
    # Cloud infrastructure components
    cloud_items = [
        {"label": "Container\nInstance", "position": (0.55, 0.85)},
        {"label": "GPU Resources", "position": (0.75, 0.85)},
        {"label": "Data Storage", "position": (0.55, 0.75)},
        {"label": "Job Scheduler", "position": (0.75, 0.75)}
    ]
    
    # Pipeline components
    pipe_items = [
        {"label": "Data\nPreparation", "position": (0.35, 0.55)},
        {"label": "Model\nTraining", "position": (0.35, 0.45)},
        {"label": "Performance\nEvaluation", "position": (0.35, 0.35)},
        {"label": "Results\nStorage", "position": (0.35, 0.25)},
        {"label": "Analysis &\nVisualization", "position": (0.65, 0.25)}
    ]
    
    # Draw all components
    for items in [config_items, cloud_items, pipe_items]:
        for item in items:
            circle = patches.Circle(item["position"], 0.04, 
                                   linewidth=1, edgecolor='black', facecolor='white', alpha=0.7)
            ax.add_patch(circle)
            ax.text(item["position"][0], item["position"][1], item["label"], 
                   ha='center', va='center', fontsize=9)
    
    # Draw connecting arrows
    # Config to Cloud
    ax.arrow(0.15, 0.81, 0.6, -0.06, head_width=0.02, head_length=0.02, 
            fc='black', ec='black', linewidth=1)
    ax.arrow(0.25, 0.81, 0.3, 0, head_width=0.02, head_length=0.02, 
            fc='black', ec='black', linewidth=1)
    ax.arrow(0.35, 0.81, 0.2, 0, head_width=0.02, head_length=0.02, 
            fc='black', ec='black', linewidth=1)
    ax.arrow(0.15, 0.71, 0.4, 0, head_width=0.02, head_length=0.02, 
            fc='black', ec='black', linewidth=1)
    ax.arrow(0.35, 0.71, 0.2, 0, head_width=0.02, head_length=0.02, 
            fc='black', ec='black', linewidth=1)
    
    # Cloud interconnections
    ax.arrow(0.75, 0.81, -0.16, 0, head_width=0.02, head_length=0.02, 
            fc='black', ec='black', linewidth=1)
    ax.arrow(0.75, 0.71, -0.16, 0.14, head_width=0.02, head_length=0.02, 
            fc='black', ec='black', linewidth=1)
    ax.arrow(0.55, 0.71, 0, -0.16, head_width=0.02, head_length=0.02, 
            fc='black', ec='black', linewidth=1)
    
    # Pipeline flow
    ax.arrow(0.35, 0.51, 0, -0.02, head_width=0.02, head_length=0.02, 
            fc='black', ec='black', linewidth=1)
    ax.arrow(0.35, 0.41, 0, -0.02, head_width=0.02, head_length=0.02, 
            fc='black', ec='black', linewidth=1)
    ax.arrow(0.35, 0.31, 0, -0.02, head_width=0.02, head_length=0.02, 
            fc='black', ec='black', linewidth=1)
    ax.arrow(0.39, 0.25, 0.22, 0, head_width=0.02, head_length=0.02, 
            fc='black', ec='black', linewidth=1)
    
    # Key insights
    ax.text(0.5, 0.15, "Key Insights:", fontsize=12, ha='center', fontweight='bold')
    ax.text(0.5, 0.11, "• 323 experiments conducted across model and feature combinations", 
            fontsize=10, ha='center')
    ax.text(0.5, 0.08, "• Parallel execution on Modal cloud reduced total runtime by 20x", 
            fontsize=10, ha='center')
    ax.text(0.5, 0.05, "• Results stored in structured format enabling automated analysis", 
            fontsize=10, ha='center')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.title(titles[4], fontsize=18, pad=20)
    plt.tight_layout()
    plt.savefig("CS297-298-Xiangyi-Report/Figures/experiment_framework.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_hybrid_fusion_architecture():
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define boxes dimensions and positions
    box_width, box_height = 0.2, 0.08
    text_x, audio_x = 0.3, 0.7
    
    # Input level
    text_input = patches.Rectangle((text_x - box_width/2, 0.9), box_width, box_height, 
                               linewidth=1, edgecolor='black', facecolor='lightblue', alpha=0.7)
    audio_input = patches.Rectangle((audio_x - box_width/2, 0.9), box_width, box_height, 
                                linewidth=1, edgecolor='black', facecolor='lightblue', alpha=0.7)
    ax.add_patch(text_input)
    ax.add_patch(audio_input)
    ax.text(text_x, 0.94, "Text Input", ha='center', va='center')
    ax.text(audio_x, 0.94, "Audio Input", ha='center', va='center')
    
    # Early processing layers - Modality specific
    text_process1 = patches.Rectangle((text_x - box_width/2, 0.75), box_width, box_height, 
                                  linewidth=1, edgecolor='black', facecolor='lightblue', alpha=0.7)
    audio_process1 = patches.Rectangle((audio_x - box_width/2, 0.75), box_width, box_height, 
                                   linewidth=1, edgecolor='black', facecolor='lightblue', alpha=0.7)
    ax.add_patch(text_process1)
    ax.add_patch(audio_process1)
    ax.text(text_x, 0.79, "Transformer\nLayers 1-8", ha='center', va='center', fontsize=9)
    ax.text(audio_x, 0.79, "CNN/LSTM\nLayers", ha='center', va='center', fontsize=9)
    
    # Intermediate representations
    text_repr = patches.Rectangle((text_x - box_width/2, 0.6), box_width, box_height, 
                             linewidth=1, edgecolor='black', facecolor='lightgreen', alpha=0.7)
    audio_repr = patches.Rectangle((audio_x - box_width/2, 0.6), box_width, box_height, 
                              linewidth=1, edgecolor='black', facecolor='lightgreen', alpha=0.7)
    ax.add_patch(text_repr)
    ax.add_patch(audio_repr)
    ax.text(text_x, 0.64, "Intermediate\nText Representation", ha='center', va='center', fontsize=9)
    ax.text(audio_x, 0.64, "Intermediate\nAudio Representation", ha='center', va='center', fontsize=9)
    
    # Feature projection
    text_proj = patches.Rectangle((text_x - box_width/2, 0.45), box_width, box_height, 
                             linewidth=1, edgecolor='black', facecolor='lightgreen', alpha=0.7)
    audio_proj = patches.Rectangle((audio_x - box_width/2, 0.45), box_width, box_height, 
                              linewidth=1, edgecolor='black', facecolor='lightgreen', alpha=0.7)
    ax.add_patch(text_proj)
    ax.add_patch(audio_proj)
    ax.text(text_x, 0.49, "Dense(256)", ha='center', va='center', fontsize=9)
    ax.text(audio_x, 0.49, "Dense(128)", ha='center', va='center', fontsize=9)
    
    # Concatenation
    concat_box = patches.Rectangle((0.4, 0.3), 0.2, 0.08, 
                              linewidth=1, edgecolor='black', facecolor='gold', alpha=0.7)
    ax.add_patch(concat_box)
    ax.text(0.5, 0.34, "Concatenation", ha='center', va='center')
    
    # Joint processing
    joint1 = patches.Rectangle((0.4, 0.2), 0.2, 0.05, 
                          linewidth=1, edgecolor='black', facecolor='lightyellow', alpha=0.7)
    joint2 = patches.Rectangle((0.4, 0.13), 0.2, 0.05, 
                          linewidth=1, edgecolor='black', facecolor='lightyellow', alpha=0.7)
    ax.add_patch(joint1)
    ax.add_patch(joint2)
    ax.text(0.5, 0.225, "Dense(384) + ReLU", ha='center', va='center', fontsize=9)
    ax.text(0.5, 0.155, "Dense(192) + ReLU", ha='center', va='center', fontsize=9)
    
    # Output
    output = patches.Rectangle((0.4, 0.05), 0.2, 0.05, 
                          linewidth=1, edgecolor='black', facecolor='lightcoral', alpha=0.7)
    ax.add_patch(output)
    ax.text(0.5, 0.075, "Emotion Prediction", ha='center', va='center')
    
    # Connecting arrows
    # Text path
    ax.arrow(text_x, 0.9, 0, -0.07, head_width=0.02, head_length=0.02, fc='black', ec='black', linewidth=1)
    ax.arrow(text_x, 0.75, 0, -0.07, head_width=0.02, head_length=0.02, fc='black', ec='black', linewidth=1)
    ax.arrow(text_x, 0.6, 0, -0.07, head_width=0.02, head_length=0.02, fc='black', ec='black', linewidth=1)
    ax.arrow(text_x, 0.45, 0.14, -0.11, head_width=0.02, head_length=0.02, fc='black', ec='black', linewidth=1)
    
    # Audio path
    ax.arrow(audio_x, 0.9, 0, -0.07, head_width=0.02, head_length=0.02, fc='black', ec='black', linewidth=1)
    ax.arrow(audio_x, 0.75, 0, -0.07, head_width=0.02, head_length=0.02, fc='black', ec='black', linewidth=1)
    ax.arrow(audio_x, 0.6, 0, -0.07, head_width=0.02, head_length=0.02, fc='black', ec='black', linewidth=1)
    ax.arrow(audio_x, 0.45, -0.14, -0.11, head_width=0.02, head_length=0.02, fc='black', ec='black', linewidth=1)
    
    # Joint path
    ax.arrow(0.5, 0.3, 0, -0.05, head_width=0.02, head_length=0.02, fc='black', ec='black', linewidth=1)
    ax.arrow(0.5, 0.2, 0, -0.02, head_width=0.02, head_length=0.02, fc='black', ec='black', linewidth=1)
    ax.arrow(0.5, 0.13, 0, -0.03, head_width=0.02, head_length=0.02, fc='black', ec='black', linewidth=1)
    
    # Title and labels
    ax.text(0.5, 0.97, "Hybrid Fusion Architecture", fontsize=14, ha='center', fontweight='bold')
    ax.text(0.2, 0.82, "Modality-Specific\nProcessing", fontsize=10, ha='center', fontweight='bold')
    ax.text(0.2, 0.52, "Partial Feature\nExtraction", fontsize=10, ha='center', fontweight='bold')
    ax.text(0.8, 0.25, "Joint Processing\nWith Shared Layers", fontsize=10, ha='center', fontweight='bold')
    
    # Add background shades to separate processing stages
    modality_bg = patches.Rectangle((0.05, 0.44), 0.9, 0.55, linewidth=1, edgecolor='black', 
                                   facecolor='lightblue', alpha=0.1)
    fusion_bg = patches.Rectangle((0.05, 0.04), 0.9, 0.35, linewidth=1, edgecolor='black', 
                                 facecolor='lightyellow', alpha=0.1)
    ax.add_patch(modality_bg)
    ax.add_patch(fusion_bg)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.title("Detailed Architecture of Hybrid Fusion Approach", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig("CS297-298-Xiangyi-Report/Figures/hybrid_fusion_detailed.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Created hybrid fusion architecture diagram")

def create_late_fusion_architecture():
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define boxes dimensions and positions
    box_width, box_height = 0.2, 0.08
    text_x, audio_x = 0.3, 0.7
    
    # Input level
    text_input = patches.Rectangle((text_x - box_width/2, 0.9), box_width, box_height, 
                               linewidth=1, edgecolor='black', facecolor='lightblue', alpha=0.7)
    audio_input = patches.Rectangle((audio_x - box_width/2, 0.9), box_width, box_height, 
                                linewidth=1, edgecolor='black', facecolor='lightblue', alpha=0.7)
    ax.add_patch(text_input)
    ax.add_patch(audio_input)
    ax.text(text_x, 0.94, "Text Input", ha='center', va='center')
    ax.text(audio_x, 0.94, "Audio Input", ha='center', va='center')
    
    # Text processing pipeline - Left side
    text_process1 = patches.Rectangle((text_x - box_width/2, 0.75), box_width, box_height, 
                                  linewidth=1, edgecolor='black', facecolor='lightblue', alpha=0.7)
    text_process2 = patches.Rectangle((text_x - box_width/2, 0.6), box_width, box_height, 
                                  linewidth=1, edgecolor='black', facecolor='lightblue', alpha=0.7)
    text_process3 = patches.Rectangle((text_x - box_width/2, 0.45), box_width, box_height, 
                                  linewidth=1, edgecolor='black', facecolor='lightblue', alpha=0.7)
    ax.add_patch(text_process1)
    ax.add_patch(text_process2)
    ax.add_patch(text_process3)
    ax.text(text_x, 0.79, "Text\nTokenization", ha='center', va='center', fontsize=9)
    ax.text(text_x, 0.64, "Transformer\nModel", ha='center', va='center', fontsize=9)
    ax.text(text_x, 0.49, "Text Emotion\nPrediction", ha='center', va='center', fontsize=9)
    
    # Audio processing pipeline - Right side
    audio_process1 = patches.Rectangle((audio_x - box_width/2, 0.75), box_width, box_height, 
                                   linewidth=1, edgecolor='black', facecolor='lightblue', alpha=0.7)
    audio_process2 = patches.Rectangle((audio_x - box_width/2, 0.6), box_width, box_height, 
                                   linewidth=1, edgecolor='black', facecolor='lightblue', alpha=0.7)
    audio_process3 = patches.Rectangle((audio_x - box_width/2, 0.45), box_width, box_height, 
                                   linewidth=1, edgecolor='black', facecolor='lightblue', alpha=0.7)
    ax.add_patch(audio_process1)
    ax.add_patch(audio_process2)
    ax.add_patch(audio_process3)
    ax.text(audio_x, 0.79, "Audio Feature\nExtraction", ha='center', va='center', fontsize=9)
    ax.text(audio_x, 0.64, "CNN/LSTM\nModel", ha='center', va='center', fontsize=9)
    ax.text(audio_x, 0.49, "Audio Emotion\nPrediction", ha='center', va='center', fontsize=9)
    
    # Fusion component - Bottom
    fusion_box = patches.Rectangle((0.4, 0.3), 0.2, 0.08, 
                              linewidth=1, edgecolor='black', facecolor='gold', alpha=0.7)
    ax.add_patch(fusion_box)
    ax.text(0.5, 0.34, "Decision Fusion\n(Weighted Averaging)", ha='center', va='center', fontsize=9)
    
    # Output
    output = patches.Rectangle((0.4, 0.15), 0.2, 0.08, 
                          linewidth=1, edgecolor='black', facecolor='lightcoral', alpha=0.7)
    ax.add_patch(output)
    ax.text(0.5, 0.19, "Final Emotion\nPrediction", ha='center', va='center', fontsize=9)
    
    # Connecting arrows
    # Text path
    ax.arrow(text_x, 0.9, 0, -0.07, head_width=0.02, head_length=0.02, fc='black', ec='black', linewidth=1)
    ax.arrow(text_x, 0.75, 0, -0.07, head_width=0.02, head_length=0.02, fc='black', ec='black', linewidth=1)
    ax.arrow(text_x, 0.6, 0, -0.07, head_width=0.02, head_length=0.02, fc='black', ec='black', linewidth=1)
    ax.arrow(text_x, 0.45, 0.12, -0.08, head_width=0.02, head_length=0.02, fc='black', ec='black', linewidth=1)
    
    # Audio path
    ax.arrow(audio_x, 0.9, 0, -0.07, head_width=0.02, head_length=0.02, fc='black', ec='black', linewidth=1)
    ax.arrow(audio_x, 0.75, 0, -0.07, head_width=0.02, head_length=0.02, fc='black', ec='black', linewidth=1)
    ax.arrow(audio_x, 0.6, 0, -0.07, head_width=0.02, head_length=0.02, fc='black', ec='black', linewidth=1)
    ax.arrow(audio_x, 0.45, -0.12, -0.08, head_width=0.02, head_length=0.02, fc='black', ec='black', linewidth=1)
    
    # Output path
    ax.arrow(0.5, 0.3, 0, -0.07, head_width=0.02, head_length=0.02, fc='black', ec='black', linewidth=1)
    
    # Add explanatory text labels
    ax.text(0.5, 0.97, "Late Fusion Architecture", fontsize=14, ha='center', fontweight='bold')
    ax.text(0.2, 0.38, "Independent Modality\nProcessing", fontsize=10, ha='center', fontweight='bold')
    ax.text(0.8, 0.25, "Decision-level\nFusion", fontsize=10, ha='center', fontweight='bold')
    
    # Add emphasis on independent processing
    indep_text_box = patches.Rectangle((0.1, 0.4), 0.4, 0.6, linewidth=2, 
                                   edgecolor='blue', facecolor='none', linestyle='--')
    indep_audio_box = patches.Rectangle((0.5, 0.4), 0.4, 0.6, linewidth=2, 
                                    edgecolor='green', facecolor='none', linestyle='--')
    ax.add_patch(indep_text_box)
    ax.add_patch(indep_audio_box)
    ax.text(0.3, 0.35, "Text Pipeline", fontsize=10, ha='center', color='blue')
    ax.text(0.7, 0.35, "Audio Pipeline", fontsize=10, ha='center', color='green')
    
    # Add weights visualization for fusion
    weight_text = patches.Rectangle((0.35, 0.25), 0.1, 0.03, linewidth=1, 
                               edgecolor='black', facecolor='blue', alpha=0.4)
    weight_audio = patches.Rectangle((0.55, 0.25), 0.1, 0.03, linewidth=1, 
                                edgecolor='black', facecolor='green', alpha=0.4)
    ax.add_patch(weight_text)
    ax.add_patch(weight_audio)
    ax.text(0.4, 0.265, "w₁", fontsize=8, ha='center')
    ax.text(0.6, 0.265, "w₂", fontsize=8, ha='center')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.title("Detailed Architecture of Late Fusion Approach", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig("CS297-298-Xiangyi-Report/Figures/late_fusion_detailed_proper.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Created detailed late fusion architecture diagram")

# Generate all diagrams
def main():
    print("Generating architecture diagrams using matplotlib...")
    
    create_system_architecture()
    print("✓ Created system architecture diagram")
    
    create_text_model_architecture()
    print("✓ Created text model architecture diagram")
    
    create_fusion_strategies()
    print("✓ Created fusion strategies comparison")
    
    create_mfcc_pipeline()
    print("✓ Created MFCC pipeline diagram")
    
    create_experiment_framework()
    print("✓ Created experiment framework diagram")
    
    create_hybrid_fusion_architecture()
    print("✓ Created hybrid fusion architecture diagram")
    
    create_late_fusion_architecture()
    print("✓ Created detailed late fusion architecture diagram")
    
    # Generate improved system architecture
    generate_improved_system_architecture()
    
    # Generate LaTeX code for inclusion
    latex_code = '% Add these diagram figures to your LaTeX document\n\n'
    diagrams = [
        'system_architecture',
        'text_model_architecture',
        'fusion_strategies_comparison',
        'mfcc_pipeline',
        'experiment_framework',
        'hybrid_fusion_detailed',
        'late_fusion_detailed_proper'
    ]
    
    for i, diagram in enumerate(diagrams):
        title = titles[i] if i < len(titles) else f"Diagram {i+1}"
        
        latex_code += f"""\\begin{{figure}}[h]
    \\centering
    \\includegraphics[width=0.9\\linewidth]{{Figures/{diagram}.png}}
    \\caption{{{title}}}
    \\label{{fig:{diagram}}}
\\end{{figure}}\n\n"""
    
    with open('python_diagrams.tex', 'w') as f:
        f.write(latex_code)
    
    print("\nAll diagrams successfully generated in CS297-298-Xiangyi-Report/Figures/")
    print("LaTeX code for including the diagrams has been written to python_diagrams.tex")

if __name__ == "__main__":
    main() 