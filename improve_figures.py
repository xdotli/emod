#!/usr/bin/env python3
"""
Script to improve figures in the CS297-298-Xiangyi-Report
Focuses on increasing readability, particularly the placement of arrows
Uses matplotlib and PIL to modify images
"""

import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import re
from matplotlib.colors import LinearSegmentedColormap

def list_figures(figures_dir):
    """List all figure files in the directory"""
    figure_extensions = ('.png', '.jpg', '.jpeg', '.pdf')
    figures = []
    
    for file in os.listdir(figures_dir):
        if file.lower().endswith(figure_extensions):
            figures.append(os.path.join(figures_dir, file))
    
    print(f"Found {len(figures)} figures in {figures_dir}")
    return figures

def analyze_figure(image_path):
    """Analyze a figure for potential improvements"""
    try:
        # Open image
        img = Image.open(image_path)
        width, height = img.size
        
        print(f"Analyzing {os.path.basename(image_path)}: {width}x{height} pixels")
        
        # Convert to numpy array for analysis
        img_array = np.array(img)
        
        # Check if image has arrows (simplistic detection)
        has_arrows = False
        
        # For PNG images with transparency
        if img.mode == 'RGBA':
            # Look for narrow elements that could be arrow lines
            gray = np.mean(img_array[:,:,:3], axis=2)
            edges = np.abs(np.diff(gray, axis=1))
            narrow_elements = np.sum(edges > 50)
            has_arrows = narrow_elements > (width * height * 0.005)
        
        # For all images, check for arrow-like shapes (very simplistic)
        # This is a heuristic that could be improved
        if not has_arrows:
            # Convert to grayscale
            if len(img_array.shape) == 3:
                gray = np.mean(img_array[:,:,:3], axis=2)
            else:
                gray = img_array
                
            # Look for diagonal patterns that might be arrows
            # This is a very simple heuristic
            diag_sum = np.sum(np.abs(np.diff(gray, axis=0)) * np.abs(np.diff(gray, axis=1)[:, :-1]))
            has_arrows = diag_sum > (width * height * 50)
        
        return {
            'path': image_path,
            'width': width,
            'height': height,
            'has_arrows': has_arrows
        }
    except Exception as e:
        print(f"Error analyzing {image_path}: {e}")
        return None

def improve_arrow_visibility(image_path, output_dir):
    """Improve arrow visibility in a figure"""
    try:
        # Get filename without extension
        base_name = os.path.basename(image_path)
        file_name, file_ext = os.path.splitext(base_name)
        
        # Create output path
        output_path = os.path.join(output_dir, f"{file_name}_improved{file_ext}")
        
        # Open image
        img = Image.open(image_path)
        
        # Detect if image is diagram-like vs. photographic
        is_diagram = is_diagram_like(np.array(img))
        
        if is_diagram:
            # For diagram-like images - enhance arrows with contour
            improved_img = enhance_diagram_arrows(img)
        else:
            # For photo-like images - standard enhancement
            improved_img = enhance_photo_arrows(img)
        
        # Save improved image
        improved_img.save(output_path)
        print(f"Improved figure saved to {output_path}")
        
        return output_path
    except Exception as e:
        print(f"Error improving {image_path}: {e}")
        return None

def is_diagram_like(img_array):
    """Detect if an image is more like a diagram or a photograph"""
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        gray = np.mean(img_array[:,:,:3], axis=2)
    else:
        gray = img_array
    
    # Calculate histogram
    hist, _ = np.histogram(gray, bins=50)
    
    # Normalize histogram
    hist = hist / np.sum(hist)
    
    # Diagrams typically have fewer distinct colors and more uniform distributions
    # Calculate entropy of histogram as a measure
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    
    # Calculate number of distinct colors (approximation)
    distinct_colors = np.sum(hist > 0.01)
    
    # Diagrams often have large areas of the same color
    uniform_areas = np.mean(np.abs(np.diff(gray, axis=1)) < 5)
    
    # Combine metrics
    diagram_score = (distinct_colors < 10) + (entropy < 3.5) + (uniform_areas > 0.7)
    
    return diagram_score >= 2  # If at least 2 criteria are met, consider it a diagram

def enhance_diagram_arrows(img):
    """Enhance arrows in diagram-like images"""
    # Convert to RGBA if not already
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    width, height = img.size
    
    # Create a new image with the same size
    enhanced = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    
    # Get drawable interface
    draw = ImageDraw.Draw(enhanced)
    
    # Find potential arrow lines
    img_array = np.array(img)
    
    # Convert to grayscale
    gray = np.mean(img_array[:,:,:3], axis=2)
    
    # Find edges
    edges_h = np.abs(np.diff(gray, axis=1))
    edges_v = np.abs(np.diff(gray, axis=0))
    
    # Combine to create edge map
    edge_map = np.zeros((height, width))
    edge_map[:, 1:] += edges_h
    edge_map[1:, :] += edges_v[:, :]
    
    # Threshold to find arrow lines
    potential_arrows = edge_map > np.percentile(edge_map, 95)
    
    # Create a mask of areas that might be arrows
    arrow_mask = potential_arrows.astype(np.uint8) * 255
    
    # Create a new image for arrow improvements
    arrow_img = Image.fromarray(arrow_mask)
    
    # Dilate the arrow mask to make arrows thicker
    from PIL import ImageFilter
    arrow_img = arrow_img.filter(ImageFilter.MaxFilter(3))
    
    # Draw arrows with enhanced visibility
    # Overlay the original image
    enhanced = Image.alpha_composite(enhanced, img)
    
    # Overlay the enhanced arrows with semi-transparency
    arrow_rgba = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    arrow_rgba.paste((255, 165, 0, 128), (0, 0), arrow_img)  # Orange with 50% opacity
    
    enhanced = Image.alpha_composite(enhanced, arrow_rgba)
    
    return enhanced

def enhance_photo_arrows(img):
    """Enhance arrows in photo-like images"""
    # Basic enhancement for photos - increase contrast and saturation
    from PIL import ImageEnhance
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.3)
    
    # Enhance color
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.2)
    
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.5)
    
    return img

def improve_figures_in_latex(latex_file, figures_dir, output_dir):
    """Replace figure references in LaTeX file with improved figures"""
    with open(latex_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all figure inclusion commands
    figure_pattern = r'\\includegraphics(\[.*?\])?\{([^}]+)\}'
    
    # Get matches
    figure_matches = re.finditer(figure_pattern, content)
    
    # Track replacements
    replacements = []
    
    for match in figure_matches:
        original_command = match.group(0)
        options = match.group(1) or ''  # Could be None
        path = match.group(2)
        
        # Extract filename
        file_name = os.path.basename(path)
        file_base, file_ext = os.path.splitext(file_name)
        
        # Check if it's in the improved figures
        improved_path = os.path.join(output_dir, f"{file_base}_improved{file_ext}")
        
        if os.path.exists(improved_path):
            # Replace with improved version
            new_path = os.path.join("Figures_Improved", f"{file_base}_improved{file_ext}")
            new_command = f"\\includegraphics{options}{{{new_path}}}"
            replacements.append((original_command, new_command))
    
    # Apply replacements
    for old, new in replacements:
        content = content.replace(old, new)
    
    # Save updated LaTeX file
    updated_latex = os.path.join(os.path.dirname(latex_file), "main_improved.tex")
    with open(updated_latex, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Updated LaTeX file saved to {updated_latex}")
    print(f"Updated {len(replacements)} figure references")
    
    return updated_latex

def generate_better_arrow_layout(fig_width, fig_height, arrows_data, output_path):
    """Generate a figure with better arrow layout based on data"""
    # Create a figure with the specified dimensions
    fig, ax = plt.subplots(figsize=(fig_width/100, fig_height/100), dpi=100)
    
    # Set background color
    ax.set_facecolor('#f8f8f8')
    
    # Create custom arrow style
    arrow_style = patches.ArrowStyle("Fancy", head_length=10, head_width=7, tail_width=2)
    
    # Create colormap for arrows
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Draw arrows with better placement
    for i, arrow in enumerate(arrows_data):
        color = colors[i % len(colors)]
        
        # Create curved arrow path with shadow for depth
        # Add a slight shadow first
        shadow = patches.FancyArrowPatch(
            arrow['start'], 
            arrow['end'],
            connectionstyle=f"arc3,rad={arrow.get('curve', 0.1)}",
            arrowstyle=arrow_style,
            alpha=0.3,
            color='gray',
            linewidth=2,
            mutation_scale=15
        )
        
        # Add the main arrow
        arr = patches.FancyArrowPatch(
            arrow['start'], 
            arrow['end'],
            connectionstyle=f"arc3,rad={arrow.get('curve', 0.1)}",
            arrowstyle=arrow_style,
            alpha=0.8,
            color=color,
            linewidth=2,
            mutation_scale=15
        )
        
        # Add shadow and arrow to the plot
        ax.add_patch(shadow)
        ax.add_patch(arr)
        
        # Add label if provided
        if 'label' in arrow:
            # Position label along the arrow
            # Calculate midpoint with offset
            midpoint_x = (arrow['start'][0] + arrow['end'][0]) / 2
            midpoint_y = (arrow['start'][1] + arrow['end'][1]) / 2
            
            # Add a small background for the text
            text = ax.text(
                midpoint_x, midpoint_y,
                arrow['label'],
                ha='center',
                va='center',
                fontsize=10,
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3),
                zorder=10
            )
    
    # Set limits and remove axes
    ax.set_xlim(0, fig_width)
    ax.set_ylim(0, fig_height)
    ax.axis('off')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generated improved arrow layout figure: {output_path}")
    return output_path

def improve_fusion_diagram():
    """Improve the fusion strategies comparison diagram"""
    # Create a diagram with better arrows for fusion strategies
    output_path = "CS297-298-Xiangyi-Report/Figures/fusion_strategies_improved.png"
    
    # Define the arrows data
    arrows_data = [
        {
            'start': (100, 250), 'end': (300, 150),
            'curve': 0.3, 'label': 'Early Fusion'
        },
        {
            'start': (100, 250), 'end': (300, 250),
            'curve': 0.0, 'label': 'Hybrid Fusion'
        },
        {
            'start': (100, 250), 'end': (300, 350),
            'curve': -0.3, 'label': 'Late Fusion'
        }
    ]
    
    # Generate the improved diagram
    generate_better_arrow_layout(400, 500, arrows_data, output_path)
    
    return output_path

def improve_system_architecture():
    """Improve the system architecture diagram"""
    # Create a diagram with better arrows for system architecture
    output_path = "CS297-298-Xiangyi-Report/Figures/system_architecture_improved.png"
    
    # Define the arrows data
    arrows_data = [
        {
            'start': (50, 150), 'end': (200, 150),
            'curve': 0.0, 'label': 'Text'
        },
        {
            'start': (50, 350), 'end': (200, 350),
            'curve': 0.0, 'label': 'Audio'
        },
        {
            'start': (250, 150), 'end': (400, 250),
            'curve': 0.2, 'label': 'AVD Values'
        },
        {
            'start': (250, 350), 'end': (400, 250),
            'curve': -0.2, 'label': 'AVD Values'
        },
        {
            'start': (450, 250), 'end': (600, 250),
            'curve': 0.0, 'label': 'Categories'
        }
    ]
    
    # Generate the improved diagram
    generate_better_arrow_layout(650, 500, arrows_data, output_path)
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Improve figures in the CS297-298-Xiangyi-Report")
    parser.add_argument('--figures-dir', type=str, default='CS297-298-Xiangyi-Report/Figures',
                        help='Directory containing figures')
    parser.add_argument('--output-dir', type=str, default='CS297-298-Xiangyi-Report/Figures_Improved',
                        help='Directory to save improved figures')
    parser.add_argument('--latex-file', type=str, default='CS297-298-Xiangyi-Report/main.tex',
                        help='LaTeX file to update with improved figures')
    parser.add_argument('--regenerate-diagrams', action='store_true',
                        help='Regenerate key diagrams with better arrow layouts')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # List figures
    figures = list_figures(args.figures_dir)
    
    # Analyze and improve figures
    improved_figures = []
    for figure in figures:
        analysis = analyze_figure(figure)
        if analysis and analysis['has_arrows']:
            print(f"Improving arrows in {os.path.basename(figure)}")
            improved = improve_arrow_visibility(figure, args.output_dir)
            if improved:
                improved_figures.append(improved)
        else:
            print(f"No arrows detected in {os.path.basename(figure)}")
    
    # Regenerate key diagrams if requested
    if args.regenerate_diagrams:
        print("Regenerating key diagrams with better arrow layouts")
        improve_fusion_diagram()
        improve_system_architecture()
    
    # Update LaTeX file if improved figures exist
    if improved_figures:
        latex_updated = improve_figures_in_latex(args.latex_file, args.figures_dir, args.output_dir)
        print(f"LaTeX file updated: {latex_updated}")
    else:
        print("No figures were improved. LaTeX file not updated.")

if __name__ == "__main__":
    main() 