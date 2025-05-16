#!/usr/bin/env python3
"""
Script to shorten the CS297-298-Xiangyi-Report by ~50%
Focuses on preserving the experiment and results sections while reducing other sections
"""

import os
import re
import argparse
from pathlib import Path

def read_tex_file(file_path):
    """Read a LaTeX file and return its content"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

def write_tex_file(file_path, content):
    """Write content to a LaTeX file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Wrote shortened content to {file_path}")

def identify_sections(content):
    """Identify sections in the LaTeX file"""
    # Find all section commands in the document
    section_pattern = r'\\section\{([^}]+)\}'
    subsection_pattern = r'\\subsection\{([^}]+)\}'
    
    sections = re.findall(section_pattern, content)
    subsections = re.findall(subsection_pattern, content)
    
    print(f"Identified {len(sections)} sections and {len(subsections)} subsections")
    print("Sections:")
    for i, section in enumerate(sections):
        print(f"  {i+1}. {section}")
    
    return sections, subsections

def split_by_sections(content):
    """Split content by sections"""
    # Split by section commands
    section_pattern = r'\\section\{([^}]+)\}'
    
    # Find all section commands and their positions
    section_matches = list(re.finditer(section_pattern, content))
    
    sections = []
    # Process each section
    for i, match in enumerate(section_matches):
        section_name = match.group(1)
        start_pos = match.start()
        
        # End position is the start of the next section or the end of the document
        if i < len(section_matches) - 1:
            end_pos = section_matches[i+1].start()
        else:
            end_pos = len(content)
        
        # Extract section content
        section_content = content[start_pos:end_pos]
        sections.append({
            'name': section_name,
            'content': section_content
        })
    
    return sections

def count_words(text):
    """Count words in text after removing LaTeX commands"""
    # Remove LaTeX commands and environments
    text = re.sub(r'\\[a-zA-Z]+(\{[^}]*\})*', ' ', text)
    text = re.sub(r'\\\[.*?\\\]', ' ', text, flags=re.DOTALL)
    text = re.sub(r'\\\(.*?\\\)', ' ', text, flags=re.DOTALL)
    text = re.sub(r'\\begin\{[^}]+\}.*?\\end\{[^}]+\}', ' ', text, flags=re.DOTALL)
    
    # Remove comments
    text = re.sub(r'%.*?$', '', text, flags=re.MULTILINE)
    
    # Count words
    words = text.split()
    return len(words)

def shorten_section(section_content, reduction_factor=0.5):
    """Shorten a section by removing paragraphs"""
    # Split by paragraphs (empty lines)
    paragraphs = re.split(r'\n\s*\n', section_content)
    
    # Calculate how many paragraphs to keep
    keep_count = max(1, int(len(paragraphs) * (1 - reduction_factor)))
    
    # Keep the first paragraph and select others to meet the reduction factor
    # Prioritize keeping paragraphs with key content indicators
    first_paragraph = paragraphs[0]
    remaining = paragraphs[1:]
    
    # Score paragraphs for importance
    scored_paragraphs = []
    for i, para in enumerate(remaining):
        score = 0
        # Higher score for paragraphs with figures, tables, citations
        if re.search(r'\\(figure|table|cite|label|ref)', para):
            score += 5
        # Higher score for paragraphs with equations
        if re.search(r'\\(eq|begin\{equation\})', para):
            score += 3
        # Higher score for paragraphs with specific keywords
        if re.search(r'(experiment|result|performance|accuracy|contribution|conclusion)', para, re.IGNORECASE):
            score += 2
        # Penalize long paragraphs slightly
        score -= (count_words(para) / 100)
        
        scored_paragraphs.append((i, para, score))
    
    # Sort paragraphs by score (descending)
    scored_paragraphs.sort(key=lambda x: x[2], reverse=True)
    
    # Select top paragraphs to keep
    keep_indices = [p[0] for p in scored_paragraphs[:keep_count-1]]
    keep_indices.sort()  # Restore original order
    
    # Reconstruct shortened section
    shortened_paragraphs = [first_paragraph] + [remaining[i] for i in keep_indices]
    shortened_content = '\n\n'.join(shortened_paragraphs)
    
    return shortened_content

def update_figure_captions(content):
    """Update figure captions to remove paragraphs"""
    # Find all figure environments
    figure_pattern = r'\\begin\{figure\}(.*?)\\end\{figure\}'
    figures = re.findall(figure_pattern, content, re.DOTALL)
    
    # Process each figure
    for figure in figures:
        # Find caption
        caption_pattern = r'\\caption\{(.*?)\}'
        caption_match = re.search(caption_pattern, figure, re.DOTALL)
        
        if caption_match:
            original_caption = caption_match.group(1)
            
            # Shorten caption - keep only the first sentence
            shortened_caption = re.split(r'\.', original_caption, 1)[0] + '.'
            
            # Replace in content
            content = content.replace(
                f"\\caption{{{original_caption}}}", 
                f"\\caption{{{shortened_caption}}}"
            )
    
    return content

def update_table_captions(content):
    """Update table captions to remove paragraphs"""
    # Find all table environments
    table_pattern = r'\\begin\{table\}(.*?)\\end\{table\}'
    tables = re.findall(table_pattern, content, re.DOTALL)
    
    # Process each table
    for table in tables:
        # Find caption
        caption_pattern = r'\\caption\{(.*?)\}'
        caption_match = re.search(caption_pattern, table, re.DOTALL)
        
        if caption_match:
            original_caption = caption_match.group(1)
            
            # Shorten caption - keep only the first sentence
            shortened_caption = re.split(r'\.', original_caption, 1)[0] + '.'
            
            # Replace in content
            content = content.replace(
                f"\\caption{{{original_caption}}}", 
                f"\\caption{{{shortened_caption}}}"
            )
    
    return content

def shorten_report(content, focus_sections=None):
    """Shorten the report with a focus on specified sections"""
    # Split document by sections
    sections = split_by_sections(content)
    
    # Define reduction factors for different sections
    reduction_factors = {
        'Introduction': 0.7,  # Reduce by 70%
        'Related Work': 0.8,  # Reduce by 80% 
        'Methodology': 0.4,   # Reduce by 40%
        'Results': 0.2,       # Reduce by only 20%
        'Discussion': 0.4,    # Reduce by 40%
        'Conclusion': 0.5     # Reduce by 50%
    }
    
    # Default reduction factor for unspecified sections
    default_reduction = 0.6
    
    # Process each section
    shortened_sections = []
    for section in sections:
        section_name = section['name']
        section_content = section['content']
        
        # Determine reduction factor
        for key in reduction_factors:
            if key in section_name:
                reduction = reduction_factors[key]
                break
        else:
            reduction = default_reduction
        
        # Apply different reduction based on focus areas
        if focus_sections and any(focus in section_name for focus in focus_sections):
            # Reduce less for focus sections
            reduction = max(0.2, reduction - 0.3)
        
        # Skip shortening for certain critical sections
        if 'Abstract' in section_name or 'Acknowledgements' in section_name:
            shortened_sections.append(section_content)
            continue
        
        # Shorten the section
        shortened_content = shorten_section(section_content, reduction)
        shortened_sections.append(shortened_content)
    
    # Get document preamble (before first section)
    first_section_start = content.find('\\section')
    preamble = content[:first_section_start]
    
    # Reconstruct document
    shortened_content = preamble + ''.join(shortened_sections)
    
    # Update figure and table captions
    shortened_content = update_figure_captions(shortened_content)
    shortened_content = update_table_captions(shortened_content)
    
    return shortened_content

def update_bibliography(content):
    """Update bibliography to reduce space"""
    # Find if there are any unnecessary spaces in the bibliography
    bib_pattern = r'\\bibliography\{([^}]+)\}'
    bib_match = re.search(bib_pattern, content)
    
    if bib_match:
        # Add a command to reduce space in bibliography
        # This works if using natbib or similar
        bib_cmds = r"""
% Reduce space in bibliography
\setlength{\bibsep}{0.0pt}
"""
        # Add before bibliography command
        content = content.replace(bib_match.group(0), bib_cmds + bib_match.group(0))
    
    return content

def main():
    parser = argparse.ArgumentParser(description="Shorten the CS297-298-Xiangyi-Report")
    parser.add_argument('--input', type=str, default='CS297-298-Xiangyi-Report/main.tex',
                        help='Input LaTeX file path')
    parser.add_argument('--output', type=str, default='CS297-298-Xiangyi-Report/main_shortened.tex',
                        help='Output LaTeX file path')
    parser.add_argument('--focus', type=str, nargs='+', 
                        default=['Results', 'Experiment', 'Discussion'],
                        help='Sections to focus on when shortening')
    
    args = parser.parse_args()
    
    # Read input file
    content = read_tex_file(args.input)
    
    # Identify sections
    identify_sections(content)
    
    # Count words in original document
    original_word_count = count_words(content)
    print(f"Original document has approximately {original_word_count} words")
    
    # Shorten report
    shortened_content = shorten_report(content, args.focus)
    
    # Update bibliography spacing
    shortened_content = update_bibliography(shortened_content)
    
    # Count words in shortened document
    shortened_word_count = count_words(shortened_content)
    print(f"Shortened document has approximately {shortened_word_count} words")
    print(f"Reduction: {100 - (shortened_word_count / original_word_count * 100):.1f}%")
    
    # Write output file
    write_tex_file(args.output, shortened_content)

if __name__ == "__main__":
    main() 