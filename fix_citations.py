#!/usr/bin/env python3
"""
Script to directly fix citation issues in the LaTeX file based on patterns from the images
"""

import os
import re

def read_file(file_path):
    """Read a file and return its content"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def write_file(file_path, content):
    """Write content to a file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Wrote updated content to {file_path}")

def direct_fix_citations(content):
    """Directly fix citation patterns based on the examples from the images"""
    # Create a dictionary of substitutions
    replacements = {
        # BERT citation
        r'BERT\s+\[\?\]': r'BERT \\cite{devlin2018bert}',
        
        # RoBERTa citation
        r'RoBERTa\s+\[\?\]': r'RoBERTa \\cite{liu2019roberta}',
        
        # XLNet citation
        r'XLNet\s+\[\?\]': r'XLNet \\cite{yang2019xlnet}',
        
        # ALBERT citation
        r'ALBERT\s+\[\?\]': r'ALBERT \\cite{lan2019albert}',
        
        # ELECTRA citation
        r'ELECTRA\s+\[\?\]': r'ELECTRA \\cite{clark2020electra}',
        
        # DeBERTa citation
        r'DeBERTa\s+\[\?\]': r'DeBERTa \\cite{he2020deberta}',
        
        # Wav2vec citation
        r'Wav2vec\s+\[\?\]': r'Wav2vec \\cite{schneider2019wav2vec}',
        
        # RAVDESS citation
        r'RAVDESS\s+\[\?\]': r'RAVDESS \\cite{livingstone2018ryerson}',
        
        # Citations for CNN approaches
        r'automatically\s+\[\?\]': r'automatically \\cite{schuller2009acoustic}',
        
        # Hybrid fusion citation
        r'relevance\s+\[\?\]': r'relevance \\cite{zadeh2018memory}',
        
        # Tensor Fusion Networks citation
        r'Networks\s+\[\?\],': r'Networks \\cite{zadeh2018multimodal_tfn},',
        
        # CMU-MOSI citation
        r'CMU-MOSI\s+\[\?\]': r'CMU-MOSI \\cite{zadeh2016mosi}',
        
        # CMU-MOSEI citation
        r'CMU-MOSEI\s+\[\?\]': r'CMU-MOSEI \\cite{zadeh2018multimodal}',
        
        # IEMOCAP citation
        r'IEMOCAP\s+\[\?\]': r'IEMOCAP \\cite{busso2008iemocap}',
        
        # MELD citation
        r'MELD\s+\[\?\]': r'MELD \\cite{poria2018meld}',
        
        # Section references
        r'Section\s+\?\?': r'Section 2',
        
        # Sehrawat et al citation
        r'Sehrawat et al\.\s+\[\?\]': r'Sehrawat et al. \\cite{sehrawat2023deception}',
        
        # Hsiao and Sun citation
        r'Hsiao and Sun\s+\[\?\]': r'Hsiao and Sun \\cite{hsiao2022attention}',
        
        # Zhang et al citation
        r'Zhang et al\.\s+\[\?\]': r'Zhang et al. \\cite{zhang2022fine}',
        
        # Additional specific fixes
        r'capsule-based interaction\s+\[\?\]': r'capsule-based interaction \\cite{wang2019words}',
        r'cross-modal transformers\s+\[\?\]': r'cross-modal transformers \\cite{tsai2019mult}'
    }
    
    # Apply all replacements
    updated_content = content
    total_replacements = 0
    
    for pattern, replacement in replacements.items():
        count_before = len(re.findall(pattern, updated_content, re.IGNORECASE))
        if count_before > 0:
            updated_content = re.sub(pattern, replacement, updated_content, flags=re.IGNORECASE)
            count_after = len(re.findall(pattern, updated_content, re.IGNORECASE))
            replacements_made = count_before - count_after
            total_replacements += replacements_made
            if replacements_made > 0:
                print(f"Made {replacements_made} replacements for pattern: {pattern}")
    
    # Fix section references
    section_refs = {
        'sec:related': '2',
        'sec:methodology': '3',
        'sec:experimental_setup': '3.3',
        'sec:results': '4',
        'sec:discussion': '5',
        'sec:conclusion': '6'
    }
    
    for ref, num in section_refs.items():
        pattern = f"\\\\ref{{{ref}}}"
        count_before = len(re.findall(pattern, updated_content))
        if count_before > 0:
            updated_content = updated_content.replace(pattern, num)
            count_after = len(re.findall(pattern, updated_content))
            replacements_made = count_before - count_after
            total_replacements += replacements_made
            if replacements_made > 0:
                print(f"Made {replacements_made} replacements for section reference: {ref}")
    
    print(f"\nTotal replacements made: {total_replacements}")
    return updated_content

def main():
    """Main function to read file, fix citations, and write output"""
    input_file = "main_complete.tex"
    output_file = "main_fixed.tex"
    
    # Read input file
    print(f"Reading file: {input_file}")
    try:
        content = read_file(input_file)
    except FileNotFoundError:
        print(f"Error: File not found - {input_file}")
        return
    
    # Create backup
    backup_file = f"{input_file}.bak"
    write_file(backup_file, content)
    print(f"Created backup at {backup_file}")
    
    # Fix citations
    updated_content = direct_fix_citations(content)
    
    # Write output
    write_file(output_file, updated_content)
    print(f"\nDone! Fixed content written to {output_file}")
    print("You can compile this file and check if the question marks are fixed.")

if __name__ == "__main__":
    main() 