#!/usr/bin/env python3
"""
Script to replace citation question marks with proper references in LaTeX file
"""

import re
import os
import argparse

def read_file(file_path):
    """Read a file and return its content"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def write_file(file_path, content):
    """Write content to a file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Wrote updated content to {file_path}")

def fix_question_marks(content):
    """Replace citation question marks with proper citations"""
    # Match patterns like: BERT [?] or RoBERTa [?]
    pattern = r'(\w+)\s+\[\?\]'
    
    # Dictionary mapping models to their citation keys
    model_citations = {
        'BERT': 'devlin2018bert',
        'RoBERTa': 'liu2019roberta',
        'XLNet': 'yang2019xlnet',
        'ALBERT': 'lan2019albert',
        'ELECTRA': 'clark2020electra',
        'DeBERTa': 'he2020deberta',
        'Wav2vec': 'schneider2019wav2vec',
        'RAVDESS': 'livingstone2018ryerson',
        'IEMOCAP': 'busso2008iemocap',
        'TFN': 'zadeh2018multimodal_tfn',
        'MFN': 'zadeh2018mfn',
        'MOSI': 'zadeh2016mosi',
        'MOSEI': 'zadeh2018multimodal',
        'MELD': 'poria2018meld'
    }
    
    # Function to replace each match with appropriate citation
    def replace_match(match):
        model = match.group(1)
        # Check if we have a citation for this model
        if model in model_citations:
            return f"{model} \\cite{{{model_citations[model]}}}"
        # For models not in our dictionary, keep as is
        return match.group(0)
    
    # Replace question marks with citations
    updated_content = re.sub(pattern, replace_match, content)
    
    # Count replacements
    original_count = len(re.findall(r'\[\?\]', content))
    remaining_count = len(re.findall(r'\[\?\]', updated_content))
    replaced_count = original_count - remaining_count
    
    print(f"Replaced {replaced_count} question marks with citations")
    print(f"Remaining question marks: {remaining_count}")
    
    return updated_content

def main():
    parser = argparse.ArgumentParser(description="Fix citation question marks in LaTeX file")
    parser.add_argument('--file', '-f', type=str, default='main_complete.tex', help='LaTeX file path')
    parser.add_argument('--output', '-o', type=str, help='Output file path (defaults to overwriting input file)')
    
    args = parser.parse_args()
    
    # Read the LaTeX file
    print(f"Reading file: {args.file}")
    try:
        content = read_file(args.file)
    except FileNotFoundError:
        print(f"Error: File not found - {args.file}")
        return
    
    # Fix question marks
    updated_content = fix_question_marks(content)
    
    # Write the updated content
    output_file = args.output or args.file
    write_file(output_file, updated_content)
    
    print(f"Done! Updated content written to {output_file}")

if __name__ == "__main__":
    main() 