#!/usr/bin/env python3
"""
Script to fix specific citation issues in the main_complete.tex file
"""

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

def fix_specific_citations(content):
    """Fix specific citation issues found in the PDF"""
    
    # Find problematic patterns like: ALBERT [?]
    replacements = [
        # Replace ALBERT [?] with proper citation
        (r'ALBERT \[\OT1/cmr/bx/n/12 \?\OT1/cmr/m/n/12 \]', r'ALBERT \\cite{lan2019albert}'),
        
        # Replace IEMOCAP [?] with proper citation
        (r'IEMOCAP \[\OT1/cmr/bx/n/12 \?\OT1/cmr/m/n/12 \]', r'IEMOCAP \\cite{busso2008iemocap}'),
        
        # Direct replacement pattern
        (r'ALBERT \[\?\]', r'ALBERT \\cite{lan2019albert}'),
        
        # Fix any remaining [?] citations with BERT as fallback
        (r'\[\?\]', r'\\cite{devlin2018bert}')
    ]
    
    # Apply replacements
    updated_content = content
    for pattern, replacement in replacements:
        if re.search(pattern, updated_content):
            print(f"Found pattern: {pattern}")
            updated_content = re.sub(pattern, replacement, updated_content)
            print(f"Replaced with: {replacement}")
    
    return updated_content

def main():
    """Main function to read file, fix citations, and write output"""
    input_file = "main_complete.tex"
    output_file = "main_complete_fixed.tex"
    
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
    
    # Fix problematic patterns
    updated_content = fix_specific_citations(content)
    
    # Check line 174 specifically
    lines = updated_content.split('\n')
    line_174 = lines[173] if len(lines) > 173 else ""
    if "ALBERT" in line_174 and "?" in line_174:
        print(f"Line 174 still has issues: {line_174}")
        lines[173] = lines[173].replace('ALBERT [?]', 'ALBERT \\cite{lan2019albert}')
        updated_content = '\n'.join(lines)
        print(f"Fixed line 174: {lines[173]}")
    
    # Write output
    write_file(output_file, updated_content)
    print(f"\nDone! Fixed content written to {output_file}")
    print(f"To compile: pdflatex {output_file}")

if __name__ == "__main__":
    main() 