#!/usr/bin/env python3
"""
Script to restore missing references in the shortened LaTeX document
Ensures all citations from the original document are preserved
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
    print(f"Wrote updated content to {file_path}")

def extract_citations(content):
    """Extract all citations from LaTeX content"""
    # Match \cite{...}, \citep{...}, \citet{...}, etc.
    citation_pattern = r'\\cite[pt]?\{([^}]+)\}'
    
    # Find all citation commands
    citation_matches = re.findall(citation_pattern, content)
    
    # Process each citation - they might contain multiple references separated by commas
    all_citations = []
    for match in citation_matches:
        citations = match.split(',')
        all_citations.extend([c.strip() for c in citations])
    
    # Return unique citations
    return set(all_citations)

def check_missing_references(original_content, shortened_content):
    """Check for citations that exist in the original but not in the shortened content"""
    original_citations = extract_citations(original_content)
    shortened_citations = extract_citations(shortened_content)
    
    # Find citations in original that are not in shortened
    missing_citations = original_citations - shortened_citations
    
    print(f"Found {len(original_citations)} citations in original document")
    print(f"Found {len(shortened_citations)} citations in shortened document")
    print(f"Missing {len(missing_citations)} citations")
    
    return missing_citations, original_citations, shortened_citations

def restore_missing_references(shortened_content, missing_citations):
    """Restore missing references in the shortened content"""
    if not missing_citations:
        print("No missing citations to restore")
        return shortened_content
    
    # Add a new section for unused references
    section = "\n\n\\section*{Additional References}\n"
    section += "The following references are part of the complete bibliography but are not directly cited in this shortened version.\n\n"
    
    # Add dummy citations for each missing reference
    for citation in sorted(missing_citations):
        section += f"\\nocite{{{citation}}}\n"
    
    # Find bibliography command to insert before it
    bib_pattern = r'\\bibliography\{([^}]+)\}'
    bib_match = re.search(bib_pattern, shortened_content)
    
    if bib_match:
        # Insert before bibliography
        updated_content = shortened_content[:bib_match.start()] + section + shortened_content[bib_match.start():]
    else:
        # Append at the end if no bibliography command found
        updated_content = shortened_content + section
        print("Warning: No \\bibliography command found, appending at the end")
    
    return updated_content

def add_nocite_command(content):
    """Add \\nocite{*} command to include all references if it doesn't exist"""
    # Check if \nocite{*} already exists
    if '\\nocite{*}' in content:
        print("\\nocite{*} command already exists")
        return content
    
    # Find bibliography command to insert before it
    bib_pattern = r'\\bibliography\{([^}]+)\}'
    bib_match = re.search(bib_pattern, content)
    
    if bib_match:
        # Insert before bibliography
        updated_content = content[:bib_match.start()] + "\n\n\\nocite{*}\n\n" + content[bib_match.start():]
        print("Added \\nocite{*} command before bibliography")
    else:
        # No change if no bibliography command found
        updated_content = content
        print("Warning: No \\bibliography command found, not adding \\nocite{*}")
    
    return updated_content

def extract_cite_commands(content):
    """Extract actual citation commands from LaTeX content"""
    citation_pattern = r'(\\cite[pt]?\{[^}]+\})'
    return re.findall(citation_pattern, content)

def ensure_bibliography_style(content):
    """Ensure bibliography style is set properly"""
    # Check if bibliography style is already set
    if '\\bibliographystyle{' in content:
        print("Bibliography style already set")
        return content
    
    # Find bibliography command to insert before it
    bib_pattern = r'\\bibliography\{([^}]+)\}'
    bib_match = re.search(bib_pattern, content)
    
    if bib_match:
        # Insert before bibliography
        updated_content = content[:bib_match.start()] + "\n\\bibliographystyle{IEEEtran}\n" + content[bib_match.start():]
        print("Added bibliography style command")
    else:
        # No change if no bibliography command found
        updated_content = content
        print("Warning: No \\bibliography command found, not adding bibliography style")
    
    return updated_content

def compare_sections(original_content, shortened_content):
    """Compare sections in original and shortened documents"""
    # Find all section commands
    section_pattern = r'\\section\{([^}]+)\}'
    
    original_sections = re.findall(section_pattern, original_content)
    shortened_sections = re.findall(section_pattern, shortened_content)
    
    print("\nSection comparison:")
    print(f"Original document: {len(original_sections)} sections")
    print(f"Shortened document: {len(shortened_sections)} sections")
    
    missing_sections = set(original_sections) - set(shortened_sections)
    if missing_sections:
        print(f"Missing sections: {missing_sections}")
    
    return original_sections, shortened_sections

def fix_citation_formatting(content):
    """Fix citation formatting issues"""
    # Look for consecutive citations that should be combined
    # e.g., \cite{ref1} \cite{ref2} -> \cite{ref1, ref2}
    consecutive_pattern = r'\\cite\{([^}]+)\}\s*\\cite\{([^}]+)\}'
    
    def combine_citations(match):
        return f"\\cite{{{match.group(1)}, {match.group(2)}}}"
    
    # Apply replacement
    updated_content = re.sub(consecutive_pattern, combine_citations, content)
    
    return updated_content

def create_references_section(content, all_citations):
    """Create a references section with all citations"""
    # Create a dedicated references section
    section = "\n\n\\section{References}\n"
    
    # Add paragraph about references
    section += "This section lists the references cited in this report.\n\n"
    
    # Add all citations
    for citation in sorted(all_citations):
        section += f"\\cite{{{citation}}}\n"
    
    # Check if there's already a References section
    if "\\section{References}" in content:
        print("References section already exists")
        return content
    
    # Find bibliography command to insert before it
    bib_pattern = r'\\bibliography\{([^}]+)\}'
    bib_match = re.search(bib_pattern, content)
    
    if bib_match:
        # Insert before bibliography
        updated_content = content[:bib_match.start()] + section + content[bib_match.start():]
    else:
        # Append at the end if no bibliography command found
        updated_content = content + section
        print("Warning: No \\bibliography command found, appending References section at the end")
    
    return updated_content

def main():
    parser = argparse.ArgumentParser(description="Restore missing references in shortened LaTeX document")
    parser.add_argument('--original', type=str, default='CS297-298-Xiangyi-Report/main.tex',
                        help='Original LaTeX file path')
    parser.add_argument('--shortened', type=str, default='CS297-298-Xiangyi-Report/main_shortened.tex',
                        help='Shortened LaTeX file path')
    parser.add_argument('--output', type=str, default='CS297-298-Xiangyi-Report/main_complete.tex',
                        help='Output LaTeX file path')
    parser.add_argument('--method', type=str, choices=['nocite', 'section', 'all'], default='all',
                        help='Method to restore references: nocite - add \\nocite{*}, section - create references section, all - both')
    
    args = parser.parse_args()
    
    # Read input files
    original_content = read_tex_file(args.original)
    shortened_content = read_tex_file(args.shortened)
    
    # Check for missing references
    missing_citations, original_citations, shortened_citations = check_missing_references(original_content, shortened_content)
    
    # Compare sections
    compare_sections(original_content, shortened_content)
    
    # Update content
    updated_content = shortened_content
    
    # Method 1: Use \nocite for missing references
    if args.method in ['nocite', 'all']:
        if missing_citations:
            updated_content = restore_missing_references(updated_content, missing_citations)
        else:
            print("No missing citations to restore with \\nocite")
    
    # Method 2: Add \nocite{*} command
    if args.method in ['all']:
        updated_content = add_nocite_command(updated_content)
    
    # Method 3: Create references section with all citations
    if args.method in ['section', 'all']:
        updated_content = create_references_section(updated_content, original_citations)
    
    # Fix formatting issues
    updated_content = fix_citation_formatting(updated_content)
    
    # Ensure bibliography style is set
    updated_content = ensure_bibliography_style(updated_content)
    
    # Write output file
    write_tex_file(args.output, updated_content)
    
    print(f"\nUpdated document saved to {args.output}")
    print(f"- Original citations: {len(original_citations)}")
    print(f"- Shortened citations: {len(shortened_citations)}")
    print(f"- Restored citations: {len(original_citations)}")

if __name__ == "__main__":
    main() 