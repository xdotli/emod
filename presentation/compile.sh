#!/bin/bash

# Compile the LaTeX presentation to PDF
echo "Compiling LaTeX presentation..."

# Run pdflatex twice to resolve references
pdflatex -interaction=nonstopmode defense.tex
pdflatex -interaction=nonstopmode defense.tex

# Check if compilation was successful
if [ -f defense.pdf ]; then
    echo "Compilation successful! Presentation PDF created."
    echo "Output file: defense.pdf"
else
    echo "Error: Compilation failed."
    exit 1
fi

# Clean up auxiliary files
echo "Cleaning up auxiliary files..."
rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb

echo "Done."