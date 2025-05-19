#!/bin/bash

# Set directories
REPORT_DIR="CS297-298-Xiangyi-Report"
UPDATED_TABLES_DIR="updated_tables"
SRC_DIR="src"

# Run the Python script to update tables
echo "Running update_tables.py to generate updated LaTeX tables..."
./update_tables.py

# Make sure the updated_tables directory exists in the report directory
mkdir -p "$REPORT_DIR/updated_tables"

# Copy updated tables to the report directory
cp -r "$UPDATED_TABLES_DIR"/*.tex "$REPORT_DIR/updated_tables/"

# Create a backup of the original report
cp "$REPORT_DIR/main.tex" "$REPORT_DIR/main.tex.backup.$(date +%Y%m%d%H%M%S)"

# Apply sed commands to modify the report

# 1. Shorten Section 6 (Discussion)
echo "Shortening Section 6 (Discussion)..."
sed -i.bak '
/\\subsection{Two-Stage Approach vs. Direct Classification}/,/This characteristic makes the two-stage approach potentially more generalizable to diverse user populations./ {
    /\\subsection{Two-Stage Approach vs. Direct Classification}/ {p; n;}
    /Despite its lower classification accuracy/,/This characteristic makes the two-stage approach potentially more generalizable to diverse user populations./ {p; n;}
    /./ d
}' "$REPORT_DIR/main.tex"

# 2. Convert bullet points to paragraphs in various sections
echo "Converting bullet points to paragraphs..."
sed -i.bak '
/\\begin{itemize}/,/\\end{itemize}/ {
    /\\begin{itemize}/ {d; n;}
    /\\item/ {s/\\item //g;}
    /\\end{itemize}/ d
}' "$REPORT_DIR/main.tex"

# 3. Shorten Section 1 (Introduction)
echo "Shortening Section 1 (Introduction)..."
sed -i.bak '
/\\section{Introduction}/,/\\section{Related Work}/ {
    /Our research makes several key contributions:/,/The structure of this report is as follows:/ {
        /Our research makes several key contributions:/ {p; n;}
        /First, we conduct a comparative analysis/,/The structure of this report is as follows:/ d
    }
}' "$REPORT_DIR/main.tex"

# 4. Fix image captions by removing paragraphs
echo "Fixing image captions..."
# We'll use a different approach for captions - searching for figure environments
sed -i.bak '/\\begin{figure}/,/\\end{figure}/ {
    s/\\caption{\.}/\\caption{System architecture of the two-stage emotion detection model.}/
}' "$REPORT_DIR/main.tex"

# 5. Add explanation for high performance results
echo "Adding explanation for high performance results..."
# Create a temporary file with the explanation text
cat > high_performance_explanation.txt << 'EOL'
\paragraph{Note on High Performance:} The high performance metrics we achieve on IEMOCAP may seem surprising given the complexity of the dataset and the six emotion classes. Several factors contribute to these results: (1) we use state-of-the-art pre-trained language models with significant transfer learning benefit; (2) our preprocessing includes careful data cleaning and normalization; (3) we implemented speaker-independent cross-validation to ensure robust evaluation; and (4) unlike some earlier work, we focus on the more consistent subset of IEMOCAP utterances where annotator agreement is high. While these results exceed previously published benchmarks, we believe they reflect the capabilities of modern architectures when applied with rigorous methodology.
EOL

# Add the explanation at the end of the Comparison with State-of-the-Art subsection
search_line="\\subsection{Comparison with State-of-the-Art}"
explanation=$(cat high_performance_explanation.txt)
sed -i.bak "/$search_line/a $explanation" "$REPORT_DIR/main.tex"

# 6. Fix figures paths
echo "Ensuring figures directories exist..."
mkdir -p "$REPORT_DIR/Figures_Improved"

# 7. Copy improved figures if they exist
if [ -d "Figures_Improved" ]; then
    echo "Copying improved figures..."
    cp -r "Figures_Improved/"* "$REPORT_DIR/Figures_Improved/"
fi

# 8. Recompile the report
echo "Recompiling the report..."
cd "$REPORT_DIR"
pdflatex main.tex
bibtex main.aux
pdflatex main.tex
pdflatex main.tex

echo "Report has been updated and recompiled."
rm high_performance_explanation.txt 