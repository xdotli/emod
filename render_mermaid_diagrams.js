#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Create the output directory if it doesn't exist
const outputDir = path.join('CS297-298-Xiangyi-Report', 'Figures');
if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir, { recursive: true });
}

// Read the Mermaid diagrams file
const mermaidFile = 'model_architecture_diagrams.md';
const content = fs.readFileSync(mermaidFile, 'utf8');

// Function to extract Mermaid code blocks
function extractMermaidBlocks(content) {
  const regex = /```mermaid\n([\s\S]*?)\n```/g;
  const blocks = [];
  let match;

  while ((match = regex.exec(content)) !== null) {
    blocks.push(match[1].trim());
  }

  return blocks;
}

// Function to determine diagram filename based on content
function getDiagramName(mermaidCode, index) {
  const nameMap = {
    'flowchart TB': 'system_architecture',
    'flowchart TD': index === 1 ? 'text_model_architecture' : 
                    index === 2 ? 'fusion_strategies_comparison' : 
                    'experiment_framework',
    'flowchart LR': 'mfcc_pipeline'
  };

  // Get first line of diagram code to determine type
  const firstLine = mermaidCode.split('\n')[0].trim();
  return nameMap[firstLine] || `diagram_${index}`;
}

// Function to render a Mermaid diagram using mmdc CLI
function renderMermaidDiagram(mermaidCode, outputFileName) {
  // Write the Mermaid code to a temporary file
  const tempFile = `temp_diagram_${Date.now()}.mmd`;
  fs.writeFileSync(tempFile, mermaidCode);

  const outputFile = path.join(outputDir, `${outputFileName}.png`);
  
  try {
    console.log(`Rendering ${outputFileName}...`);
    
    // Install @mermaid-js/mermaid-cli if not already installed
    try {
      execSync('npx @mermaid-js/mermaid-cli -v', { stdio: 'ignore' });
    } catch (error) {
      console.log('Installing @mermaid-js/mermaid-cli...');
      execSync('npm install -g @mermaid-js/mermaid-cli');
    }
    
    // Execute the CLI command to render the diagram
    execSync(`npx mmdc -i ${tempFile} -o ${outputFile} -b transparent -t neutral`, 
             { stdio: 'inherit' });
    
    console.log(`✓ Created ${outputFileName}.png`);
  } catch (error) {
    console.error(`Error rendering ${outputFileName}:`, error.message);
  } finally {
    // Clean up the temporary file
    if (fs.existsSync(tempFile)) {
      fs.unlinkSync(tempFile);
    }
  }
}

// Main execution
console.log('Extracting Mermaid diagrams from markdown file...');
const mermaidBlocks = extractMermaidBlocks(content);

console.log(`Found ${mermaidBlocks.length} Mermaid diagrams.`);

// Render each diagram
mermaidBlocks.forEach((block, index) => {
  const fileName = getDiagramName(block, index);
  renderMermaidDiagram(block, fileName);
});

console.log('Diagram rendering complete!');

// Create a sample usage in the LaTeX file
// Extract section titles from markdown
function extractSectionTitles(content) {
  const regex = /## (\d+)\. (.*)/g;
  const titles = [];
  let match;

  while ((match = regex.exec(content)) !== null) {
    titles.push({ number: match[1], title: match[2].trim() });
  }

  return titles;
}

// Generate LaTeX code for the diagrams
function generateLatexCode(sectionTitles) {
  const diagrams = [
    'system_architecture',
    'text_model_architecture',
    'fusion_strategies_comparison',
    'mfcc_pipeline',
    'experiment_framework'
  ];
  
  let latexCode = '% Add these diagram figures to your LaTeX document\n\n';
  
  diagrams.forEach((diagram, index) => {
    const title = sectionTitles[index] ? sectionTitles[index].title : `Diagram ${index + 1}`;
    
    latexCode += `\\begin{figure}[h]\n`;
    latexCode += `    \\centering\n`;
    latexCode += `    \\includegraphics[width=0.9\\linewidth]{Figures/${diagram}.png}\n`;
    latexCode += `    \\caption{${title}}\n`;
    latexCode += `    \\label{fig:${diagram}}\n`;
    latexCode += `\\end{figure}\n\n`;
  });
  
  return latexCode;
}

// Get section titles and generate LaTeX code
const sectionTitles = extractSectionTitles(content);
const latexCode = generateLatexCode(sectionTitles);

// Write the LaTeX code to a file
const latexFile = 'mermaid_figures.tex';
fs.writeFileSync(latexFile, latexCode);

console.log(`LaTeX code for including the diagrams has been written to ${latexFile}.`); 