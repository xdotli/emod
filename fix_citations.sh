#!/bin/bash

# Create a backup
cp main_complete.tex main_complete.tex.bak
echo "Backup created at main_complete.tex.bak"

# Process the main_complete.tex file
cat main_complete.tex |
  # Fix model citations
  sed 's/BERT \[\?\]/BERT \\cite{devlin2018bert}/g' |
  sed 's/RoBERTa \[\?\]/RoBERTa \\cite{liu2019roberta}/g' |
  sed 's/XLNet \[\?\]/XLNet \\cite{yang2019xlnet}/g' |
  sed 's/ALBERT \[\?\]/ALBERT \\cite{lan2019albert}/g' |
  sed 's/ELECTRA \[\?\]/ELECTRA \\cite{clark2020electra}/g' |
  sed 's/DeBERTa \[\?\]/DeBERTa \\cite{he2020deberta}/g' |
  sed 's/Wav2vec \[\?\]/Wav2vec \\cite{schneider2019wav2vec}/g' |
  sed 's/RAVDESS \[\?\]/RAVDESS \\cite{livingstone2018ryerson}/g' |
  
  # Fix dataset citations
  sed 's/IEMOCAP \[\?\]/IEMOCAP \\cite{busso2008iemocap}/g' |
  sed 's/CMU-MOSI \[\?\]/CMU-MOSI \\cite{zadeh2016mosi}/g' |
  sed 's/CMU-MOSEI \[\?\]/CMU-MOSEI \\cite{zadeh2018multimodal}/g' |
  sed 's/MELD \[\?\]/MELD \\cite{poria2018meld}/g' |
  
  # Fix author citations
  sed 's/Sehrawat et al\. \[\?\]/Sehrawat et al. \\cite{sehrawat2023deception}/g' |
  sed 's/Hsiao and Sun \[\?\]/Hsiao and Sun \\cite{hsiao2022attention}/g' |
  sed 's/Zhang et al\. \[\?\]/Zhang et al. \\cite{zhang2022fine}/g' |
  
  # Fix other citations
  sed 's/automatically \[\?\]/automatically \\cite{schuller2009acoustic}/g' |
  sed 's/relevance \[\?\]/relevance \\cite{zadeh2018memory}/g' |
  sed 's/Networks \[\?\]/Networks \\cite{zadeh2018multimodal_tfn}/g' |
  sed 's/interaction \[\?\]/interaction \\cite{wang2019words}/g' |
  sed 's/transformers \[\?\]/transformers \\cite{tsai2019mult}/g' |
  
  # Fix section references
  sed 's/Section \?\?/Section 2/g' |
  sed 's/\\ref{sec:related}/2/g' |
  sed 's/\\ref{sec:methodology}/3/g' |
  sed 's/\\ref{sec:experimental_setup}/3.3/g' |
  sed 's/\\ref{sec:results}/4/g' |
  sed 's/\\ref{sec:discussion}/5/g' |
  sed 's/\\ref{sec:conclusion}/6/g' > main_fixed.tex

echo "Fixed citations in main_fixed.tex"
echo "To compile the fixed file: pdflatex main_fixed.tex && bibtex main_fixed && pdflatex main_fixed.tex && pdflatex main_fixed.tex" 