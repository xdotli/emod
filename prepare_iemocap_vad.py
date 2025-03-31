#!/usr/bin/env python3
"""
IEMOCAP Dataset Preparation for VAD Prediction

This script processes the IEMOCAP dataset to extract utterances,
transcripts, emotion labels, and VAD values for training VAD prediction models.
"""

import os
import re
import pandas as pd
import glob
import argparse
from pathlib import Path
import numpy as np

# Constants
EMOTIONS = ["angry", "happy", "sad", "neutral"]  # Target emotions
EMOTION_MAP = {
    'ang': 'angry',
    'hap': 'happy',
    'exc': 'happy',  # Map excited to happy
    'sad': 'sad',
    'neu': 'neutral',
    'fru': 'angry',  # Map frustrated to angry
    # Other emotions like fear, disgust, surprise will be filtered out
}

# Default paths
DEFAULT_IEMOCAP_DIR = "Datasets/IEMOCAP_full_release"
DEFAULT_OUTPUT_DIR = "data"


def extract_vad_values(evalline):
    """
    Extract VAD values from evaluation line
    
    Args:
        evalline: Line from evaluation file containing VAD values
        
    Returns:
        List of VAD values [valence, arousal, dominance]
    """
    # Extract values in square brackets
    vad_match = re.search(r'\[([^]]+)\]', evalline)
    if vad_match:
        try:
            # Parse VAD values and convert to float
            vad_str = vad_match.group(1)
            vad_values = [float(val.strip()) for val in vad_str.split(',')]
            
            # Convert from 1-5 scale to [-1, 1] scale
            vad_values = [(val - 3) / 2 for val in vad_values]
            
            return vad_values
        except:
            pass
    
    # Default values if extraction fails
    return [0.0, 0.0, 0.0]


def get_transcript(session_dir, utterance_id):
    """
    Get transcript text for a given utterance ID
    
    Args:
        session_dir: Path to session directory
        utterance_id: Utterance ID
        
    Returns:
        Transcript text for the utterance
    """
    # Prepare parts of the utterance ID
    parts = utterance_id.split('_')
    session = parts[0]
    dialogue_id = parts[0] + '_' + parts[1]
    
    # Construct path to transcript file
    transcript_file = os.path.join(
        session_dir,
        'dialog',
        'transcriptions',
        f'{dialogue_id}.txt'
    )
    
    # Search for utterance in transcript file
    if os.path.exists(transcript_file):
        with open(transcript_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if utterance_id in line:
                    # Extract text after ":" or "]"
                    text_match = re.search(r'[:\]](.+)', line)
                    if text_match:
                        text = text_match.group(1).strip()
                        return text
    
    return ""


def process_evaluation_file(eval_file, session_dir):
    """
    Process a single evaluation file to extract data
    
    Args:
        eval_file: Path to evaluation file
        session_dir: Path to session directory
        
    Returns:
        List of dictionaries with utterance data
    """
    data = []
    
    with open(eval_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        in_utterance = False
        current_utterance = None
        emotion = None
        vad_values = None
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('%'):
                continue
            
            # Check if line contains utterance information
            if '[' in line and ']' in line and '\t' in line:
                # Start new utterance
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    # Get utterance ID
                    utterance_id = parts[1].strip()
                    
                    # Get emotion label
                    raw_emotion = parts[2]
                    
                    # Get VAD values if available
                    vad_values = [0, 0, 0]  # Default
                    if len(parts) >= 4 and "[" in parts[3] and "]" in parts[3]:
                        vad_values = extract_vad_values(parts[3])
                    
                    # Map emotion to standardized categories and filter
                    if raw_emotion in EMOTION_MAP:
                        emotion = EMOTION_MAP[raw_emotion]
                    elif raw_emotion in EMOTIONS:
                        emotion = raw_emotion
                    else:
                        # Skip other emotions or "xxx" labels
                        continue
                    
                    # Get transcript
                    text = get_transcript(session_dir, utterance_id)
                    
                    # Add to dataset
                    if text:
                        data.append({
                            'utterance_id': utterance_id,
                            'text': text,
                            'emotion': emotion,
                            'valence': vad_values[0],
                            'arousal': vad_values[1],
                            'dominance': vad_values[2]
                        })
    
    return data


def process_iemocap(iemocap_dir, output_dir):
    """
    Process entire IEMOCAP dataset
    
    Args:
        iemocap_dir: Path to IEMOCAP dataset
        output_dir: Path to output directory
        
    Returns:
        Path to output CSV file
    """
    print(f"Processing IEMOCAP dataset from {iemocap_dir}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Output file path
    output_file = os.path.join(output_dir, 'iemocap_vad.csv')
    
    # Check if output file already exists
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Skipping processing.")
        return output_file
    
    # Find all evaluation files
    all_data = []
    
    # Process each session
    for session_idx in range(1, 6):  # Sessions 1-5
        session_dir = os.path.join(iemocap_dir, f'Session{session_idx}')
        if not os.path.exists(session_dir):
            print(f"Session directory {session_dir} not found. Skipping.")
            continue
        
        # Find evaluation files in this session
        eval_dir = os.path.join(session_dir, 'dialog', 'EmoEvaluation')
        eval_files = glob.glob(os.path.join(eval_dir, '*.txt'))
        
        print(f"Processing Session {session_idx} - Found {len(eval_files)} evaluation files")
        
        # Process each evaluation file
        for eval_file in eval_files:
            data = process_evaluation_file(eval_file, session_dir)
            all_data.extend(data)
            print(f"Processed {os.path.basename(eval_file)} - Found {len(data)} utterances")
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Print dataset statistics
    print("\nDataset statistics:")
    print(f"Total utterances: {len(df)}")
    print(f"Emotion distribution:\n{df['emotion'].value_counts()}")
    print(f"Average utterance length: {df['text'].str.len().mean():.1f} characters")
    
    # Calculate VAD statistics
    print("\nVAD statistics:")
    print(f"Valence mean: {df['valence'].mean():.3f}, std: {df['valence'].std():.3f}")
    print(f"Arousal mean: {df['arousal'].mean():.3f}, std: {df['arousal'].std():.3f}")
    print(f"Dominance mean: {df['dominance'].mean():.3f}, std: {df['dominance'].std():.3f}")
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Prepare IEMOCAP dataset for VAD prediction")
    parser.add_argument('--iemocap_dir', type=str, default=DEFAULT_IEMOCAP_DIR,
                        help=f'Path to IEMOCAP dataset (default: {DEFAULT_IEMOCAP_DIR})')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Path to output directory (default: {DEFAULT_OUTPUT_DIR})')
    
    args = parser.parse_args()
    
    # Process dataset
    output_file = process_iemocap(args.iemocap_dir, args.output_dir)
    print(f"Dataset preparation complete. Data saved to {output_file}")


if __name__ == "__main__":
    main() 