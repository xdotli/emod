#!/usr/bin/env python3
"""
Script to preprocess IEMOCAP data with reduced emotion categories.
"""

import pandas as pd
import numpy as np
import argparse
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Preprocess IEMOCAP data with reduced emotion categories')
    
    parser.add_argument('--input_path', type=str, default='IEMOCAP_Final.csv',
                        help='Path to the original IEMOCAP_Final.csv')
    parser.add_argument('--output_path', type=str, default='IEMOCAP_Reduced.csv',
                        help='Path to save the preprocessed data')
    
    return parser.parse_args()

def map_to_reduced_categories(emotion):
    """
    Map original emotion labels to reduced categories.
    
    Args:
        emotion (str): Original emotion label
        
    Returns:
        str: Mapped emotion label (Angry, Happy, Neutral, Sad)
    """
    emotion = emotion.strip()
    
    # Angry category
    if emotion in ['Anger', 'Frustration', 'Disgust', 'Other annoyed', 'Other indignation', 'Other exasperation']:
        return 'Angry'
    
    # Happy category
    elif emotion in ['Happiness', 'Excited', 'Other amused', 'Other pride', 'Other proud', 'Other impressed']:
        return 'Happy'
    
    # Sad category
    elif emotion in ['Sadness', 'Fear', 'Other despair', 'Other melancolia', 'Other melancolic', 
                    'Fear anxious as in expecting something big to happen', 'Other bored']:
        return 'Sad'
    
    # Neutral category
    elif emotion in ['Neutral state', 'Surprise', 'Other', 'Other shocked', 'Other sympathizing', 
                    'Other nervous', 'Other panic']:
        return 'Neutral'
    
    # Default to Neutral for any unmatched categories
    else:
        logger.warning(f"Unmatched emotion category: {emotion}, mapping to Neutral")
        return 'Neutral'

def extract_vad_values(dimension_str):
    """
    Extract VAD values from the dimension string.
    
    Args:
        dimension_str (str): String containing VAD values
        
    Returns:
        dict: Dictionary with valence, arousal, and dominance values
    """
    try:
        # Convert string representation to list of dictionaries
        dimension_list = eval(dimension_str)
        
        # Extract the first dictionary (assuming there's only one)
        if isinstance(dimension_list, list) and len(dimension_list) > 0:
            vad_dict = dimension_list[0]
            return vad_dict
        return None
    except:
        return None

def preprocess_data(df):
    """
    Preprocess IEMOCAP data with reduced emotion categories.
    
    Args:
        df (pd.DataFrame): Original IEMOCAP DataFrame
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame with reduced categories
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Extract VAD values from the 'dimension' column
    df['vad_values'] = df['dimension'].apply(extract_vad_values)
    
    # Create separate columns for valence, arousal, and dominance
    df['valence'] = df['vad_values'].apply(lambda x: x['valence'] if x else None)
    df['arousal'] = df['vad_values'].apply(lambda x: x['arousal'] if x else None)
    df['dominance'] = df['vad_values'].apply(lambda x: x['dominance'] if x else None)
    
    # Drop rows with missing VAD values
    df = df.dropna(subset=['valence', 'arousal', 'dominance'])
    
    # Map original emotion labels to reduced categories
    df['original_emotion'] = df['Major_emotion'].str.strip()
    df['emotion'] = df['original_emotion'].apply(map_to_reduced_categories)
    
    # Keep only necessary columns
    cols_to_keep = ['Transcript', 'valence', 'arousal', 'dominance', 'emotion', 'original_emotion']
    if 'Audio_Uttrance_Path' in df.columns:
        cols_to_keep.append('Audio_Uttrance_Path')
    
    df = df[cols_to_keep]
    
    return df

def main():
    """Main function."""
    args = parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.input_path}")
    df = pd.read_csv(args.input_path)
    
    # Preprocess data
    logger.info("Preprocessing data with reduced emotion categories")
    df_reduced = preprocess_data(df)
    
    # Save preprocessed data
    logger.info(f"Saving preprocessed data to {args.output_path}")
    df_reduced.to_csv(args.output_path, index=False)
    
    # Print statistics
    logger.info("Emotion category distribution:")
    emotion_counts = df_reduced['emotion'].value_counts()
    for emotion, count in emotion_counts.items():
        percentage = count / len(df_reduced) * 100
        logger.info(f"{emotion}: {count} ({percentage:.2f}%)")

if __name__ == '__main__':
    main()
