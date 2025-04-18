"""
Data loader for IEMOCAP dataset.
This module provides functions to load and preprocess the IEMOCAP dataset.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import json
import logging

logger = logging.getLogger(__name__)

def load_iemocap_data(file_path, test_size=0.2, val_size=0.1, random_state=42):
    """
    Load IEMOCAP data from CSV file and split into train, validation, and test sets.

    Args:
        file_path (str): Path to the IEMOCAP_Final.csv file
        test_size (float): Proportion of data to use for testing
        val_size (float): Proportion of data to use for validation
        random_state (int): Random seed for reproducibility

    Returns:
        dict: Dictionary containing train, val, and test DataFrames
    """
    logger.info(f"Loading IEMOCAP data from {file_path}")

    # Load the data
    df = pd.read_csv(file_path)

    # Clean and preprocess the data
    df = preprocess_iemocap_data(df)

    # Split into train and test
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    # Split train into train and validation
    train_df, val_df = train_test_split(train_df, test_size=val_size/(1-test_size), random_state=random_state)

    logger.info(f"Data split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    return {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }

def preprocess_iemocap_data(df):
    """
    Preprocess IEMOCAP data.

    Args:
        df (pd.DataFrame): Raw IEMOCAP DataFrame

    Returns:
        pd.DataFrame: Preprocessed DataFrame
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

    # Use Major_emotion as the emotion label
    df['emotion'] = df['Major_emotion'].str.strip()

    # Keep only necessary columns
    cols_to_keep = ['Transcript', 'valence', 'arousal', 'dominance', 'emotion']
    df = df[cols_to_keep]

    return df

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

def scale_vad_values(train_data, val_data, test_data, scaler_path=None):
    """
    Scale VAD values using StandardScaler.

    Args:
        train_data (pd.DataFrame): Training data
        val_data (pd.DataFrame): Validation data
        test_data (pd.DataFrame): Test data
        scaler_path (str): Path to save the scaler

    Returns:
        tuple: Tuple containing scaled train, val, and test DataFrames
    """
    # Extract VAD values
    train_vad = train_data[['valence', 'arousal', 'dominance']].values
    val_vad = val_data[['valence', 'arousal', 'dominance']].values
    test_vad = test_data[['valence', 'arousal', 'dominance']].values

    # Fit scaler on training data
    scaler = StandardScaler()
    train_vad_scaled = scaler.fit_transform(train_vad)

    # Transform validation and test data
    val_vad_scaled = scaler.transform(val_vad)
    test_vad_scaled = scaler.transform(test_vad)

    # Create new DataFrames with scaled values
    train_data_scaled = train_data.copy()
    val_data_scaled = val_data.copy()
    test_data_scaled = test_data.copy()

    train_data_scaled[['valence', 'arousal', 'dominance']] = train_vad_scaled
    val_data_scaled[['valence', 'arousal', 'dominance']] = val_vad_scaled
    test_data_scaled[['valence', 'arousal', 'dominance']] = test_vad_scaled

    # Save the scaler if path is provided
    if scaler_path:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"Scaler saved to {scaler_path}")

    return train_data_scaled, val_data_scaled, test_data_scaled, scaler

def get_emotion_mapping(data):
    """
    Create a mapping between emotion labels and indices.

    Args:
        data (pd.DataFrame): DataFrame containing emotion labels

    Returns:
        tuple: Tuple containing emotion-to-index and index-to-emotion mappings
    """
    # Get unique emotions from all data
    unique_emotions = sorted(data['emotion'].unique())

    # Create mappings
    emotion_to_idx = {emotion: idx for idx, emotion in enumerate(unique_emotions)}
    idx_to_emotion = {idx: emotion for emotion, idx in emotion_to_idx.items()}

    # Add an 'unknown' category for handling unseen emotions
    unknown_idx = len(emotion_to_idx)
    emotion_to_idx['unknown'] = unknown_idx
    idx_to_emotion[unknown_idx] = 'unknown'

    return emotion_to_idx, idx_to_emotion

def save_data_splits(data_splits, output_dir):
    """
    Save data splits to CSV files.

    Args:
        data_splits (dict): Dictionary containing train, val, and test DataFrames
        output_dir (str): Directory to save the CSV files
    """
    os.makedirs(output_dir, exist_ok=True)

    for split_name, df in data_splits.items():
        output_path = os.path.join(output_dir, f"{split_name}.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {split_name} data to {output_path}")
