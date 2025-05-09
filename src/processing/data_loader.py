"""
Data loading and preprocessing module for IEMOCAP dataset.

This module handles loading, parsing, and preprocessing of the IEMOCAP dataset
for use in the two-stage emotion recognition system.
"""

import os
import ast
import numpy as np
import pandas as pd
from collections import Counter
import librosa
from scipy.stats import skew, kurtosis
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
EMOTION_MAPPING = {
    'neutral': 'neutral',
    'frustration': 'angry',
    'anger': 'angry',
    'surprise': None,
    'disgust': None,
    'other': None,
    'sadness': 'sad',
    'fear': None,
    'happiness': 'happy',
    'excited': 'happy'
}

# Audio feature extraction parameters
SAMPLE_RATE = 16000  # Hz
FRAME_LENGTH = 0.025  # seconds
FRAME_STEP = 0.01  # seconds
NUM_MFCC = 13  # Number of MFCC coefficients to extract

def get_majority_label(label_str):
    """Get the majority emotion label from a list of labels."""
    try:
        labels = ast.literal_eval(label_str)
        count = Counter(labels)
        most_common = count.most_common(1)[0]
        if most_common[1] >= 2:
            return most_common[0]
        else:
            return None
    except:
        return None

def smart_parse(x):
    """Parse the dimension column to extract VAD values."""
    try:
        if pd.isna(x):
            return None
        parsed = ast.literal_eval(x)
        if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
            return parsed[0]
        elif isinstance(parsed, dict):
            return parsed
    except Exception:
        return None

def extract_audio_features(audio_path, sr=SAMPLE_RATE):
    """Extract acoustic features from audio file."""
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=sr)
        
        # Feature extraction
        # 1. Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=NUM_MFCC)
        mfcc_means = np.mean(mfccs, axis=1)
        mfcc_vars = np.var(mfccs, axis=1)
        
        # 2. Spectral features
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)[0]
        
        # 3. Temporal features
        zero_cross = librosa.feature.zero_crossing_rate(y)[0]
        rms = librosa.feature.rms(y=y)[0]
        
        # 4. Statistics (mean, std, skewness, kurtosis)
        feature_list = [
            np.mean(spec_centroid), np.std(spec_centroid), skew(spec_centroid), kurtosis(spec_centroid),
            np.mean(spec_rolloff), np.std(spec_rolloff), skew(spec_rolloff), kurtosis(spec_rolloff),
            np.mean(spec_contrast), np.std(spec_contrast), skew(spec_contrast), kurtosis(spec_contrast),
            np.mean(zero_cross), np.std(zero_cross), skew(zero_cross), kurtosis(zero_cross),
            np.mean(rms), np.std(rms), skew(rms), kurtosis(rms)
        ]
        
        # 5. Pitch-related features
        pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
        mean_pitches = np.mean(pitches, axis=1)
        mean_magnitudes = np.mean(magnitudes, axis=1)
        pitch_features = [
            np.mean(mean_pitches[mean_pitches > 0]) if np.any(mean_pitches > 0) else 0,
            np.std(mean_pitches[mean_pitches > 0]) if np.any(mean_pitches > 0) else 0,
            np.mean(mean_magnitudes)
        ]
        
        # Combine all features
        all_features = np.concatenate([
            mfcc_means, mfcc_vars,
            feature_list,
            pitch_features
        ])
        
        return all_features
    
    except Exception as e:
        logger.warning(f"Error extracting features from {audio_path}: {e}")
        # Return zero vector as fallback
        return np.zeros(NUM_MFCC * 2 + 20 + 3)  # Match the expected dimension

def load_iemocap(csv_path, audio_base_path=None, use_audio=False):
    """
    Load and preprocess the IEMOCAP dataset.
    
    Args:
        csv_path (str): Path to the IEMOCAP CSV file
        audio_base_path (str, optional): Base path to the audio files
        use_audio (bool): Whether to extract audio features
        
    Returns:
        tuple: (df_model, audio_features) or (df_model, None) if use_audio is False
    """
    logger.info(f"Loading data from {csv_path}...")
    labels_df = pd.read_csv(csv_path)
    
    # Extract relevant columns
    df = labels_df[['Speaker_id', 'Transcript', 'dimension', 'category']].copy()
    if use_audio:
        df['Audio_Uttrance_Path'] = labels_df['Audio_Uttrance_Path']
    
    df['category'] = df['category'].astype(str).str.lower()
    df = df.drop_duplicates(subset='Transcript')
    
    # Create emotion column and filter rows
    df['Emotion'] = df['category'].apply(get_majority_label)
    df_filtered = df.dropna(subset=['Emotion'])
    
    # Map to simplified emotion categories
    df_filtered['Mapped_Emotion'] = df_filtered['Emotion'].map(EMOTION_MAPPING)
    df_final = df_filtered.dropna(subset=['Mapped_Emotion'])
    
    # Parse dimension column
    df_final['parsed_dim'] = df_final['dimension'].apply(smart_parse)
    
    # Flatten VAD scores and prepare final dataframe
    vad_df = pd.json_normalize(df_final['parsed_dim'])
    df_final = df_final.reset_index(drop=True)
    vad_df = vad_df.reset_index(drop=True)
    
    # Combine with transcript
    if use_audio:
        df_model = pd.concat([df_final[['Transcript', 'Mapped_Emotion', 'Audio_Uttrance_Path']], vad_df], axis=1)
    else:
        df_model = pd.concat([df_final[['Transcript', 'Mapped_Emotion']], vad_df], axis=1)
    
    # Extract audio features if needed
    if use_audio and audio_base_path is not None:
        logger.info("Extracting audio features...")
        audio_features = []
        
        from tqdm import tqdm
        for path in tqdm(df_model['Audio_Uttrance_Path'], desc="Processing audio"):
            # Adjust path if needed
            if not os.path.isabs(path):
                full_path = os.path.join(audio_base_path, path)
            else:
                full_path = path
            
            # Extract features or use fallback
            if os.path.exists(full_path):
                features = extract_audio_features(full_path)
            else:
                logger.warning(f"Audio file not found: {full_path}")
                features = np.zeros(NUM_MFCC * 2 + 20 + 3)  # Match the expected dimension
            
            audio_features.append(features)
        
        # Convert to numpy array
        audio_features = np.vstack(audio_features)
        
        logger.info(f"Processed {len(df_model)} samples with valid labels")
        return df_model, audio_features
    
    logger.info(f"Processed {len(df_model)} samples with valid labels (without audio)")
    return df_model, None

def split_dataset(df, audio_features=None, test_size=0.2, val_size=0.25, random_state=42):
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        df (DataFrame): Dataset to split
        audio_features (ndarray, optional): Audio features
        test_size (float): Proportion of data to use for testing
        val_size (float): Proportion of non-test data to use for validation
        random_state (int): Random seed
        
    Returns:
        dict: Dictionary containing split datasets
    """
    from sklearn.model_selection import train_test_split
    
    # Extract features and labels
    X_text = df['Transcript'].values
    y_vad = df[['valence', 'arousal', 'dominance']].values
    y_emotion = df['Mapped_Emotion'].values
    
    # First split: training vs test
    if audio_features is not None:
        # Use indices for splitting to ensure alignment
        indices = np.arange(len(X_text))
        train_val_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=random_state, stratify=y_emotion
        )
        
        # Second split: training vs validation
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=val_size, random_state=random_state,
            stratify=y_emotion[train_val_indices]
        )
        
        # Text data
        X_train_text, X_val_text, X_test_text = X_text[train_indices], X_text[val_indices], X_text[test_indices]
        
        # Audio features
        X_train_audio = audio_features[train_indices]
        X_val_audio = audio_features[val_indices]
        X_test_audio = audio_features[test_indices]
        
        # VAD and emotion labels
        y_train_vad, y_val_vad, y_test_vad = y_vad[train_indices], y_vad[val_indices], y_vad[test_indices]
        y_train_emotion, y_val_emotion, y_test_emotion = y_emotion[train_indices], y_emotion[val_indices], y_emotion[test_indices]
        
        return {
            'X_train_text': X_train_text,
            'X_val_text': X_val_text,
            'X_test_text': X_test_text,
            'X_train_audio': X_train_audio,
            'X_val_audio': X_val_audio,
            'X_test_audio': X_test_audio,
            'y_train_vad': y_train_vad,
            'y_val_vad': y_val_vad,
            'y_test_vad': y_test_vad,
            'y_train_emotion': y_train_emotion,
            'y_val_emotion': y_val_emotion,
            'y_test_emotion': y_test_emotion
        }
    else:
        # Split for text-only approach
        X_train_val, X_test, y_train_val_vad, y_test_vad, y_train_val_emotion, y_test_emotion = train_test_split(
            X_text, y_vad, y_emotion, test_size=test_size, random_state=random_state, stratify=y_emotion
        )
        
        X_train, X_val, y_train_vad, y_val_vad, y_train_emotion, y_val_emotion = train_test_split(
            X_train_val, y_train_val_vad, y_train_val_emotion, test_size=val_size, 
            random_state=random_state, stratify=y_train_val_emotion
        )
        
        return {
            'X_train_text': X_train,
            'X_val_text': X_val,
            'X_test_text': X_test,
            'y_train_vad': y_train_vad,
            'y_val_vad': y_val_vad,
            'y_test_vad': y_test_vad,
            'y_train_emotion': y_train_emotion,
            'y_val_emotion': y_val_emotion,
            'y_test_emotion': y_test_emotion
        }

if __name__ == "__main__":
    # Test data loading
    import argparse
    
    parser = argparse.ArgumentParser(description="Test IEMOCAP data loading")
    parser.add_argument('--data_path', type=str, default='IEMOCAP_Final.csv',
                        help='Path to the IEMOCAP dataset CSV file')
    parser.add_argument('--audio_path', type=str, default=None,
                        help='Base path to audio files')
    parser.add_argument('--use_audio', action='store_true',
                        help='Extract audio features')
    
    args = parser.parse_args()
    
    df, audio_features = load_iemocap(args.data_path, args.audio_path, args.use_audio)
    
    print(f"Loaded dataset with {len(df)} samples")
    print(f"Emotion distribution:")
    print(df['Mapped_Emotion'].value_counts())
    
    if audio_features is not None:
        print(f"Audio features shape: {audio_features.shape}")
        
    # Test splitting
    splits = split_dataset(df, audio_features)
    print("\nDataset splits:")
    for key, value in splits.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {len(value)}") 