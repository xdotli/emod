#!/usr/bin/env python3
"""
Create a new dataset using the real IEMOCAP data.
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def main():
    """Main function."""
    # Load the real data from iemocap_vad.csv
    iemocap_vad_path = "data/iemocap_vad.csv"
    if os.path.exists(iemocap_vad_path):
        print(f"Loading real IEMOCAP data from {iemocap_vad_path}")
        iemocap_df = pd.read_csv(iemocap_vad_path)
    else:
        print(f"IEMOCAP data file not found: {iemocap_vad_path}")
        return
    
    # Create a new DataFrame with the real data
    print("Creating new dataset with real data...")
    
    # Get the audio feature dimension from the existing processed_data.csv
    processed_data_path = "data/processed_data.csv"
    if os.path.exists(processed_data_path):
        print(f"Loading existing processed data to get audio feature dimension")
        processed_df = pd.read_csv(processed_data_path)
        # Get the first row's audio_features
        first_row = processed_df.iloc[0]
        if 'audio_features' in first_row:
            try:
                # Convert string representation to list
                audio_features_str = first_row['audio_features']
                if isinstance(audio_features_str, str):
                    # Remove brackets and split by whitespace
                    audio_features_str = audio_features_str.strip('[]')
                    audio_features_str = audio_features_str.replace('\n', ' ')
                    audio_features = [float(x) for x in audio_features_str.split() if x.strip()]
                    audio_feature_dim = len(audio_features)
                else:
                    # Default dimension
                    audio_feature_dim = 100
            except:
                audio_feature_dim = 100
        else:
            audio_feature_dim = 100
    else:
        audio_feature_dim = 100
    
    print(f"Using audio feature dimension: {audio_feature_dim}")
    
    # Create a new DataFrame with the real data
    new_data = []
    for idx, row in tqdm(iemocap_df.iterrows(), total=len(iemocap_df)):
        # Map emotion to standardized format
        emotion = row['emotion']
        if emotion in ["neu", "neutral"]:
            std_emotion = "neutral"
        elif emotion in ["ang", "angry"]:
            std_emotion = "angry"
        elif emotion in ["hap", "happy", "exc", "excited"]:
            std_emotion = "happy"
        elif emotion in ["sad"]:
            std_emotion = "sad"
        else:
            # Skip other emotions for this demo
            continue
        
        # Create a new entry with real text and emotion, but synthetic audio features
        new_entry = {
            'utterance_id': row['utterance_id'],
            'valence': row['valence'],
            'arousal': row['arousal'],
            'dominance': row['dominance'],
            'emotion': std_emotion,
            'text': row['text'],
            'audio_features': np.random.randn(audio_feature_dim).tolist(),
            'audio_waveform': np.zeros(16000).tolist()  # Placeholder
        }
        new_data.append(new_entry)
    
    # Create a new DataFrame
    new_df = pd.DataFrame(new_data)
    
    # Save the new dataset
    new_dataset_path = "data/real_iemocap_data.csv"
    print(f"Saving new dataset to {new_dataset_path}")
    new_df.to_csv(new_dataset_path, index=False)
    
    # Also save as processed_data.csv to be used by the existing code
    print(f"Saving new dataset as {processed_data_path}")
    new_df.to_csv(processed_data_path, index=False)
    
    print(f"Created new dataset with {len(new_df)} entries")
    print("Done!")

if __name__ == "__main__":
    main()
