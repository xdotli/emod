#!/usr/bin/env python3
"""
Create a smaller sample of the processed data for demonstration purposes.
"""

import os
import pandas as pd
import numpy as np

def create_sample_data(input_file, output_file, sample_size=100, random_seed=42):
    """Create a smaller sample of the data."""
    print(f"Creating sample data from {input_file}")
    
    # Load the data
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows from {input_file}")
    
    # Get a balanced sample of emotions
    emotions = df['emotion'].unique()
    print(f"Found {len(emotions)} unique emotions: {emotions}")
    
    samples_per_emotion = sample_size // len(emotions)
    print(f"Sampling {samples_per_emotion} rows per emotion")
    
    # Create a balanced sample
    sample_df = pd.DataFrame()
    for emotion in emotions:
        emotion_df = df[df['emotion'] == emotion]
        if len(emotion_df) > samples_per_emotion:
            emotion_sample = emotion_df.sample(samples_per_emotion, random_state=random_seed)
        else:
            emotion_sample = emotion_df
        sample_df = pd.concat([sample_df, emotion_sample])
    
    # Shuffle the sample
    sample_df = sample_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Simplify audio features to reduce file size
    if 'audio_features' in sample_df.columns:
        # Replace with smaller random vectors
        for idx in range(len(sample_df)):
            sample_df.at[idx, 'audio_features'] = np.random.randn(10).tolist()
    
    if 'audio_waveform' in sample_df.columns:
        # Replace with smaller random vectors
        for idx in range(len(sample_df)):
            sample_df.at[idx, 'audio_waveform'] = np.random.randn(100).tolist()
    
    # Save the sample
    print(f"Saving {len(sample_df)} rows to {output_file}")
    sample_df.to_csv(output_file, index=False)
    print(f"Sample data saved to {output_file}")
    
    # Print file size
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")

def main():
    """Main function."""
    # Create sample from processed_data.csv
    if os.path.exists("data/processed_data.csv"):
        create_sample_data(
            "data/processed_data.csv", 
            "data/sample_processed_data.csv", 
            sample_size=100
        )
    
    # Create sample from real_iemocap_data.csv
    if os.path.exists("data/real_iemocap_data.csv"):
        create_sample_data(
            "data/real_iemocap_data.csv", 
            "data/sample_real_iemocap_data.csv", 
            sample_size=100
        )
    
    print("Done!")

if __name__ == "__main__":
    main()
