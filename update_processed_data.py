#!/usr/bin/env python3
"""
Update the processed_data.csv file to use real text from iemocap_vad.csv.
"""

import os
import pandas as pd
from tqdm import tqdm

def main():
    """Main function."""
    # Load the existing processed_data.csv
    processed_data_path = "data/processed_data.csv"
    if os.path.exists(processed_data_path):
        print(f"Loading existing processed data from {processed_data_path}")
        processed_df = pd.read_csv(processed_data_path)
    else:
        print(f"Processed data file not found: {processed_data_path}")
        return
    
    # Load the real data from iemocap_vad.csv
    iemocap_vad_path = "data/iemocap_vad.csv"
    if os.path.exists(iemocap_vad_path):
        print(f"Loading real IEMOCAP data from {iemocap_vad_path}")
        iemocap_df = pd.read_csv(iemocap_vad_path)
    else:
        print(f"IEMOCAP data file not found: {iemocap_vad_path}")
        return
    
    # Create a mapping from utterance_id to text
    utterance_to_text = dict(zip(iemocap_df['utterance_id'], iemocap_df['text']))
    
    # Update the text in processed_df
    print("Updating text in processed data...")
    updated_count = 0
    for idx, row in tqdm(processed_df.iterrows(), total=len(processed_df)):
        utterance_id = row['utterance_id']
        if utterance_id in utterance_to_text:
            processed_df.at[idx, 'text'] = utterance_to_text[utterance_id]
            updated_count += 1
    
    print(f"Updated {updated_count} out of {len(processed_df)} entries")
    
    # Save the updated processed data
    print(f"Saving updated processed data to {processed_data_path}")
    processed_df.to_csv(processed_data_path, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
