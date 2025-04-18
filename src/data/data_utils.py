"""
Utility functions for data handling.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class IEMOCAPDataset(Dataset):
    """
    Dataset class for IEMOCAP data.
    """
    def __init__(self, data, emotion_to_idx=None, include_audio=False, transform=None):
        """
        Initialize the dataset.

        Args:
            data (pd.DataFrame): DataFrame containing the data
            emotion_to_idx (dict): Mapping from emotion labels to indices
            include_audio (bool): Whether to include audio paths in the samples
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data = data
        self.transform = transform
        self.emotion_to_idx = emotion_to_idx
        self.include_audio = include_audio

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        text = self.data.iloc[idx]['Transcript']
        vad = self.data.iloc[idx][['valence', 'arousal', 'dominance']].values.astype(np.float32)

        # Create sample dictionary
        sample = {'text': text, 'vad': vad}

        # Add audio path if requested
        if self.include_audio and 'Audio_Uttrance_Path' in self.data.columns:
            sample['audio_path'] = self.data.iloc[idx]['Audio_Uttrance_Path']

        # Convert emotion to index if mapping is provided
        if self.emotion_to_idx:
            emotion = self.data.iloc[idx]['emotion']
            # Handle unknown emotions
            if emotion in self.emotion_to_idx:
                emotion_idx = self.emotion_to_idx[emotion]
            else:
                emotion_idx = self.emotion_to_idx['unknown']
            sample['emotion'] = emotion_idx

        if self.transform:
            sample = self.transform(sample)

        return sample

def create_dataloaders(data_splits, emotion_to_idx=None, include_audio=False, batch_size=32, num_workers=4):
    """
    Create DataLoader objects for train, validation, and test sets.

    Args:
        data_splits (dict): Dictionary containing train, val, and test DataFrames
        emotion_to_idx (dict): Mapping from emotion labels to indices
        batch_size (int): Batch size for DataLoader
        num_workers (int): Number of workers for DataLoader

    Returns:
        dict: Dictionary containing train, val, and test DataLoader objects
    """
    dataloaders = {}

    for split_name, df in data_splits.items():
        dataset = IEMOCAPDataset(df, emotion_to_idx, include_audio=include_audio)
        shuffle = split_name == 'train'

        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )

    return dataloaders

def get_class_weights(data, emotion_to_idx):
    """
    Calculate class weights for imbalanced dataset.

    Args:
        data (pd.DataFrame): DataFrame containing the data
        emotion_to_idx (dict): Mapping from emotion labels to indices

    Returns:
        torch.Tensor: Tensor containing class weights
    """
    class_counts = data['emotion'].value_counts().to_dict()
    total_samples = len(data)

    # Calculate weights as inverse of frequency
    weights = []
    for emotion in emotion_to_idx.keys():
        count = class_counts.get(emotion, 0)
        if count == 0:
            weight = 1.0
        else:
            weight = total_samples / (len(emotion_to_idx) * count)
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float)
