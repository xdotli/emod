"""
Data processing utilities for the EMOD project.

This module provides functions for loading, preprocessing, and augmenting
emotion recognition data.
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import json
import random
import re

def load_dataset(filepath, format='csv'):
    """
    Load a dataset from a file.
    
    Args:
        filepath (str): Path to the dataset file
        format (str): File format ('csv', 'json', 'npy', 'pkl')
        
    Returns:
        DataFrame or ndarray: Loaded dataset
    """
    if format.lower() == 'csv':
        return pd.read_csv(filepath)
    elif format.lower() == 'json':
        return pd.read_json(filepath)
    elif format.lower() == 'npy':
        return np.load(filepath)
    elif format.lower() == 'pkl' or format.lower() == 'pickle':
        return pd.read_pickle(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")

def save_dataset(data, filepath, format='csv'):
    """
    Save a dataset to a file.
    
    Args:
        data: Dataset to save (DataFrame or ndarray)
        filepath (str): Path to save the dataset
        format (str): File format ('csv', 'json', 'npy', 'pkl')
        
    Returns:
        bool: True if successful
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if format.lower() == 'csv':
        data.to_csv(filepath, index=False)
    elif format.lower() == 'json':
        data.to_json(filepath)
    elif format.lower() == 'npy':
        np.save(filepath, data)
    elif format.lower() == 'pkl' or format.lower() == 'pickle':
        pd.to_pickle(data, filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return True

def preprocess_text(text, lowercase=True, remove_punctuation=True, 
                   remove_numbers=False, remove_stopwords=False):
    """
    Preprocess text data for NLP tasks.
    
    Args:
        text (str): Text to preprocess
        lowercase (bool): Convert to lowercase
        remove_punctuation (bool): Remove punctuation
        remove_numbers (bool): Remove numbers
        remove_stopwords (bool): Remove stopwords
        
    Returns:
        str: Preprocessed text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    if lowercase:
        text = text.lower()
    
    # Remove punctuation
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', '', text)
    
    # Remove numbers
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    
    # Remove stopwords
    if remove_stopwords:
        import nltk
        from nltk.corpus import stopwords
        
        # Download stopwords if not already downloaded
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        stop_words = set(stopwords.words('english'))
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        text = ' '.join(filtered_words)
    
    return text

def encode_labels(labels):
    """
    Encode categorical labels to integers.
    
    Args:
        labels (array-like): Categorical labels
        
    Returns:
        tuple: (encoded_labels, encoder) where encoder is the fitted LabelEncoder
    """
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    return encoded_labels, encoder

def normalize_features(features, method='standard', feature_range=(0, 1)):
    """
    Normalize features using different methods.
    
    Args:
        features (array-like): Features to normalize
        method (str): Normalization method ('standard', 'minmax')
        feature_range (tuple): Range for MinMaxScaler
        
    Returns:
        tuple: (normalized_features, scaler) where scaler is the fitted scaler
    """
    if method.lower() == 'standard':
        scaler = StandardScaler()
    elif method.lower() == 'minmax':
        scaler = MinMaxScaler(feature_range=feature_range)
    else:
        raise ValueError(f"Unsupported normalization method: {method}")
    
    # Reshape if needed
    if len(features.shape) == 1:
        features = features.reshape(-1, 1)
    
    normalized_features = scaler.fit_transform(features)
    return normalized_features, scaler

def split_dataset(X, y, test_size=0.2, val_size=0.1, random_state=42, stratify=None):
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        X (array-like): Features
        y (array-like): Labels
        test_size (float): Proportion of data for test set
        val_size (float): Proportion of data for validation set
        random_state (int): Random seed for reproducibility
        stratify (array-like): Labels to stratify by
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Stratify parameter
    stratify_param = y if stratify else None
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )
    
    # Second split: separate validation set from remaining data
    if val_size > 0:
        # Recalculate validation size as a proportion of the remaining data
        val_size_adjusted = val_size / (1 - test_size)
        
        # Stratify parameter for second split
        stratify_temp = y_temp if stratify else None
        
        # Second split
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=stratify_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        return X_temp, None, X_test, y_temp, None, y_test

def augment_text_data(texts, labels, augmentation_factor=2, methods=None):
    """
    Augment text data for NLP tasks.
    
    Args:
        texts (array-like): Original text samples
        labels (array-like): Corresponding labels
        augmentation_factor (int): Number of augmented samples per original sample
        methods (list): List of augmentation methods to use
            ('synonym_replacement', 'random_insertion', 'random_swap', 'random_deletion')
        
    Returns:
        tuple: (augmented_texts, augmented_labels)
    """
    try:
        import nlpaug.augmenter.word as naw
        import nlpaug.flow as naf
    except ImportError:
        raise ImportError("nlpaug package is required for text augmentation. Install it with: pip install nlpaug")
    
    # Default methods
    if methods is None:
        methods = ['synonym_replacement', 'random_swap']
    
    # Create augmenters based on specified methods
    augmenters = []
    for method in methods:
        if method == 'synonym_replacement':
            augmenters.append(naw.SynonymAug(aug_src='wordnet'))
        elif method == 'random_insertion':
            augmenters.append(naw.RandomWordAug(action="insert"))
        elif method == 'random_swap':
            augmenters.append(naw.RandomWordAug(action="swap"))
        elif method == 'random_deletion':
            augmenters.append(naw.RandomWordAug(action="delete"))
    
    # Create augmentation flow
    aug_flow = naf.Sequential(augmenters)
    
    # Augment data
    augmented_texts = []
    augmented_labels = []
    
    for i, text in enumerate(texts):
        augmented_texts.append(text)  # Add original text
        augmented_labels.append(labels[i])  # Add original label
        
        # Generate augmented samples
        for _ in range(augmentation_factor):
            augmented_text = aug_flow.augment(text)
            augmented_texts.append(augmented_text)
            augmented_labels.append(labels[i])
    
    return augmented_texts, augmented_labels

def balance_dataset(X, y, method='oversample', random_state=42):
    """
    Balance an imbalanced dataset.
    
    Args:
        X (array-like): Features
        y (array-like): Labels
        method (str): Balancing method ('oversample', 'undersample', 'smote')
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_balanced, y_balanced)
    """
    if method.lower() == 'oversample':
        from imblearn.over_sampling import RandomOverSampler
        balancer = RandomOverSampler(random_state=random_state)
    elif method.lower() == 'undersample':
        from imblearn.under_sampling import RandomUnderSampler
        balancer = RandomUnderSampler(random_state=random_state)
    elif method.lower() == 'smote':
        from imblearn.over_sampling import SMOTE
        balancer = SMOTE(random_state=random_state)
    else:
        raise ValueError(f"Unsupported balancing method: {method}")
    
    # Reshape if needed
    X_reshaped = X
    if hasattr(X, 'shape') and len(X.shape) == 1:
        X_reshaped = X.reshape(-1, 1)
    
    # Balance dataset
    X_balanced, y_balanced = balancer.fit_resample(X_reshaped, y)
    
    return X_balanced, y_balanced

def convert_emotion_to_vad(emotions, emotion_vad_mapping=None):
    """
    Convert categorical emotion labels to VAD (Valence, Arousal, Dominance) values.
    
    Args:
        emotions (array-like): Categorical emotion labels
        emotion_vad_mapping (dict): Mapping from emotions to VAD values
        
    Returns:
        ndarray: VAD values (shape: [n_samples, 3])
    """
    # Default mapping based on Mehrabian's PAD emotional state model
    if emotion_vad_mapping is None:
        emotion_vad_mapping = {
            'happy': [0.8, 0.7, 0.6],     # High valence, high arousal, high dominance
            'sad': [0.2, 0.2, 0.2],       # Low valence, low arousal, low dominance
            'angry': [0.2, 0.8, 0.7],     # Low valence, high arousal, high dominance
            'fear': [0.2, 0.8, 0.2],      # Low valence, high arousal, low dominance
            'surprise': [0.6, 0.8, 0.4],  # Medium valence, high arousal, medium dominance
            'disgust': [0.2, 0.6, 0.5],   # Low valence, medium arousal, medium dominance
            'neutral': [0.5, 0.5, 0.5]    # Medium valence, medium arousal, medium dominance
        }
    
    # Convert emotions to VAD values
    vad_values = []
    for emotion in emotions:
        emotion = emotion.lower() if isinstance(emotion, str) else emotion
        if emotion in emotion_vad_mapping:
            vad_values.append(emotion_vad_mapping[emotion])
        else:
            # Default to neutral for unknown emotions
            vad_values.append(emotion_vad_mapping.get('neutral', [0.5, 0.5, 0.5]))
    
    return np.array(vad_values)

def convert_vad_to_emotion(vad_values, emotion_vad_mapping=None, method='closest'):
    """
    Convert VAD (Valence, Arousal, Dominance) values to categorical emotion labels.
    
    Args:
        vad_values (array-like): VAD values (shape: [n_samples, 3])
        emotion_vad_mapping (dict): Mapping from emotions to VAD values
        method (str): Conversion method ('closest', 'threshold')
        
    Returns:
        list: Categorical emotion labels
    """
    # Default mapping based on Mehrabian's PAD emotional state model
    if emotion_vad_mapping is None:
        emotion_vad_mapping = {
            'happy': [0.8, 0.7, 0.6],     # High valence, high arousal, high dominance
            'sad': [0.2, 0.2, 0.2],       # Low valence, low arousal, low dominance
            'angry': [0.2, 0.8, 0.7],     # Low valence, high arousal, high dominance
            'fear': [0.2, 0.8, 0.2],      # Low valence, high arousal, low dominance
            'surprise': [0.6, 0.8, 0.4],  # Medium valence, high arousal, medium dominance
            'disgust': [0.2, 0.6, 0.5],   # Low valence, medium arousal, medium dominance
            'neutral': [0.5, 0.5, 0.5]    # Medium valence, medium arousal, medium dominance
        }
    
    emotions = []
    vad_values = np.array(vad_values)
    
    if method == 'closest':
        # Find closest emotion for each VAD value using Euclidean distance
        for vad in vad_values:
            min_distance = float('inf')
            closest_emotion = None
            
            for emotion, emotion_vad in emotion_vad_mapping.items():
                distance = np.linalg.norm(np.array(vad) - np.array(emotion_vad))
                
                if distance < min_distance:
                    min_distance = distance
                    closest_emotion = emotion
            
            emotions.append(closest_emotion)
    
    elif method == 'threshold':
        # Use thresholds for each dimension
        for vad in vad_values:
            v, a, d = vad
            
            if v > 0.6:  # High valence
                if a > 0.6:  # High arousal
                    emotions.append('happy')
                else:  # Low/medium arousal
                    emotions.append('neutral')
            elif v < 0.4:  # Low valence
                if a > 0.6:  # High arousal
                    if d > 0.6:  # High dominance
                        emotions.append('angry')
                    else:  # Low/medium dominance
                        emotions.append('fear')
                elif a < 0.4:  # Low arousal
                    emotions.append('sad')
                else:  # Medium arousal
                    emotions.append('disgust')
            else:  # Medium valence
                if a > 0.6:  # High arousal
                    emotions.append('surprise')
                else:  # Low/medium arousal
                    emotions.append('neutral')
    
    return emotions

def extract_features_from_text(texts, method='tfidf', max_features=1000):
    """
    Extract features from text data.
    
    Args:
        texts (array-like): Text samples
        method (str): Feature extraction method ('tfidf', 'count', 'doc2vec', 'bert')
        max_features (int): Maximum number of features (for tfidf and count)
        
    Returns:
        tuple: (features, vectorizer/model) where the second element is the fitted feature extractor
    """
    if method == 'tfidf':
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=max_features)
        features = vectorizer.fit_transform(texts)
        return features, vectorizer
    
    elif method == 'count':
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer(max_features=max_features)
        features = vectorizer.fit_transform(texts)
        return features, vectorizer
    
    elif method == 'doc2vec':
        try:
            from gensim.models.doc2vec import Doc2Vec, TaggedDocument
        except ImportError:
            raise ImportError("gensim package is required for Doc2Vec. Install it with: pip install gensim")
        
        # Preprocess and tag documents
        tagged_data = [TaggedDocument(words=text.split(), tags=[i]) for i, text in enumerate(texts)]
        
        # Train Doc2Vec model
        model = Doc2Vec(vector_size=100, min_count=2, epochs=30)
        model.build_vocab(tagged_data)
        model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
        
        # Extract features
        features = np.array([model.infer_vector(text.split()) for text in texts])
        
        return features, model
    
    elif method == 'bert':
        try:
            from transformers import BertTokenizer, BertModel
            import torch
        except ImportError:
            raise ImportError("transformers and torch packages are required for BERT. Install them with: pip install transformers torch")
        
        # Load pre-trained model and tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        
        # Tokenize and extract features
        features = []
        for text in texts:
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Use CLS token embedding as the feature vector
            features.append(outputs.last_hidden_state[:, 0, :].numpy().flatten())
        
        features = np.array(features)
        extractor = {'tokenizer': tokenizer, 'model': model}
        
        return features, extractor
    
    else:
        raise ValueError(f"Unsupported feature extraction method: {method}")

def extract_vad_features(data, text_column, label_column=None, include_text_features=True):
    """
    Extract features specifically for VAD prediction.
    
    Args:
        data (DataFrame): Dataset containing text and optional labels
        text_column (str): Column name containing text data
        label_column (str): Column name containing emotion labels
        include_text_features (bool): Whether to include text-based features
        
    Returns:
        tuple: (features, labels) where labels are VAD values if label_column is provided
    """
    # Extract text features
    text_features = None
    if include_text_features:
        texts = data[text_column].tolist()
        text_features, _ = extract_features_from_text(texts, method='tfidf', max_features=500)
    
    # Convert emotion labels to VAD values if label_column is provided
    vad_labels = None
    if label_column is not None and label_column in data.columns:
        emotions = data[label_column].tolist()
        vad_labels = convert_emotion_to_vad(emotions)
    
    # Return features and labels
    if text_features is not None:
        return text_features, vad_labels
    else:
        return None, vad_labels 