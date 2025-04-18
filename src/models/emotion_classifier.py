"""
VAD-to-emotion classifier.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import pickle
import os
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class VADEmotionClassifier:
    """
    Classifier for predicting emotions from VAD values.
    """
    def __init__(self, n_estimators=100, random_state=42):
        """
        Initialize the classifier.
        
        Args:
            n_estimators (int): Number of trees in the random forest
            random_state (int): Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        self.emotion_to_idx = None
        self.idx_to_emotion = None
        
    def fit(self, X, y, emotion_to_idx=None, idx_to_emotion=None):
        """
        Train the classifier.
        
        Args:
            X (np.ndarray): Array of VAD values
            y (np.ndarray): Array of emotion indices
            emotion_to_idx (dict): Mapping from emotion labels to indices
            idx_to_emotion (dict): Mapping from indices to emotion labels
            
        Returns:
            self: The trained classifier
        """
        logger.info(f"Training VADEmotionClassifier with {self.n_estimators} trees")
        
        self.emotion_to_idx = emotion_to_idx
        self.idx_to_emotion = idx_to_emotion
        
        # Train the model
        self.model.fit(X, y)
        
        # Get feature importances
        importances = self.model.feature_importances_
        logger.info(f"Feature importances: valence={importances[0]:.4f}, arousal={importances[1]:.4f}, dominance={importances[2]:.4f}")
        
        return self
    
    def predict(self, X):
        """
        Predict emotion indices from VAD values.
        
        Args:
            X (np.ndarray): Array of VAD values
            
        Returns:
            np.ndarray: Array of predicted emotion indices
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict emotion probabilities from VAD values.
        
        Args:
            X (np.ndarray): Array of VAD values
            
        Returns:
            np.ndarray: Array of predicted emotion probabilities
        """
        return self.model.predict_proba(X)
    
    def predict_emotion_labels(self, X):
        """
        Predict emotion labels from VAD values.
        
        Args:
            X (np.ndarray): Array of VAD values
            
        Returns:
            list: List of predicted emotion labels
        """
        if self.idx_to_emotion is None:
            raise ValueError("idx_to_emotion mapping is not set")
        
        y_pred = self.predict(X)
        return [self.idx_to_emotion[idx] for idx in y_pred]
    
    def save(self, model_path, mappings_path=None):
        """
        Save the classifier to disk.
        
        Args:
            model_path (str): Path to save the model
            mappings_path (str): Path to save the emotion mappings
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        if mappings_path and self.emotion_to_idx and self.idx_to_emotion:
            mappings = {
                'emotion_to_idx': self.emotion_to_idx,
                'idx_to_emotion': self.idx_to_emotion
            }
            
            with open(mappings_path, 'wb') as f:
                pickle.dump(mappings, f)
        
        logger.info(f"Model saved to {model_path}")
    
    @classmethod
    def load(cls, model_path, mappings_path=None):
        """
        Load a classifier from disk.
        
        Args:
            model_path (str): Path to the saved model
            mappings_path (str): Path to the saved emotion mappings
            
        Returns:
            VADEmotionClassifier: The loaded classifier
        """
        instance = cls()
        
        with open(model_path, 'rb') as f:
            instance.model = pickle.load(f)
        
        if mappings_path:
            with open(mappings_path, 'rb') as f:
                mappings = pickle.load(f)
                instance.emotion_to_idx = mappings['emotion_to_idx']
                instance.idx_to_emotion = mappings['idx_to_emotion']
        
        logger.info(f"Model loaded from {model_path}")
        
        return instance

def evaluate_emotion_classifier(classifier, X, y_true, idx_to_emotion=None):
    """
    Evaluate the emotion classifier.
    
    Args:
        classifier: Trained classifier
        X (np.ndarray): Array of VAD values
        y_true (np.ndarray): Array of true emotion indices
        idx_to_emotion (dict): Mapping from indices to emotion labels
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Predict emotions
    y_pred = classifier.predict(X)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Convert to emotion labels if mapping is provided
    if idx_to_emotion:
        y_true_labels = [idx_to_emotion[idx] for idx in y_true]
        y_pred_labels = [idx_to_emotion[idx] for idx in y_pred]
        class_report = classification_report(y_true_labels, y_pred_labels, output_dict=True)
    else:
        class_report = classification_report(y_true, y_pred, output_dict=True)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
    }
