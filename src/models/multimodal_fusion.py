"""
Multimodal fusion for emotion recognition.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
import logging
import os
import pickle

logger = logging.getLogger(__name__)

class EarlyFusion:
    """
    Early fusion of text and audio modalities.
    """
    def __init__(self, classifier_type="rf", random_state=42):
        """
        Initialize the early fusion model.
        
        Args:
            classifier_type (str): Type of classifier to use ('rf', 'svm', 'mlp')
            random_state (int): Random seed for reproducibility
        """
        self.classifier_type = classifier_type
        self.random_state = random_state
        self.classifier = None
        self.emotion_to_idx = None
        self.idx_to_emotion = None
    
    def fit(self, text_vad, audio_vad, emotions, emotion_to_idx=None, idx_to_emotion=None):
        """
        Train the fusion model.
        
        Args:
            text_vad (np.ndarray): VAD values from text
            audio_vad (np.ndarray): VAD values from audio
            emotions (np.ndarray): Emotion labels
            emotion_to_idx (dict): Mapping from emotion labels to indices
            idx_to_emotion (dict): Mapping from indices to emotion labels
            
        Returns:
            self: The trained model
        """
        # Store emotion mappings
        self.emotion_to_idx = emotion_to_idx
        self.idx_to_emotion = idx_to_emotion
        
        # Concatenate features
        features = np.concatenate([text_vad, audio_vad], axis=1)
        
        # Initialize classifier based on the specified type
        if self.classifier_type == "rf":
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.classifier_type == "svm":
            from sklearn.svm import SVC
            self.classifier = SVC(
                probability=True,
                random_state=self.random_state
            )
        elif self.classifier_type == "mlp":
            from sklearn.neural_network import MLPClassifier
            self.classifier = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
        
        # Train the classifier
        self.classifier.fit(features, emotions)
        
        return self
    
    def predict(self, text_vad, audio_vad):
        """
        Predict emotions from VAD values.
        
        Args:
            text_vad (np.ndarray): VAD values from text
            audio_vad (np.ndarray): VAD values from audio
            
        Returns:
            np.ndarray: Predicted emotion indices
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained")
        
        # Concatenate features
        features = np.concatenate([text_vad, audio_vad], axis=1)
        
        # Predict emotions
        return self.classifier.predict(features)
    
    def predict_proba(self, text_vad, audio_vad):
        """
        Predict emotion probabilities from VAD values.
        
        Args:
            text_vad (np.ndarray): VAD values from text
            audio_vad (np.ndarray): VAD values from audio
            
        Returns:
            np.ndarray: Predicted emotion probabilities
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained")
        
        # Concatenate features
        features = np.concatenate([text_vad, audio_vad], axis=1)
        
        # Predict emotion probabilities
        return self.classifier.predict_proba(features)
    
    def predict_emotion_labels(self, text_vad, audio_vad):
        """
        Predict emotion labels from VAD values.
        
        Args:
            text_vad (np.ndarray): VAD values from text
            audio_vad (np.ndarray): VAD values from audio
            
        Returns:
            list: List of predicted emotion labels
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained")
        
        if self.idx_to_emotion is None:
            raise ValueError("idx_to_emotion mapping is not set")
        
        # Predict emotion indices
        y_pred = self.predict(text_vad, audio_vad)
        
        # Convert to emotion labels
        return [self.idx_to_emotion[idx] for idx in y_pred]
    
    def save(self, model_path, mappings_path=None):
        """
        Save the model to disk.
        
        Args:
            model_path (str): Path to save the model
            mappings_path (str): Path to save the emotion mappings
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained")
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.classifier, f)
        
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
        Load a model from disk.
        
        Args:
            model_path (str): Path to the saved model
            mappings_path (str): Path to the saved emotion mappings
            
        Returns:
            EarlyFusion: The loaded model
        """
        instance = cls()
        
        with open(model_path, 'rb') as f:
            instance.classifier = pickle.load(f)
        
        if mappings_path:
            with open(mappings_path, 'rb') as f:
                mappings = pickle.load(f)
                instance.emotion_to_idx = mappings['emotion_to_idx']
                instance.idx_to_emotion = mappings['idx_to_emotion']
        
        logger.info(f"Model loaded from {model_path}")
        
        return instance

class LateFusion:
    """
    Late fusion of text and audio modalities.
    """
    def __init__(self, text_weight=0.7, audio_weight=0.3):
        """
        Initialize the late fusion model.
        
        Args:
            text_weight (float): Weight for text modality
            audio_weight (float): Weight for audio modality
        """
        self.text_weight = text_weight
        self.audio_weight = audio_weight
        self.text_classifier = None
        self.audio_classifier = None
        self.emotion_to_idx = None
        self.idx_to_emotion = None
    
    def fit(self, text_vad, audio_vad, emotions, emotion_to_idx=None, idx_to_emotion=None):
        """
        Train the fusion model.
        
        Args:
            text_vad (np.ndarray): VAD values from text
            audio_vad (np.ndarray): VAD values from audio
            emotions (np.ndarray): Emotion labels
            emotion_to_idx (dict): Mapping from emotion labels to indices
            idx_to_emotion (dict): Mapping from indices to emotion labels
            
        Returns:
            self: The trained model
        """
        # Store emotion mappings
        self.emotion_to_idx = emotion_to_idx
        self.idx_to_emotion = idx_to_emotion
        
        # Train text classifier
        self.text_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        self.text_classifier.fit(text_vad, emotions)
        
        # Train audio classifier
        self.audio_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        self.audio_classifier.fit(audio_vad, emotions)
        
        return self
    
    def predict(self, text_vad, audio_vad):
        """
        Predict emotions from VAD values.
        
        Args:
            text_vad (np.ndarray): VAD values from text
            audio_vad (np.ndarray): VAD values from audio
            
        Returns:
            np.ndarray: Predicted emotion indices
        """
        if self.text_classifier is None or self.audio_classifier is None:
            raise ValueError("Classifiers not trained")
        
        # Get probabilities from each modality
        text_proba = self.text_classifier.predict_proba(text_vad)
        audio_proba = self.audio_classifier.predict_proba(audio_vad)
        
        # Weighted fusion
        fused_proba = self.text_weight * text_proba + self.audio_weight * audio_proba
        
        # Get the most likely class
        return np.argmax(fused_proba, axis=1)
    
    def predict_proba(self, text_vad, audio_vad):
        """
        Predict emotion probabilities from VAD values.
        
        Args:
            text_vad (np.ndarray): VAD values from text
            audio_vad (np.ndarray): VAD values from audio
            
        Returns:
            np.ndarray: Predicted emotion probabilities
        """
        if self.text_classifier is None or self.audio_classifier is None:
            raise ValueError("Classifiers not trained")
        
        # Get probabilities from each modality
        text_proba = self.text_classifier.predict_proba(text_vad)
        audio_proba = self.audio_classifier.predict_proba(audio_vad)
        
        # Weighted fusion
        return self.text_weight * text_proba + self.audio_weight * audio_proba
    
    def predict_emotion_labels(self, text_vad, audio_vad):
        """
        Predict emotion labels from VAD values.
        
        Args:
            text_vad (np.ndarray): VAD values from text
            audio_vad (np.ndarray): VAD values from audio
            
        Returns:
            list: List of predicted emotion labels
        """
        if self.text_classifier is None or self.audio_classifier is None:
            raise ValueError("Classifiers not trained")
        
        if self.idx_to_emotion is None:
            raise ValueError("idx_to_emotion mapping is not set")
        
        # Predict emotion indices
        y_pred = self.predict(text_vad, audio_vad)
        
        # Convert to emotion labels
        return [self.idx_to_emotion[idx] for idx in y_pred]
    
    def save(self, model_dir, mappings_path=None):
        """
        Save the model to disk.
        
        Args:
            model_dir (str): Directory to save the models
            mappings_path (str): Path to save the emotion mappings
        """
        if self.text_classifier is None or self.audio_classifier is None:
            raise ValueError("Classifiers not trained")
        
        os.makedirs(model_dir, exist_ok=True)
        
        text_model_path = os.path.join(model_dir, "text_classifier.pkl")
        audio_model_path = os.path.join(model_dir, "audio_classifier.pkl")
        
        with open(text_model_path, 'wb') as f:
            pickle.dump(self.text_classifier, f)
        
        with open(audio_model_path, 'wb') as f:
            pickle.dump(self.audio_classifier, f)
        
        # Save weights
        weights_path = os.path.join(model_dir, "fusion_weights.pkl")
        with open(weights_path, 'wb') as f:
            pickle.dump({
                'text_weight': self.text_weight,
                'audio_weight': self.audio_weight
            }, f)
        
        if mappings_path and self.emotion_to_idx and self.idx_to_emotion:
            mappings = {
                'emotion_to_idx': self.emotion_to_idx,
                'idx_to_emotion': self.idx_to_emotion
            }
            
            with open(mappings_path, 'wb') as f:
                pickle.dump(mappings, f)
        
        logger.info(f"Models saved to {model_dir}")
    
    @classmethod
    def load(cls, model_dir, mappings_path=None):
        """
        Load a model from disk.
        
        Args:
            model_dir (str): Directory containing the saved models
            mappings_path (str): Path to the saved emotion mappings
            
        Returns:
            LateFusion: The loaded model
        """
        text_model_path = os.path.join(model_dir, "text_classifier.pkl")
        audio_model_path = os.path.join(model_dir, "audio_classifier.pkl")
        weights_path = os.path.join(model_dir, "fusion_weights.pkl")
        
        if not os.path.exists(text_model_path) or not os.path.exists(audio_model_path):
            raise ValueError(f"Model files not found in {model_dir}")
        
        # Load weights if available
        if os.path.exists(weights_path):
            with open(weights_path, 'rb') as f:
                weights = pickle.load(f)
                instance = cls(
                    text_weight=weights['text_weight'],
                    audio_weight=weights['audio_weight']
                )
        else:
            instance = cls()
        
        # Load classifiers
        with open(text_model_path, 'rb') as f:
            instance.text_classifier = pickle.load(f)
        
        with open(audio_model_path, 'rb') as f:
            instance.audio_classifier = pickle.load(f)
        
        # Load mappings if available
        if mappings_path:
            with open(mappings_path, 'rb') as f:
                mappings = pickle.load(f)
                instance.emotion_to_idx = mappings['emotion_to_idx']
                instance.idx_to_emotion = mappings['idx_to_emotion']
        
        logger.info(f"Models loaded from {model_dir}")
        
        return instance
