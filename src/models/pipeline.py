"""
End-to-end pipeline for emotion recognition.
"""

import numpy as np
import pandas as pd
import torch
import logging
from tqdm import tqdm
import os
import json
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from src.models.vad_predictor import ZeroShotVADPredictor, BARTZeroShotVADPredictor
from src.models.emotion_classifier import VADEmotionClassifier

logger = logging.getLogger(__name__)

class EmotionRecognitionPipeline:
    """
    End-to-end pipeline for emotion recognition.
    """
    def __init__(self, vad_predictor, emotion_classifier, vad_scaler=None):
        """
        Initialize the pipeline.
        
        Args:
            vad_predictor: Model for predicting VAD values from text
            emotion_classifier: Model for predicting emotions from VAD values
            vad_scaler: Scaler for normalizing VAD values
        """
        self.vad_predictor = vad_predictor
        self.emotion_classifier = emotion_classifier
        self.vad_scaler = vad_scaler
        
    def predict(self, texts, batch_size=16):
        """
        Predict emotions for a list of texts.
        
        Args:
            texts (list): List of text strings
            batch_size (int): Batch size for processing
            
        Returns:
            tuple: Tuple containing predicted emotions and VAD values
        """
        # Predict VAD values
        vad_values = self.vad_predictor.predict_vad(texts, batch_size=batch_size)
        
        # Apply scaler if provided
        if self.vad_scaler:
            vad_values = self.vad_scaler.transform(vad_values)
        
        # Predict emotions
        emotions = self.emotion_classifier.predict_emotion_labels(vad_values)
        
        return emotions, vad_values
    
    def evaluate(self, data_loader):
        """
        Evaluate the pipeline on a dataset.
        
        Args:
            data_loader: DataLoader containing the evaluation data
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        all_texts = []
        all_true_vad = []
        all_true_emotions = []
        
        # Collect all texts, true VAD values, and true emotions
        for batch in tqdm(data_loader, desc="Collecting data"):
            texts = batch['text']
            vad = batch['vad'].numpy()
            emotions = batch['emotion'].numpy()
            
            all_texts.extend(texts)
            all_true_vad.append(vad)
            all_true_emotions.append(emotions)
        
        all_true_vad = np.vstack(all_true_vad)
        all_true_emotions = np.concatenate(all_true_emotions)
        
        # Predict VAD values
        logger.info("Predicting VAD values")
        all_pred_vad = self.vad_predictor.predict_vad(all_texts)
        
        # Apply scaler if provided
        if self.vad_scaler:
            all_pred_vad = self.vad_scaler.transform(all_pred_vad)
        
        # Predict emotions
        logger.info("Predicting emotions")
        all_pred_emotions = self.emotion_classifier.predict(all_pred_vad)
        
        # Calculate VAD metrics
        vad_mse = np.mean((all_true_vad - all_pred_vad) ** 2)
        vad_rmse = np.sqrt(vad_mse)
        vad_mae = np.mean(np.abs(all_true_vad - all_pred_vad))
        
        # Calculate emotion metrics
        emotion_accuracy = accuracy_score(all_true_emotions, all_pred_emotions)
        emotion_f1_macro = f1_score(all_true_emotions, all_pred_emotions, average='macro')
        emotion_f1_weighted = f1_score(all_true_emotions, all_pred_emotions, average='weighted')
        emotion_conf_matrix = confusion_matrix(all_true_emotions, all_pred_emotions)
        
        # Convert to emotion labels
        idx_to_emotion = self.emotion_classifier.idx_to_emotion
        all_true_emotion_labels = [idx_to_emotion[idx] for idx in all_true_emotions]
        all_pred_emotion_labels = [idx_to_emotion[idx] for idx in all_pred_emotions]
        
        emotion_class_report = classification_report(
            all_true_emotion_labels, 
            all_pred_emotion_labels, 
            output_dict=True
        )
        
        return {
            'vad_metrics': {
                'mse': vad_mse,
                'rmse': vad_rmse,
                'mae': vad_mae
            },
            'emotion_metrics': {
                'accuracy': emotion_accuracy,
                'f1_macro': emotion_f1_macro,
                'f1_weighted': emotion_f1_weighted,
                'confusion_matrix': emotion_conf_matrix,
                'classification_report': emotion_class_report
            },
            'true_vad': all_true_vad,
            'pred_vad': all_pred_vad,
            'true_emotions': all_true_emotions,
            'pred_emotions': all_pred_emotions,
            'true_emotion_labels': all_true_emotion_labels,
            'pred_emotion_labels': all_pred_emotion_labels
        }
    
    def save(self, output_dir):
        """
        Save the pipeline components.
        
        Args:
            output_dir (str): Directory to save the components
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save emotion classifier
        classifier_path = os.path.join(output_dir, 'emotion_classifier.pkl')
        mappings_path = os.path.join(output_dir, 'emotion_mappings.pkl')
        self.emotion_classifier.save(classifier_path, mappings_path)
        
        # Save VAD scaler if provided
        if self.vad_scaler:
            scaler_path = os.path.join(output_dir, 'vad_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                import pickle
                pickle.dump(self.vad_scaler, f)
        
        logger.info(f"Pipeline components saved to {output_dir}")
    
    @classmethod
    def load(cls, output_dir, vad_model_name="facebook/bart-large-mnli", device=None):
        """
        Load a pipeline from disk.
        
        Args:
            output_dir (str): Directory containing the saved components
            vad_model_name (str): Name of the pre-trained model for VAD prediction
            device (str): Device to use for inference ('cuda' or 'cpu')
            
        Returns:
            EmotionRecognitionPipeline: The loaded pipeline
        """
        # Load emotion classifier
        classifier_path = os.path.join(output_dir, 'emotion_classifier.pkl')
        mappings_path = os.path.join(output_dir, 'emotion_mappings.pkl')
        emotion_classifier = VADEmotionClassifier.load(classifier_path, mappings_path)
        
        # Load VAD scaler if exists
        scaler_path = os.path.join(output_dir, 'vad_scaler.pkl')
        vad_scaler = None
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                import pickle
                vad_scaler = pickle.load(f)
        
        # Initialize VAD predictor
        if 'bart' in vad_model_name.lower():
            vad_predictor = BARTZeroShotVADPredictor(vad_model_name, device)
        else:
            vad_predictor = ZeroShotVADPredictor(vad_model_name, device)
        
        logger.info(f"Pipeline loaded from {output_dir}")
        
        return cls(vad_predictor, emotion_classifier, vad_scaler)
