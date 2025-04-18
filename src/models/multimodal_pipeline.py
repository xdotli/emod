"""
Multimodal pipeline for emotion recognition.
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
from src.models.audio_processor import AudioVADPredictor
from src.models.emotion_classifier import VADEmotionClassifier
from src.models.multimodal_fusion import EarlyFusion, LateFusion

logger = logging.getLogger(__name__)

class MultimodalEmotionRecognitionPipeline:
    """
    Multimodal pipeline for emotion recognition.
    """
    def __init__(self, text_vad_predictor, audio_vad_predictor, fusion_model, vad_scaler=None):
        """
        Initialize the pipeline.
        
        Args:
            text_vad_predictor: Model for predicting VAD values from text
            audio_vad_predictor: Model for predicting VAD values from audio
            fusion_model: Model for fusing text and audio modalities
            vad_scaler: Scaler for normalizing VAD values
        """
        self.text_vad_predictor = text_vad_predictor
        self.audio_vad_predictor = audio_vad_predictor
        self.fusion_model = fusion_model
        self.vad_scaler = vad_scaler
        
    def predict(self, texts, audio_paths, batch_size=16):
        """
        Predict emotions for a list of texts and audio files.
        
        Args:
            texts (list): List of text strings
            audio_paths (list): List of paths to audio files
            batch_size (int): Batch size for processing
            
        Returns:
            tuple: Tuple containing predicted emotions, text VAD values, and audio VAD values
        """
        # Predict text VAD values
        text_vad = self.text_vad_predictor.predict_vad(texts, batch_size=batch_size)
        
        # Predict audio VAD values
        audio_vad = self.audio_vad_predictor.predict(audio_paths)
        
        # Apply scaler if provided
        if self.vad_scaler:
            text_vad = self.vad_scaler.transform(text_vad)
            audio_vad = self.vad_scaler.transform(audio_vad)
        
        # Predict emotions using fusion model
        emotions = self.fusion_model.predict_emotion_labels(text_vad, audio_vad)
        
        return emotions, text_vad, audio_vad
    
    def evaluate(self, data_loader):
        """
        Evaluate the pipeline on a dataset.
        
        Args:
            data_loader: DataLoader containing the evaluation data
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        all_texts = []
        all_audio_paths = []
        all_true_vad = []
        all_true_emotions = []
        
        # Collect all texts, audio paths, true VAD values, and true emotions
        for batch in tqdm(data_loader, desc="Collecting data"):
            texts = batch['text']
            audio_paths = batch['audio_path']
            vad = batch['vad'].numpy()
            emotions = batch['emotion'].numpy()
            
            all_texts.extend(texts)
            all_audio_paths.extend(audio_paths)
            all_true_vad.append(vad)
            all_true_emotions.append(emotions)
        
        all_true_vad = np.vstack(all_true_vad)
        all_true_emotions = np.concatenate(all_true_emotions)
        
        # Predict text VAD values
        logger.info("Predicting text VAD values")
        all_text_vad = self.text_vad_predictor.predict_vad(all_texts)
        
        # Predict audio VAD values
        logger.info("Predicting audio VAD values")
        all_audio_vad = self.audio_vad_predictor.predict(all_audio_paths)
        
        # Apply scaler if provided
        if self.vad_scaler:
            all_text_vad = self.vad_scaler.transform(all_text_vad)
            all_audio_vad = self.vad_scaler.transform(all_audio_vad)
        
        # Predict emotions using fusion model
        logger.info("Predicting emotions")
        all_pred_emotions = self.fusion_model.predict(all_text_vad, all_audio_vad)
        
        # Calculate VAD metrics for text
        text_vad_mse = np.mean((all_true_vad - all_text_vad) ** 2)
        text_vad_rmse = np.sqrt(text_vad_mse)
        text_vad_mae = np.mean(np.abs(all_true_vad - all_text_vad))
        
        # Calculate VAD metrics for audio
        audio_vad_mse = np.mean((all_true_vad - all_audio_vad) ** 2)
        audio_vad_rmse = np.sqrt(audio_vad_mse)
        audio_vad_mae = np.mean(np.abs(all_true_vad - all_audio_vad))
        
        # Calculate emotion metrics
        emotion_accuracy = accuracy_score(all_true_emotions, all_pred_emotions)
        emotion_f1_macro = f1_score(all_true_emotions, all_pred_emotions, average='macro')
        emotion_f1_weighted = f1_score(all_true_emotions, all_pred_emotions, average='weighted')
        emotion_conf_matrix = confusion_matrix(all_true_emotions, all_pred_emotions)
        
        # Convert to emotion labels
        idx_to_emotion = self.fusion_model.idx_to_emotion
        all_true_emotion_labels = [idx_to_emotion[idx] for idx in all_true_emotions]
        all_pred_emotion_labels = [idx_to_emotion[idx] for idx in all_pred_emotions]
        
        emotion_class_report = classification_report(
            all_true_emotion_labels, 
            all_pred_emotion_labels, 
            output_dict=True
        )
        
        return {
            'text_vad_metrics': {
                'mse': text_vad_mse,
                'rmse': text_vad_rmse,
                'mae': text_vad_mae
            },
            'audio_vad_metrics': {
                'mse': audio_vad_mse,
                'rmse': audio_vad_rmse,
                'mae': audio_vad_mae
            },
            'emotion_metrics': {
                'accuracy': emotion_accuracy,
                'f1_macro': emotion_f1_macro,
                'f1_weighted': emotion_f1_weighted,
                'confusion_matrix': emotion_conf_matrix,
                'classification_report': emotion_class_report
            },
            'true_vad': all_true_vad,
            'text_vad': all_text_vad,
            'audio_vad': all_audio_vad,
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
        
        # Save fusion model
        fusion_dir = os.path.join(output_dir, 'fusion_model')
        os.makedirs(fusion_dir, exist_ok=True)
        
        if isinstance(self.fusion_model, EarlyFusion):
            model_path = os.path.join(fusion_dir, 'early_fusion.pkl')
            mappings_path = os.path.join(fusion_dir, 'emotion_mappings.pkl')
            self.fusion_model.save(model_path, mappings_path)
        elif isinstance(self.fusion_model, LateFusion):
            mappings_path = os.path.join(fusion_dir, 'emotion_mappings.pkl')
            self.fusion_model.save(fusion_dir, mappings_path)
        
        # Save audio VAD predictor
        audio_model_path = os.path.join(output_dir, 'audio_vad_predictor.pkl')
        self.audio_vad_predictor.save_model(audio_model_path)
        
        # Save VAD scaler if provided
        if self.vad_scaler:
            scaler_path = os.path.join(output_dir, 'vad_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                import pickle
                pickle.dump(self.vad_scaler, f)
        
        # Save configuration
        config = {
            'text_vad_model': self.text_vad_predictor.model_name,
            'fusion_type': 'early' if isinstance(self.fusion_model, EarlyFusion) else 'late'
        }
        
        config_path = os.path.join(output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Pipeline components saved to {output_dir}")
    
    @classmethod
    def load(cls, output_dir, text_vad_model_name="facebook/bart-large-mnli", device=None):
        """
        Load a pipeline from disk.
        
        Args:
            output_dir (str): Directory containing the saved components
            text_vad_model_name (str): Name of the pre-trained model for text VAD prediction
            device (str): Device to use for inference ('cuda' or 'cpu')
            
        Returns:
            MultimodalEmotionRecognitionPipeline: The loaded pipeline
        """
        # Load configuration
        config_path = os.path.join(output_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            text_vad_model_name = config.get('text_vad_model', text_vad_model_name)
            fusion_type = config.get('fusion_type', 'early')
        else:
            fusion_type = 'early'
        
        # Load fusion model
        fusion_dir = os.path.join(output_dir, 'fusion_model')
        mappings_path = os.path.join(fusion_dir, 'emotion_mappings.pkl')
        
        if fusion_type == 'early':
            model_path = os.path.join(fusion_dir, 'early_fusion.pkl')
            fusion_model = EarlyFusion.load(model_path, mappings_path)
        else:
            fusion_model = LateFusion.load(fusion_dir, mappings_path)
        
        # Load audio VAD predictor
        audio_model_path = os.path.join(output_dir, 'audio_vad_predictor.pkl')
        audio_vad_predictor = AudioVADPredictor(model_path=audio_model_path)
        
        # Load VAD scaler if exists
        scaler_path = os.path.join(output_dir, 'vad_scaler.pkl')
        vad_scaler = None
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                import pickle
                vad_scaler = pickle.load(f)
        
        # Initialize text VAD predictor
        if 'bart' in text_vad_model_name.lower():
            text_vad_predictor = BARTZeroShotVADPredictor(text_vad_model_name, device)
        else:
            text_vad_predictor = ZeroShotVADPredictor(text_vad_model_name, device)
        
        logger.info(f"Pipeline loaded from {output_dir}")
        
        return cls(text_vad_predictor, audio_vad_predictor, fusion_model, vad_scaler)
