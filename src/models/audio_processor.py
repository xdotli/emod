"""
Audio processing and feature extraction for emotion recognition.
"""

import os
import numpy as np
import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class AudioFeatureExtractor:
    """
    Extract features from audio files for emotion recognition.
    """
    def __init__(self, feature_type="mfcc", sr=16000, n_mfcc=40):
        """
        Initialize the feature extractor.
        
        Args:
            feature_type (str): Type of features to extract ('mfcc', 'spectral', 'wav2vec')
            sr (int): Sample rate for audio processing
            n_mfcc (int): Number of MFCC coefficients to extract
        """
        self.feature_type = feature_type
        self.sr = sr
        self.n_mfcc = n_mfcc
        
        # Initialize Wav2Vec2 model if needed
        if feature_type == "wav2vec":
            logger.info("Initializing Wav2Vec2 model")
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
            
            # Move model to GPU if available
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device)
    
    def extract_features(self, audio_path):
        """
        Extract features from an audio file.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            np.ndarray: Extracted features
        """
        if not os.path.exists(audio_path):
            logger.warning(f"Audio file not found: {audio_path}")
            return None
        
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=self.sr)
            
            # Extract features based on the specified type
            if self.feature_type == "mfcc":
                return self._extract_mfcc(y, sr)
            elif self.feature_type == "spectral":
                return self._extract_spectral(y, sr)
            elif self.feature_type == "wav2vec":
                return self._extract_wav2vec(y, sr)
            else:
                logger.error(f"Unknown feature type: {self.feature_type}")
                return None
        except Exception as e:
            logger.error(f"Error extracting features from {audio_path}: {str(e)}")
            return None
    
    def _extract_mfcc(self, y, sr):
        """
        Extract MFCC features.
        
        Args:
            y (np.ndarray): Audio signal
            sr (int): Sample rate
            
        Returns:
            np.ndarray: MFCC features
        """
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        
        # Compute statistics over time
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        mfcc_max = np.max(mfccs, axis=1)
        mfcc_min = np.min(mfccs, axis=1)
        
        # Concatenate statistics
        features = np.concatenate([mfcc_mean, mfcc_std, mfcc_max, mfcc_min])
        
        return features
    
    def _extract_spectral(self, y, sr):
        """
        Extract spectral features.
        
        Args:
            y (np.ndarray): Audio signal
            sr (int): Sample rate
            
        Returns:
            np.ndarray: Spectral features
        """
        # Extract spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # Compute statistics
        features = []
        
        # Add centroid statistics
        features.extend([np.mean(spectral_centroid), np.std(spectral_centroid)])
        
        # Add bandwidth statistics
        features.extend([np.mean(spectral_bandwidth), np.std(spectral_bandwidth)])
        
        # Add rolloff statistics
        features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])
        
        # Add contrast statistics
        features.extend([np.mean(spectral_contrast), np.std(spectral_contrast)])
        
        # Add zero crossing rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        features.extend([np.mean(zero_crossing_rate), np.std(zero_crossing_rate)])
        
        return np.array(features)
    
    def _extract_wav2vec(self, y, sr):
        """
        Extract Wav2Vec2 features.
        
        Args:
            y (np.ndarray): Audio signal
            sr (int): Sample rate
            
        Returns:
            np.ndarray: Wav2Vec2 features
        """
        # Process audio with Wav2Vec2
        inputs = self.processor(y, sampling_rate=sr, return_tensors="pt")
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get the last hidden state
        last_hidden_state = outputs.last_hidden_state
        
        # Compute mean over time dimension
        features = last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        
        return features

class AudioVADPredictor:
    """
    Predict VAD values from audio features.
    """
    def __init__(self, model_path=None, feature_type="mfcc"):
        """
        Initialize the VAD predictor.
        
        Args:
            model_path (str): Path to the trained model
            feature_type (str): Type of features to extract
        """
        self.feature_extractor = AudioFeatureExtractor(feature_type=feature_type)
        self.model = None
        
        # Load model if path is provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def extract_features_batch(self, audio_paths):
        """
        Extract features from a batch of audio files.
        
        Args:
            audio_paths (list): List of paths to audio files
            
        Returns:
            np.ndarray: Batch of extracted features
        """
        features = []
        
        for path in tqdm(audio_paths, desc="Extracting audio features"):
            feat = self.feature_extractor.extract_features(path)
            if feat is not None:
                features.append(feat)
            else:
                # Use zeros as fallback
                if len(features) > 0:
                    features.append(np.zeros_like(features[0]))
                else:
                    # Assume a default feature size
                    features.append(np.zeros(40 * 4))  # For MFCC with 4 statistics
        
        return np.array(features)
    
    def train(self, audio_paths, vad_values, model_type="ridge", **kwargs):
        """
        Train a model to predict VAD values from audio features.
        
        Args:
            audio_paths (list): List of paths to audio files
            vad_values (np.ndarray): Array of VAD values
            model_type (str): Type of model to train ('ridge', 'svr', 'rf')
            **kwargs: Additional arguments for the model
            
        Returns:
            self: The trained model
        """
        # Extract features
        features = self.extract_features_batch(audio_paths)
        
        # Train model based on the specified type
        if model_type == "ridge":
            from sklearn.linear_model import Ridge
            self.model = Ridge(**kwargs)
        elif model_type == "svr":
            from sklearn.svm import SVR
            self.model = SVR(**kwargs)
        elif model_type == "rf":
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train the model
        self.model.fit(features, vad_values)
        
        return self
    
    def predict(self, audio_paths):
        """
        Predict VAD values for a list of audio files.
        
        Args:
            audio_paths (list): List of paths to audio files
            
        Returns:
            np.ndarray: Array of predicted VAD values
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Extract features
        features = self.extract_features_batch(audio_paths)
        
        # Predict VAD values
        vad_values = self.model.predict(features)
        
        return vad_values
    
    def save_model(self, model_path):
        """
        Save the trained model.
        
        Args:
            model_path (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        import pickle
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """
        Load a trained model.
        
        Args:
            model_path (str): Path to the trained model
        """
        import pickle
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        logger.info(f"Model loaded from {model_path}")

def evaluate_audio_vad_predictor(predictor, audio_paths, true_vad):
    """
    Evaluate the audio VAD predictor.
    
    Args:
        predictor: Trained AudioVADPredictor
        audio_paths (list): List of paths to audio files
        true_vad (np.ndarray): Array of true VAD values
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Predict VAD values
    pred_vad = predictor.predict(audio_paths)
    
    # Calculate metrics
    mse = np.mean((true_vad - pred_vad) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(true_vad - pred_vad))
    
    # Calculate metrics for each dimension
    dim_metrics = {}
    for i, dim in enumerate(['valence', 'arousal', 'dominance']):
        dim_mse = np.mean((true_vad[:, i] - pred_vad[:, i]) ** 2)
        dim_rmse = np.sqrt(dim_mse)
        dim_mae = np.mean(np.abs(true_vad[:, i] - pred_vad[:, i]))
        
        dim_metrics[dim] = {
            'mse': dim_mse,
            'rmse': dim_rmse,
            'mae': dim_mae
        }
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'dim_metrics': dim_metrics,
        'true_vad': true_vad,
        'pred_vad': pred_vad
    }
