"""
Text-to-VAD model using pre-trained transformers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import logging
import os
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ZeroShotVADPredictor:
    """
    Zero-shot VAD predictor using pre-trained language models.
    """
    def __init__(self, model_name="roberta-base", device=None):
        """
        Initialize the VAD predictor.
        
        Args:
            model_name (str): Name of the pre-trained model
            device (str): Device to use for inference ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Initializing ZeroShotVADPredictor with {model_name} on {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # Define VAD prompts
        self.vad_prompts = {
            'valence': [
                "This text expresses negative emotions.",
                "This text expresses positive emotions."
            ],
            'arousal': [
                "This text expresses calm emotions.",
                "This text expresses excited emotions."
            ],
            'dominance': [
                "This text expresses submissive emotions.",
                "This text expresses dominant emotions."
            ]
        }
        
    def predict_vad(self, texts, batch_size=16):
        """
        Predict VAD values for a list of texts.
        
        Args:
            texts (list): List of text strings
            batch_size (int): Batch size for processing
            
        Returns:
            np.ndarray: Array of shape (len(texts), 3) containing VAD values
        """
        vad_values = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_vad = self._predict_batch(batch_texts)
            vad_values.append(batch_vad)
        
        return np.vstack(vad_values)
    
    def _predict_batch(self, texts):
        """
        Predict VAD values for a batch of texts.
        
        Args:
            texts (list): List of text strings
            
        Returns:
            np.ndarray: Array of shape (len(texts), 3) containing VAD values
        """
        batch_vad = np.zeros((len(texts), 3))
        
        for dim_idx, (dim_name, prompts) in enumerate(self.vad_prompts.items()):
            # Create text pairs for each dimension
            text_pairs = []
            for text in texts:
                for prompt in prompts:
                    text_pairs.append((text, prompt))
            
            # Tokenize text pairs
            encoded_inputs = self.tokenizer(
                [pair[0] for pair in text_pairs],
                [pair[1] for pair in text_pairs],
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Get model outputs
            with torch.no_grad():
                outputs = self.model(**encoded_inputs)
                
            # Get embeddings for the [CLS] token
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            
            # Calculate similarity scores
            for i, text in enumerate(texts):
                # Get embeddings for the two prompts
                idx1 = i * 2
                idx2 = i * 2 + 1
                
                emb1 = cls_embeddings[idx1]
                emb2 = cls_embeddings[idx2]
                
                # Calculate cosine similarity
                sim1 = F.cosine_similarity(emb1.unsqueeze(0), cls_embeddings[idx1].unsqueeze(0)).item()
                sim2 = F.cosine_similarity(emb2.unsqueeze(0), cls_embeddings[idx2].unsqueeze(0)).item()
                
                # Normalize to [1, 5] range (IEMOCAP VAD scale)
                score = 1 + 4 * (sim2 / (sim1 + sim2))
                batch_vad[i, dim_idx] = score
        
        return batch_vad

class BARTZeroShotVADPredictor:
    """
    Zero-shot VAD predictor using BART for natural language inference.
    """
    def __init__(self, model_name="facebook/bart-large-mnli", device=None):
        """
        Initialize the VAD predictor.
        
        Args:
            model_name (str): Name of the pre-trained model
            device (str): Device to use for inference ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Initializing BARTZeroShotVADPredictor with {model_name} on {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        
        # Define VAD hypotheses
        self.vad_hypotheses = {
            'valence': [
                "The text expresses negative emotions.",
                "The text expresses neutral emotions.",
                "The text expresses positive emotions."
            ],
            'arousal': [
                "The text expresses calm emotions.",
                "The text expresses moderate arousal.",
                "The text expresses excited emotions."
            ],
            'dominance': [
                "The text expresses submissive emotions.",
                "The text expresses neutral dominance.",
                "The text expresses dominant emotions."
            ]
        }
        
    def predict_vad(self, texts, batch_size=16):
        """
        Predict VAD values for a list of texts.
        
        Args:
            texts (list): List of text strings
            batch_size (int): Batch size for processing
            
        Returns:
            np.ndarray: Array of shape (len(texts), 3) containing VAD values
        """
        vad_values = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Predicting VAD"):
            batch_texts = texts[i:i+batch_size]
            batch_vad = self._predict_batch(batch_texts)
            vad_values.append(batch_vad)
        
        return np.vstack(vad_values)
    
    def _predict_batch(self, texts):
        """
        Predict VAD values for a batch of texts.
        
        Args:
            texts (list): List of text strings
            
        Returns:
            np.ndarray: Array of shape (len(texts), 3) containing VAD values
        """
        batch_vad = np.zeros((len(texts), 3))
        
        for dim_idx, (dim_name, hypotheses) in enumerate(self.vad_hypotheses.items()):
            for text_idx, text in enumerate(texts):
                scores = []
                
                for hypothesis in hypotheses:
                    # Tokenize premise-hypothesis pair
                    encoded_input = self.tokenizer(
                        text,
                        hypothesis,
                        padding=True,
                        truncation=True,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    # Get model outputs
                    with torch.no_grad():
                        output = self.model(**encoded_input)
                    
                    # Get entailment score (last element of logits)
                    entailment_score = output.logits[0, 2].item()
                    scores.append(entailment_score)
                
                # Softmax the scores
                scores = np.exp(scores) / np.sum(np.exp(scores))
                
                # Calculate weighted average (1 for negative/calm/submissive, 
                # 3 for neutral, 5 for positive/excited/dominant)
                weighted_score = 1 * scores[0] + 3 * scores[1] + 5 * scores[2]
                batch_vad[text_idx, dim_idx] = weighted_score
        
        return batch_vad

def evaluate_vad_predictor(predictor, data_loader, vad_scaler=None):
    """
    Evaluate the VAD predictor on a dataset.
    
    Args:
        predictor: VAD predictor object
        data_loader: DataLoader containing the evaluation data
        vad_scaler: Scaler used to normalize VAD values
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    all_texts = []
    all_true_vad = []
    
    # Collect all texts and true VAD values
    for batch in data_loader:
        texts = batch['text']
        vad = batch['vad'].numpy()
        
        all_texts.extend(texts)
        all_true_vad.append(vad)
    
    all_true_vad = np.vstack(all_true_vad)
    
    # Predict VAD values
    all_pred_vad = predictor.predict_vad(all_texts)
    
    # Inverse transform if scaler is provided
    if vad_scaler:
        all_pred_vad = vad_scaler.inverse_transform(all_pred_vad)
    
    # Calculate metrics
    mse = np.mean((all_true_vad - all_pred_vad) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_true_vad - all_pred_vad))
    
    # Calculate metrics for each dimension
    dim_metrics = {}
    for i, dim in enumerate(['valence', 'arousal', 'dominance']):
        dim_mse = np.mean((all_true_vad[:, i] - all_pred_vad[:, i]) ** 2)
        dim_rmse = np.sqrt(dim_mse)
        dim_mae = np.mean(np.abs(all_true_vad[:, i] - all_pred_vad[:, i]))
        
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
        'true_vad': all_true_vad,
        'pred_vad': all_pred_vad
    }
