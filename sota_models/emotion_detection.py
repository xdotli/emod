"""
State-of-the-art emotion detection models for text and audio.
"""

import os
import logging
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

from utils.config import TEXT_EMOTION_MODEL, AUDIO_EMOTION_MODEL, HF_TOKEN

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextEmotionDetector:
    """Text-based emotion detection using state-of-the-art models."""
    
    def __init__(self, model_name: str = TEXT_EMOTION_MODEL, use_cuda: bool = True):
        """Initialize the text emotion detector.
        
        Args:
            model_name: Name or path of the model to use
            use_cuda: Whether to use CUDA for inference
        """
        self.model_name = model_name
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            logger.info(f"Using CUDA for text emotion model")
        else:
            logger.info(f"Using CPU for text emotion model")
        
        # Load the model
        logger.info(f"Loading text emotion model: {model_name}")
        try:
            self.classifier = pipeline(
                "text-classification",
                model=model_name,
                tokenizer=model_name,
                device=self.device,
                token=HF_TOKEN
            )
            logger.info(f"Text emotion model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading text emotion model: {e}")
            raise
    
    def detect_emotion(self, text: str) -> Dict[str, Any]:
        """Detect emotions in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with emotion detection results
        """
        logger.info(f"Detecting emotions in text: {text[:50]}...")
        
        try:
            # Perform emotion detection
            result = self.classifier(text)
            
            # Format the result
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
            
            emotion = {
                "label": result["label"],
                "score": result["score"],
                "model": self.model_name
            }
            
            logger.info(f"Emotion detection completed: {emotion['label']} ({emotion['score']:.4f})")
            return emotion
            
        except Exception as e:
            logger.error(f"Error during emotion detection: {e}")
            raise


class AudioEmotionDetector:
    """Audio-based emotion detection using state-of-the-art models."""
    
    def __init__(self, model_name: str = AUDIO_EMOTION_MODEL, use_cuda: bool = True):
        """Initialize the audio emotion detector.
        
        Args:
            model_name: Name or path of the model to use
            use_cuda: Whether to use CUDA for inference
        """
        self.model_name = model_name
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            logger.info(f"Using CUDA for audio emotion model")
        else:
            logger.info(f"Using CPU for audio emotion model")
        
        # Load the model
        logger.info(f"Loading audio emotion model: {model_name}")
        try:
            self.classifier = pipeline(
                "audio-classification",
                model=model_name,
                device=self.device,
                token=HF_TOKEN
            )
            logger.info(f"Audio emotion model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading audio emotion model: {e}")
            raise
    
    def detect_emotion(self, audio_path: Union[str, Path]) -> Dict[str, Any]:
        """Detect emotions in audio.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary with emotion detection results
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Detecting emotions in audio file: {audio_path}")
        
        try:
            # Perform emotion detection
            result = self.classifier(audio_path)
            
            # Format the result
            emotions = []
            for item in result:
                emotions.append({
                    "label": item["label"],
                    "score": item["score"]
                })
            
            # Sort by score in descending order
            emotions.sort(key=lambda x: x["score"], reverse=True)
            
            detection_result = {
                "primary_emotion": emotions[0],
                "all_emotions": emotions,
                "model": self.model_name
            }
            
            logger.info(f"Audio emotion detection completed: {emotions[0]['label']} ({emotions[0]['score']:.4f})")
            return detection_result
            
        except Exception as e:
            logger.error(f"Error during audio emotion detection: {e}")
            raise


class MultimodalEmotionDetector:
    """Multimodal emotion detection combining text and audio analysis."""
    
    def __init__(
        self,
        text_model: str = TEXT_EMOTION_MODEL,
        audio_model: str = AUDIO_EMOTION_MODEL,
        use_cuda: bool = True
    ):
        """Initialize the multimodal emotion detector.
        
        Args:
            text_model: Name or path of the text model to use
            audio_model: Name or path of the audio model to use
            use_cuda: Whether to use CUDA for inference
        """
        self.text_detector = TextEmotionDetector(text_model, use_cuda)
        self.audio_detector = AudioEmotionDetector(audio_model, use_cuda)
    
    def detect_emotion(
        self,
        text: Optional[str] = None,
        audio_path: Optional[Union[str, Path]] = None,
        fusion_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Detect emotions using both text and audio modalities.
        
        Args:
            text: Text to analyze
            audio_path: Path to the audio file
            fusion_weights: Weights for fusion (e.g., {"text": 0.7, "audio": 0.3})
            
        Returns:
            Dictionary with multimodal emotion detection results
        """
        if text is None and audio_path is None:
            raise ValueError("At least one of text or audio_path must be provided")
        
        # Default fusion weights
        if fusion_weights is None:
            fusion_weights = {"text": 0.6, "audio": 0.4}
        
        results = {}
        
        # Detect emotions in text
        if text is not None:
            results["text"] = self.text_detector.detect_emotion(text)
        
        # Detect emotions in audio
        if audio_path is not None:
            results["audio"] = self.audio_detector.detect_emotion(audio_path)
        
        # Perform fusion if both modalities are available
        if "text" in results and "audio" in results:
            results["fusion"] = self._fuse_results(
                results["text"],
                results["audio"],
                fusion_weights
            )
        
        return results
    
    def _fuse_results(
        self,
        text_result: Dict[str, Any],
        audio_result: Dict[str, Any],
        weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """Fuse text and audio emotion detection results.
        
        Args:
            text_result: Text emotion detection result
            audio_result: Audio emotion detection result
            weights: Fusion weights
            
        Returns:
            Fused emotion detection result
        """
        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
        
        # Get text emotion
        text_emotion = text_result["label"]
        text_score = text_result["score"]
        
        # Get audio emotions
        audio_emotions = {e["label"]: e["score"] for e in audio_result["all_emotions"]}
        
        # Combine scores for matching emotions
        combined_emotions = {}
        
        # Add text emotion
        combined_emotions[text_emotion] = normalized_weights["text"] * text_score
        
        # Add audio emotions
        for emotion, score in audio_emotions.items():
            if emotion in combined_emotions:
                combined_emotions[emotion] += normalized_weights["audio"] * score
            else:
                combined_emotions[emotion] = normalized_weights["audio"] * score
        
        # Sort emotions by score
        sorted_emotions = sorted(
            [{"label": k, "score": v} for k, v in combined_emotions.items()],
            key=lambda x: x["score"],
            reverse=True
        )
        
        return {
            "primary_emotion": sorted_emotions[0],
            "all_emotions": sorted_emotions,
            "fusion_weights": normalized_weights
        }
