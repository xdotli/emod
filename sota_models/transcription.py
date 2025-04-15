"""
State-of-the-art transcription models for speech-to-text conversion.
"""

import os
import logging
import torch
from typing import Dict, Any, Optional, Union
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline

from utils.config import TRANSCRIPTION_MODEL, HF_TOKEN

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TranscriptionModel:
    """Wrapper for state-of-the-art speech transcription models."""
    
    def __init__(self, model_name: str = TRANSCRIPTION_MODEL, use_cuda: bool = True):
        """Initialize the transcription model.
        
        Args:
            model_name: Name or path of the model to use
            use_cuda: Whether to use CUDA for inference
        """
        self.model_name = model_name
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            logger.info(f"Using CUDA for transcription model")
        else:
            logger.info(f"Using CPU for transcription model")
        
        # Load the model
        logger.info(f"Loading transcription model: {model_name}")
        try:
            self.transcriber = pipeline(
                "automatic-speech-recognition",
                model=model_name,
                device=self.device,
                token=HF_TOKEN
            )
            logger.info(f"Transcription model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading transcription model: {e}")
            raise
    
    def transcribe(
        self, 
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        return_timestamps: bool = False
    ) -> Dict[str, Any]:
        """Transcribe audio file to text.
        
        Args:
            audio_path: Path to the audio file
            language: Language code (e.g., "en" for English)
            return_timestamps: Whether to return word-level timestamps
            
        Returns:
            Dictionary with transcription results
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Transcribing audio file: {audio_path}")
        
        try:
            # Set up transcription parameters
            params = {}
            if language:
                params["language"] = language
            if return_timestamps:
                params["return_timestamps"] = "word"
            
            # Perform transcription
            result = self.transcriber(audio_path, **params)
            
            # Format the result
            transcription = {
                "text": result["text"],
                "language": language or "auto-detected"
            }
            
            if return_timestamps and "chunks" in result:
                transcription["timestamps"] = result["chunks"]
            
            logger.info(f"Transcription completed successfully")
            return transcription
            
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            raise
