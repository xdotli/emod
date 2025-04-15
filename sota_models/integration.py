"""
Integration of SOTA models with the existing emotion detection pipeline.
"""

import os
import logging
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

from utils.config import get_config, CLAUDE_MODEL, GPT4O_MODEL
from utils.openrouter_client import OpenRouterClient
from sota_models.transcription import TranscriptionModel
from sota_models.emotion_detection import MultimodalEmotionDetector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SotaEmotionAnalyzer:
    """Integrates SOTA models for comprehensive emotion analysis."""
    
    def __init__(self, use_cuda: bool = True):
        """Initialize the SOTA emotion analyzer.
        
        Args:
            use_cuda: Whether to use CUDA for inference
        """
        self.config = get_config()
        self.use_cuda = use_cuda
        
        # Initialize components
        logger.info("Initializing SOTA emotion analyzer components")
        
        # OpenRouter client for LLM access
        self.openrouter = OpenRouterClient()
        
        # Transcription model
        self.transcriber = TranscriptionModel(use_cuda=use_cuda)
        
        # Emotion detection models
        self.emotion_detector = MultimodalEmotionDetector(use_cuda=use_cuda)
        
        logger.info("SOTA emotion analyzer initialized successfully")
    
    def analyze_audio(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        llm_model: str = CLAUDE_MODEL,
        return_timestamps: bool = False
    ) -> Dict[str, Any]:
        """Analyze emotions in an audio file.
        
        Args:
            audio_path: Path to the audio file
            language: Language code (e.g., "en" for English)
            llm_model: LLM model to use for analysis
            return_timestamps: Whether to return word-level timestamps
            
        Returns:
            Dictionary with comprehensive emotion analysis
        """
        logger.info(f"Analyzing emotions in audio file: {audio_path}")
        
        # Step 1: Transcribe audio
        transcription = self.transcriber.transcribe(
            audio_path,
            language=language,
            return_timestamps=return_timestamps
        )
        
        # Step 2: Detect emotions using multimodal models
        emotion_results = self.emotion_detector.detect_emotion(
            text=transcription["text"],
            audio_path=audio_path
        )
        
        # Step 3: Analyze with LLM for deeper insights
        llm_analysis = self.openrouter.analyze_emotion(
            model=llm_model,
            text=transcription["text"],
            audio_transcription=None,  # Already included in text
            context=f"Audio emotion detection result: {json.dumps(emotion_results)}"
        )
        
        # Combine all results
        analysis = {
            "transcription": transcription,
            "model_detection": emotion_results,
            "llm_analysis": llm_analysis,
            "metadata": {
                "audio_path": str(audio_path),
                "language": language or "auto-detected",
                "llm_model": llm_model
            }
        }
        
        logger.info(f"Audio emotion analysis completed")
        return analysis
    
    def analyze_text(
        self,
        text: str,
        llm_model: str = CLAUDE_MODEL
    ) -> Dict[str, Any]:
        """Analyze emotions in text.
        
        Args:
            text: Text to analyze
            llm_model: LLM model to use for analysis
            
        Returns:
            Dictionary with comprehensive emotion analysis
        """
        logger.info(f"Analyzing emotions in text: {text[:50]}...")
        
        # Step 1: Detect emotions using model
        emotion_results = self.emotion_detector.detect_emotion(text=text)
        
        # Step 2: Analyze with LLM for deeper insights
        llm_analysis = self.openrouter.analyze_emotion(
            model=llm_model,
            text=text,
            context=f"Text emotion detection result: {json.dumps(emotion_results)}"
        )
        
        # Combine all results
        analysis = {
            "text": text,
            "model_detection": emotion_results,
            "llm_analysis": llm_analysis,
            "metadata": {
                "llm_model": llm_model
            }
        }
        
        logger.info(f"Text emotion analysis completed")
        return analysis
    
    def compare_llm_analyses(
        self,
        text: str,
        models: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compare emotion analyses from different LLMs.
        
        Args:
            text: Text to analyze
            models: List of LLM models to use (defaults to Claude and GPT-4o)
            
        Returns:
            Dictionary with comparative emotion analyses
        """
        if models is None:
            models = [CLAUDE_MODEL, GPT4O_MODEL]
        
        logger.info(f"Comparing LLM emotion analyses for text: {text[:50]}...")
        
        # Detect emotions using model
        emotion_results = self.emotion_detector.detect_emotion(text=text)
        
        # Analyze with each LLM
        llm_analyses = {}
        for model in models:
            logger.info(f"Analyzing with model: {model}")
            llm_analyses[model] = self.openrouter.analyze_emotion(
                model=model,
                text=text,
                context=f"Text emotion detection result: {json.dumps(emotion_results)}"
            )
        
        # Combine all results
        analysis = {
            "text": text,
            "model_detection": emotion_results,
            "llm_analyses": llm_analyses,
            "metadata": {
                "models": models
            }
        }
        
        logger.info(f"Comparative LLM emotion analysis completed")
        return analysis
