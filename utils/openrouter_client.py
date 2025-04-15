"""
OpenRouter client for accessing various LLMs.
"""

import os
import json
import logging
import requests
from typing import Dict, List, Optional, Union, Any

from utils.config import OPENROUTER_API_KEY

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OpenRouterClient:
    """Client for interacting with the OpenRouter API to access various LLMs."""
    
    BASE_URL = "https://openrouter.ai/api/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the OpenRouter client.
        
        Args:
            api_key: OpenRouter API key. If None, will use the OPENROUTER_API_KEY from environment.
        """
        self.api_key = api_key or OPENROUTER_API_KEY
        if not self.api_key:
            raise ValueError("OpenRouter API key is required")
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://emod.local",  # Required for OpenRouter
            "X-Title": "EMOD Emotion Detection"    # Optional but helpful for tracking
        }
    
    def list_models(self) -> Dict[str, Any]:
        """List all available models on OpenRouter."""
        url = f"{self.BASE_URL}/models"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def generate_text(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text using the specified model.
        
        Args:
            model: Model ID (e.g., "anthropic/claude-3-7-sonnet-20240620")
            messages: List of message objects with role and content
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            API response as a dictionary
        """
        url = f"{self.BASE_URL}/chat/completions"
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            **kwargs
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, stream=stream)
            response.raise_for_status()
            
            if stream:
                return response  # Return the response object for streaming
            else:
                return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling OpenRouter API: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
            raise
    
    def analyze_emotion(
        self,
        model: str,
        text: str,
        audio_transcription: Optional[str] = None,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze emotion in text using an LLM.
        
        Args:
            model: Model ID to use
            text: Text to analyze
            audio_transcription: Optional transcription from audio
            context: Optional additional context
            
        Returns:
            Dictionary with emotion analysis
        """
        system_prompt = """
        You are an expert emotion detection system. Analyze the provided text and identify the emotions expressed.
        
        For each emotion detected, provide:
        1. The emotion name
        2. Confidence score (0-100)
        3. Brief explanation of why this emotion was detected
        
        Focus on these primary emotions: anger, disgust, fear, joy, sadness, surprise, neutral.
        You may also detect secondary emotions if clearly present.
        
        Return your analysis in JSON format with the following structure:
        {
            "primary_emotion": {
                "name": "emotion_name",
                "confidence": confidence_score,
                "explanation": "brief explanation"
            },
            "secondary_emotions": [
                {
                    "name": "emotion_name",
                    "confidence": confidence_score,
                    "explanation": "brief explanation"
                }
            ],
            "overall_sentiment": "positive/negative/neutral",
            "intensity": intensity_score
        }
        """
        
        content = f"Text to analyze: {text}\n"
        if audio_transcription:
            content += f"\nAudio transcription: {audio_transcription}\n"
        if context:
            content += f"\nAdditional context: {context}\n"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]
        
        response = self.generate_text(model, messages, temperature=0.3)
        
        try:
            # Extract JSON from the response
            content = response['choices'][0]['message']['content']
            # Sometimes the LLM might wrap the JSON in markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()
            
            result = json.loads(content)
            return result
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing LLM response: {e}")
            logger.error(f"Raw response: {response}")
            # Return a simplified version if parsing fails
            return {
                "error": "Failed to parse LLM response",
                "raw_response": response['choices'][0]['message']['content'] if 'choices' in response else str(response)
            }
