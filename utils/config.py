"""
Configuration utilities for the emotion detection system.
Loads environment variables and provides configuration settings.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# API Keys
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
HF_TOKEN = os.getenv('HF_TOKEN')

# Model Configuration
USE_SOTA_MODELS = os.getenv('USE_SOTA_MODELS', 'true').lower() == 'true'

# OpenRouter Model IDs
CLAUDE_MODEL = os.getenv('CLAUDE_MODEL', 'anthropic/claude-3-7-sonnet-20240620')
GPT4O_MODEL = os.getenv('GPT4O_MODEL', 'openai/gpt-4o-2024-05-13')
DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-ai/deepseek-coder-v2')

# Transcription Model
TRANSCRIPTION_MODEL = os.getenv('TRANSCRIPTION_MODEL', 'openai/whisper-large-v3')

# Emotion Detection Models
TEXT_EMOTION_MODEL = os.getenv('TEXT_EMOTION_MODEL', 'AnkitAI/deberta-v3-small-base-emotions-classifier')
AUDIO_EMOTION_MODEL = os.getenv('AUDIO_EMOTION_MODEL', 'MIT/ast-finetuned-audioset-10-10-0.4593')

# Check if API keys are available
if not OPENROUTER_API_KEY:
    logger.warning("OpenRouter API key not found. Some features may not work.")

if not HF_TOKEN:
    logger.warning("Hugging Face token not found. Some features may not work.")

def get_config():
    """Return the configuration as a dictionary."""
    return {
        'api_keys': {
            'openrouter': OPENROUTER_API_KEY,
            'huggingface': HF_TOKEN,
        },
        'use_sota_models': USE_SOTA_MODELS,
        'models': {
            'llm': {
                'claude': CLAUDE_MODEL,
                'gpt4o': GPT4O_MODEL,
                'deepseek': DEEPSEEK_MODEL,
            },
            'transcription': TRANSCRIPTION_MODEL,
            'emotion': {
                'text': TEXT_EMOTION_MODEL,
                'audio': AUDIO_EMOTION_MODEL,
            }
        }
    }
