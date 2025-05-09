"""
Test utilities for EMOD project.

This module provides common utilities for testing EMOD components:
- Sample data generation
- Mock Modal functionality
- Test helpers and fixtures

Note: After the code restructuring, core EMOD modules are now in the src/ directory:
- src/core/ - Core model implementation
- src/processing/ - Data and results processing 
- src/utils/ - Utilities
- src/modal/ - Modal integration
"""

import os
import json
import shutil
import tempfile
import unittest
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from unittest.mock import MagicMock, patch

# Sample data for testing
SAMPLE_DATA = {
    "text": [
        "I'm feeling really happy today!",
        "This makes me so angry.",
        "I'm feeling quite sad and down.",
        "Just a regular day, nothing special."
    ],
    "audio": [
        "path/to/happy_audio.wav",
        "path/to/angry_audio.wav",
        "path/to/sad_audio.wav",
        "path/to/neutral_audio.wav"
    ],
    "valence": [0.8, 0.2, 0.3, 0.5],
    "arousal": [0.7, 0.9, 0.3, 0.4],
    "dominance": [0.8, 0.8, 0.2, 0.5],
    "emotion": ["happy", "angry", "sad", "neutral"]
}

# Sample VAD prediction results
SAMPLE_VAD_RESULTS = {
    "final_metrics": {
        "Valence": {"MSE": 0.1231, "RMSE": 0.3509, "MAE": 0.2879, "R2": 0.7654},
        "Arousal": {"MSE": 0.1421, "RMSE": 0.3769, "MAE": 0.3012, "R2": 0.6987},
        "Dominance": {"MSE": 0.1598, "RMSE": 0.3998, "MAE": 0.3245, "R2": 0.6543},
        "Test Loss": 0.1417
    },
    "best_val_loss": 0.1356,
    "training_time": 342.5
}

# Sample classifier results
SAMPLE_CLASSIFIER_RESULTS = [
    {
        "classifier": "gradient_boosting",
        "accuracy": 0.7912,
        "weighted_f1": 0.7843,
        "macro_f1": 0.7651,
        "class_f1": {"happy": 0.8124, "angry": 0.8231, "sad": 0.7012, "neutral": 0.7421}
    },
    {
        "classifier": "random_forest",
        "accuracy": 0.7651,
        "weighted_f1": 0.7598,
        "macro_f1": 0.7432,
        "class_f1": {"happy": 0.7865, "angry": 0.7921, "sad": 0.6987, "neutral": 0.7213}
    }
]

# Mock training logs
SAMPLE_TRAINING_LOG = {
    "start_time": "2023-05-07 23:15:42",
    "end_time": "2023-05-08 00:10:31",
    "hyperparameters": {
        "epochs": 20,
        "batch_size": 16,
        "learning_rate": 2e-5
    },
    "epoch_logs": [
        {"epoch": 1, "train_loss": 0.6523, "val_loss": 0.5987},
        {"epoch": 2, "train_loss": 0.5432, "val_loss": 0.5123},
        # Additional epochs would be here
        {"epoch": 19, "train_loss": 0.1632, "val_loss": 0.1423},
        {"epoch": 20, "train_loss": 0.1534, "val_loss": 0.1392}
    ]
}


class MockModalVolume:
    """Mock implementation of Modal Volume for testing."""
    
    def __init__(self, name: str = "test-volume"):
        self.name = name
        self.storage = {}
    
    def put(self, local_path: str, remote_path: str) -> None:
        """Mock putting a file to Modal volume."""
        with open(local_path, 'rb') as f:
            self.storage[remote_path] = f.read()
    
    def get(self, remote_path: str, local_path: str) -> None:
        """Mock getting a file from Modal volume."""
        if remote_path in self.storage:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, 'wb') as f:
                f.write(self.storage[remote_path])
        else:
            raise FileNotFoundError(f"File {remote_path} not found in volume")


class MockModalFunction:
    """Mock implementation of Modal Function for testing."""
    
    def __init__(self, name: str = "test-function"):
        self.name = name
        self.calls = []
    
    def call(self, *args, **kwargs):
        """Mock calling a Modal function."""
        self.calls.append((args, kwargs))
        return {"status": "success", "result": "mock-result"}


class TestBase(unittest.TestCase):
    """Base class for EMOD tests with common setup and teardown."""
    
    def setUp(self):
        """Set up test environment with temporary directories."""
        # Create temporary directories for test data
        self.test_dir = tempfile.mkdtemp()
        self.results_dir = os.path.join(self.test_dir, "results")
        self.reports_dir = os.path.join(self.test_dir, "reports")
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Create sample dataset
        self.sample_df = pd.DataFrame(SAMPLE_DATA)
        self.sample_csv_path = os.path.join(self.test_dir, "sample_data.csv")
        self.sample_df.to_csv(self.sample_csv_path, index=False)
        
        # Set up mock Modal objects
        self.mock_volume = MockModalVolume()
        self.mock_function = MockModalFunction()
        
        # Create a sample experiment directory
        self.exp_dir = os.path.join(self.results_dir, "text_model_roberta_base_123456")
        os.makedirs(os.path.join(self.exp_dir, "logs"), exist_ok=True)
        
        # Create sample results files
        with open(os.path.join(self.exp_dir, "logs", "final_results.json"), 'w') as f:
            json.dump(SAMPLE_VAD_RESULTS, f)
        
        with open(os.path.join(self.exp_dir, "logs", "training_log.json"), 'w') as f:
            json.dump(SAMPLE_TRAINING_LOG, f)
            
        with open(os.path.join(self.exp_dir, "ml_classifier_results.json"), 'w') as f:
            json.dump(SAMPLE_CLASSIFIER_RESULTS, f)
    
    def tearDown(self):
        """Clean up temporary test directories."""
        shutil.rmtree(self.test_dir)
    
    def create_experiment_result(self, experiment_type: str, text_model: str, 
                               audio_feature: Optional[str] = None, 
                               fusion_type: Optional[str] = None,
                               timestamp: str = "123456") -> str:
        """
        Create a sample experiment result directory with necessary files.
        
        Args:
            experiment_type: Either "text" or "multimodal"
            text_model: Name of the text model
            audio_feature: Name of the audio feature (for multimodal)
            fusion_type: Type of fusion (for multimodal)
            timestamp: Timestamp string for the experiment
            
        Returns:
            str: Path to the created experiment directory
        """
        # Create directory name
        if experiment_type == "text":
            dir_name = f"text_model_{text_model}_{timestamp}"
        else:
            dir_name = f"multimodal_{text_model}_{audio_feature}_{fusion_type}_{timestamp}"
        
        # Create directory structure
        exp_dir = os.path.join(self.results_dir, dir_name)
        os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
        
        # Create result files
        with open(os.path.join(exp_dir, "logs", "final_results.json"), 'w') as f:
            json.dump(SAMPLE_VAD_RESULTS, f)
        
        with open(os.path.join(exp_dir, "logs", "training_log.json"), 'w') as f:
            json.dump(SAMPLE_TRAINING_LOG, f)
            
        with open(os.path.join(exp_dir, "ml_classifier_results.json"), 'w') as f:
            json.dump(SAMPLE_CLASSIFIER_RESULTS, f)
        
        return exp_dir


# Patch decorators for Modal
def mock_modal_function(func):
    """Decorator to mock Modal function."""
    return func

def mock_modal_volume(name):
    """Mock Modal volume."""
    return MockModalVolume(name)

def mock_modal_image():
    """Mock Modal image."""
    return MagicMock()


# Patchers for Modal modules
def patch_modal():
    """Create patchers for Modal modules."""
    modal_mock = MagicMock()
    modal_mock.function = mock_modal_function
    modal_mock.Volume = mock_modal_volume
    modal_mock.Image = mock_modal_image
    
    return patch.dict('sys.modules', {'modal': modal_mock})


def generate_sample_audio_features(n_samples: int = 4, feature_dim: int = 40) -> np.ndarray:
    """Generate sample audio features for testing."""
    return np.random.randn(n_samples, feature_dim) 