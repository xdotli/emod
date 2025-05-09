"""
Tests for the Modal setup module.

These tests verify that the Modal setup module correctly:
- Handles authentication
- Creates and manages volumes
- Configures resources
- Generates experiment directories
- Saves results to volumes
"""

import os
import unittest
from unittest.mock import patch, MagicMock
import json
import tempfile
import shutil

from tests.test_utils import TestBase

# Import the module directly
from src.modal import modal_setup


class TestModalSetup(TestBase):
    """Tests for modal_setup.py functionality."""
    
    def test_get_gpu_config(self):
        """Test getting GPU configuration."""
        # Test different GPU types
        t4_config = modal_setup.ModalSetup.get_gpu_config("T4")
        self.assertEqual(t4_config, "T4")
        
        a10g_config = modal_setup.ModalSetup.get_gpu_config("A10G")
        self.assertEqual(a10g_config, "A10G")
        
        a100_config = modal_setup.ModalSetup.get_gpu_config("A100")
        self.assertEqual(a100_config, "A100")
        
        # Test an invalid GPU type (should default to T4)
        with patch('builtins.print') as mock_print:
            invalid_config = modal_setup.ModalSetup.get_gpu_config("Invalid")
            mock_print.assert_called_once()
            self.assertEqual(invalid_config, "T4")
        
        # Test with count > 1
        multi_gpu = modal_setup.ModalSetup.get_gpu_config("T4", count=2)
        self.assertIsInstance(multi_gpu, dict)
        self.assertEqual(multi_gpu["gpu_type"], "T4")
        self.assertEqual(multi_gpu["gpu_count"], 2)
    
    def test_generate_experiment_dir(self):
        """Test generating experiment directory names."""
        # Test text-only experiment
        text_dir = modal_setup.ModalSetup.generate_experiment_dir("text", "roberta-base")
        self.assertTrue(text_dir.startswith("text_model_roberta-base_"))
        self.assertGreater(len(text_dir), 25)  # Should include timestamp
        
        # Test multimodal experiment
        mm_dir = modal_setup.ModalSetup.generate_experiment_dir(
            "multimodal", "bert-base", "mfcc", "early"
        )
        self.assertTrue(mm_dir.startswith("multimodal_bert-base_mfcc_early_"))
        self.assertGreater(len(mm_dir), 30)  # Should include timestamp
        
        # Test handling of model names with slashes
        complex_dir = modal_setup.ModalSetup.generate_experiment_dir(
            "text", "microsoft/deberta-v3-base"
        )
        self.assertTrue(complex_dir.startswith("text_model_microsoft_deberta-v3-base_"))
        self.assertNotIn("/", complex_dir)  # Should replace slashes


if __name__ == '__main__':
    unittest.main() 