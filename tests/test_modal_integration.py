"""
Tests for the experiment runner component of Modal integration.

These tests verify that the experiment grid functionality works correctly.
"""

import os
import unittest
from unittest.mock import patch, MagicMock

from tests.test_utils import TestBase
from src.modal import experiment_runner
from src.modal import modal_setup


class TestModalIntegration(TestBase):
    """Tests for Modal integration in EMOD."""
    
    def test_gpu_config(self):
        """Test that GPU configuration works correctly."""
        # Test different GPU types
        t4_config = modal_setup.ModalSetup.get_gpu_config("T4", count=1)
        self.assertEqual(t4_config, "T4")
        
        a10g_config = modal_setup.ModalSetup.get_gpu_config("A10G", count=1)
        self.assertEqual(a10g_config, "A10G")
        
        # Test multiple GPUs
        multi_gpu = modal_setup.ModalSetup.get_gpu_config("T4", count=2)
        self.assertIsInstance(multi_gpu, dict)
        self.assertEqual(multi_gpu["gpu_type"], "T4")
        self.assertEqual(multi_gpu["gpu_count"], 2)
    
    def test_experiment_directory_generation(self):
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
    
    @patch('experiment_runner.run_command')
    def test_experiment_grid(self, mock_run_command):
        """Test running a grid of experiments."""
        # Mock successful command execution
        mock_run_command.return_value = True
        
        # Run a text experiment grid
        success = experiment_runner.run_experiment_grid(
            text_models=["roberta-base", "bert-base"],
            epochs=5,
            dry_run=True
        )
        
        # Verify grid was successful
        self.assertTrue(success)
        
        # Verify correct number of commands
        self.assertEqual(mock_run_command.call_count, 2)
        
        # Reset mock and test multimodal grid
        mock_run_command.reset_mock()
        
        success = experiment_runner.run_experiment_grid(
            text_models=["roberta-base"],
            audio_features=["mfcc", "spectrogram"],
            fusion_types=["early", "late"],
            epochs=5,
            dry_run=True
        )
        
        # Verify grid was successful
        self.assertTrue(success)
        
        # Verify correct number of commands (2 audio features × 2 fusion types = 4 experiments)
        self.assertEqual(mock_run_command.call_count, 4)


if __name__ == '__main__':
    unittest.main() 