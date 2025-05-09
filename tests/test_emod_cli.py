"""
Tests for the EMOD CLI module.

These tests verify that the CLI correctly:
- Parses command-line arguments
- Handles different commands (experiment, results, report)
- Calls the appropriate functions with the correct parameters
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock, call
from io import StringIO

from tests.test_utils import TestBase, patch_modal
import emod_cli

class TestEmodCli(TestBase):
    """Tests for emod_cli.py functionality."""
    
    def setUp(self):
        """Set up test environment."""
        super().setUp()
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = StringIO()  # Capture stdout
        sys.stderr = StringIO()  # Capture stderr
    
    def tearDown(self):
        """Restore stdout and stderr."""
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        super().tearDown()
    
    @patch('emod_cli.run_experiment_grid')
    def test_handle_experiments(self, mock_run_grid):
        """Test handling experiment grid search."""
        # Set up mock
        mock_run_grid.return_value = True
        
        # Create arguments
        args = MagicMock()
        args.text_models = "roberta-base,bert-base"
        args.multimodal = True
        args.audio_features = "mfcc,spectrogram"
        args.fusion_types = "early,late"
        args.ml_classifiers = "gradient_boosting,random_forest"
        args.epochs = 10
        args.batch_size = 16
        args.parallel = False
        args.dry_run = False
        args.yes = True  # Skip confirmation
        
        # Call function
        emod_cli.handle_experiments(args)
        
        # Verify run_experiment_grid was called with correct parameters
        mock_run_grid.assert_called_once()
        call_args = mock_run_grid.call_args[1]
        self.assertEqual(call_args["text_models"], ["roberta-base", "bert-base"])
        self.assertEqual(call_args["audio_features"], ["mfcc", "spectrogram"])
        self.assertEqual(call_args["fusion_types"], ["early", "late"])
        self.assertEqual(call_args["ml_classifiers"], ["gradient_boosting", "random_forest"])
        self.assertEqual(call_args["epochs"], 10)
        self.assertEqual(call_args["batch_size"], 16)
        self.assertEqual(call_args["parallel"], False)
        self.assertEqual(call_args["dry_run"], False)
    
    @patch('emod_cli.download_results')
    @patch('emod_cli.process_results')
    @patch('emod_cli.generate_report')
    def test_handle_results(self, mock_generate_report, mock_process_results, mock_download_results):
        """Test handling results processing."""
        # Set up mocks
        mock_download_results.return_value = True
        mock_process_results.return_value = True
        mock_generate_report.return_value = "/path/to/report.md"
        
        # Create arguments
        args = MagicMock()
        args.skip_download = False
        args.download_only = False
        args.list_only = False
        args.target_dir = self.results_dir
        args.skip_report = False
        args.template = None
        args.format = "markdown"
        
        # Patch subprocess.run to avoid actually opening the report
        with patch('subprocess.run'):
            # Call function
            emod_cli.handle_results(args)
        
        # Verify functions were called with correct parameters
        mock_download_results.assert_called_once_with(
            target_dir=self.results_dir,
            list_only=False
        )
        
        mock_process_results.assert_called_once_with(
            results_dir=self.results_dir,
            output_dir=emod_cli.REPORTS_DIR
        )
        
        mock_generate_report.assert_called_once_with(
            results_dir=self.results_dir,
            output_dir=emod_cli.REPORTS_DIR,
            template=None,
            format="markdown"
        )
        
        # Test download-only mode
        mock_download_results.reset_mock()
        mock_process_results.reset_mock()
        mock_generate_report.reset_mock()
        
        args.download_only = True
        emod_cli.handle_results(args)
        
        mock_download_results.assert_called_once()
        mock_process_results.assert_not_called()
        mock_generate_report.assert_not_called()
        
        # Test skip-download mode
        mock_download_results.reset_mock()
        mock_process_results.reset_mock()
        mock_generate_report.reset_mock()
        
        args.download_only = False
        args.skip_download = True
        emod_cli.handle_results(args)
        
        mock_download_results.assert_not_called()
        mock_process_results.assert_called_once()
        mock_generate_report.assert_called_once()
    
    @patch('emod_cli.generate_report')
    def test_handle_report(self, mock_generate_report):
        """Test handling report generation."""
        # Set up mock
        mock_generate_report.return_value = "/path/to/report.html"
        
        # Create arguments
        args = MagicMock()
        args.target_dir = self.results_dir
        args.template = None
        args.format = "html"
        
        # Patch subprocess.run to avoid actually opening the report
        with patch('subprocess.run'):
            # Call function
            emod_cli.handle_report(args)
        
        # Verify generate_report was called with correct parameters
        mock_generate_report.assert_called_once_with(
            results_dir=self.results_dir,
            output_dir=emod_cli.REPORTS_DIR,
            template=None,
            format="html"
        )
    
    @patch('emod_cli.handle_experiments')
    @patch('emod_cli.handle_results')
    @patch('emod_cli.handle_report')
    @patch('emod_cli.argparse.ArgumentParser.parse_args')
    def test_main_function(self, mock_parse_args, mock_handle_report, mock_handle_results, mock_handle_experiments):
        """Test the main CLI function."""
        # Test experiment command
        args = MagicMock()
        args.command = "experiment"
        mock_parse_args.return_value = args
        
        emod_cli.main()
        mock_handle_experiments.assert_called_once_with(args)
        mock_handle_results.assert_not_called()
        mock_handle_report.assert_not_called()
        
        # Test results command
        mock_handle_experiments.reset_mock()
        args.command = "results"
        emod_cli.main()
        mock_handle_experiments.assert_not_called()
        mock_handle_results.assert_called_once_with(args)
        mock_handle_report.assert_not_called()
        
        # Test report command
        mock_handle_results.reset_mock()
        args.command = "report"
        emod_cli.main()
        mock_handle_experiments.assert_not_called()
        mock_handle_results.assert_not_called()
        mock_handle_report.assert_called_once_with(args)
        
        # Test no command (help)
        mock_handle_report.reset_mock()
        args.command = None
        with patch('emod_cli.argparse.ArgumentParser.print_help') as mock_print_help:
            emod_cli.main()
            mock_print_help.assert_called_once()
        mock_handle_experiments.assert_not_called()
        mock_handle_results.assert_not_called()
        mock_handle_report.assert_not_called()


if __name__ == '__main__':
    unittest.main() 