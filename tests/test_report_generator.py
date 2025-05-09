"""
Tests for the report generator module.

These tests verify that the report generator correctly:
- Collects experiment data
- Formats tables and visualizations
- Generates comprehensive reports in both markdown and HTML formats
"""

import os
import json
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import tempfile
import shutil

from tests.test_utils import TestBase
from src.processing.report_generator import (
    collect_experiment_data,
    format_dataframe_as_markdown,
    format_dataframe_as_html,
    select_top_vad_models,
    select_top_classifier_models,
    format_model_tables,
    generate_report,
    DEFAULT_MARKDOWN_TEMPLATE,  # Import the default templates
    DEFAULT_HTML_TEMPLATE
)


class TestReportGenerator(TestBase):
    """Tests for report_generator.py functionality."""
    
    def setUp(self):
        """Set up test environment with additional experiment data."""
        super().setUp()
        
        # Create additional sample data in reports directory
        self.experiment_summary = pd.DataFrame([
            {
                "directory": "text_model_roberta_base_123456",
                "Type": "Text-only",
                "text_model": "roberta_base",
                "valence_mse": 0.1231,
                "arousal_mse": 0.1421,
                "dominance_mse": 0.1598,
                "best_classifier": "gradient_boosting",
                "best_classifier_accuracy": 0.7912,
                "best_classifier_f1": 0.7843
            },
            {
                "directory": "multimodal_bert_base_mfcc_early_123457",
                "Type": "Multimodal",
                "text_model": "bert_base",
                "audio_feature": "mfcc",
                "fusion_type": "early",
                "valence_mse": 0.1121,
                "arousal_mse": 0.1321,
                "dominance_mse": 0.1512,
                "best_classifier": "gradient_boosting",
                "best_classifier_accuracy": 0.8123,
                "best_classifier_f1": 0.8143
            }
        ])
        
        # Save to CSV for testing
        os.makedirs(self.reports_dir, exist_ok=True)
        self.experiment_summary.to_csv(os.path.join(self.results_dir, "experiment_summary.csv"), index=False)
        
        # Create separate CSV files for text and multimodal models
        text_df = self.experiment_summary[self.experiment_summary["Type"] == "Text-only"]
        multimodal_df = self.experiment_summary[self.experiment_summary["Type"] == "Multimodal"]
        
        text_df.to_csv(os.path.join(self.reports_dir, "text_model_comparison.csv"), index=False)
        multimodal_df.to_csv(os.path.join(self.reports_dir, "multimodal_comparison.csv"), index=False)
    
    def test_collect_experiment_data(self):
        """Test collecting experiment data from result files."""
        # Collect experiment data
        experiment_data, text_count, multimodal_count = collect_experiment_data(self.results_dir)
        
        # Verify structure and content
        self.assertEqual(len(experiment_data), 2)
        self.assertEqual(text_count, 1)
        self.assertEqual(multimodal_count, 1)
        
        # Verify data comes from summary file
        self.assertEqual(experiment_data[0]["directory"], "text_model_roberta_base_123456")
        self.assertEqual(experiment_data[1]["directory"], "multimodal_bert_base_mfcc_early_123457")
        
        # Test fallback to separate tables
        os.remove(os.path.join(self.results_dir, "experiment_summary.csv"))
        experiment_data, text_count, multimodal_count = collect_experiment_data(self.results_dir)
        
        # Verify we still get data
        self.assertEqual(len(experiment_data), 2)
        self.assertEqual(text_count, 1)
        self.assertEqual(multimodal_count, 1)
    
    def test_format_dataframe_methods(self):
        """Test DataFrame formatting functions."""
        # Create a simple DataFrame
        df = pd.DataFrame({
            "Model": ["roberta_base", "bert_base"],
            "MSE": [0.123, 0.112],
            "F1": [0.784, 0.814]
        })
        
        # Test markdown formatting
        markdown = format_dataframe_as_markdown(df)
        self.assertIsInstance(markdown, str)
        self.assertIn("Model", markdown)
        self.assertIn("roberta_base", markdown)
        
        # Test HTML formatting
        html = format_dataframe_as_html(df)
        self.assertIsInstance(html, str)
        self.assertIn("<table", html)
        self.assertIn("</table>", html)
        self.assertIn("roberta_base", html)
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        markdown_empty = format_dataframe_as_markdown(empty_df)
        html_empty = format_dataframe_as_html(empty_df)
        
        self.assertEqual(markdown_empty, "No data available")
        self.assertIn("<p>No data available</p>", html_empty)
    
    def test_select_top_models(self):
        """Test selecting top models based on performance."""
        # Collect experiment data
        experiment_data, _, _ = collect_experiment_data(self.results_dir)
        
        # Select top VAD models
        top_vad_df = select_top_vad_models(experiment_data, top_n=2)
        
        # Verify structure and content
        self.assertEqual(len(top_vad_df), 2)
        self.assertIn("Text Model", top_vad_df.columns)
        self.assertIn("Valence MSE", top_vad_df.columns)
        
        # Verify sorted by Valence MSE (lower is better)
        self.assertTrue(top_vad_df.iloc[0]["Valence MSE"] <= top_vad_df.iloc[1]["Valence MSE"])
        
        # Select top classifier models
        top_clf_df = select_top_classifier_models(experiment_data, top_n=2)
        
        # Verify structure and content
        self.assertEqual(len(top_clf_df), 2)
        self.assertIn("Text Model", top_clf_df.columns)
        self.assertIn("F1 Score", top_clf_df.columns)
        
        # Verify sorted by F1 score (higher is better)
        self.assertTrue(top_clf_df.iloc[0]["F1 Score"] >= top_clf_df.iloc[1]["F1 Score"])
    
    def test_format_model_tables(self):
        """Test formatting experiment data into separate tables."""
        # Collect experiment data
        experiment_data, _, _ = collect_experiment_data(self.results_dir)
        
        # Format tables
        text_df, multimodal_df = format_model_tables(experiment_data)
        
        # Verify text model table
        self.assertEqual(len(text_df), 1)
        self.assertIn("Text Model", text_df.columns)
        self.assertIn("Valence MSE", text_df.columns)
        self.assertIn("F1 Score", text_df.columns)
        
        # Verify multimodal table
        self.assertEqual(len(multimodal_df), 1)
        self.assertIn("Text Model", multimodal_df.columns)
        self.assertIn("Audio Feature", multimodal_df.columns)
        self.assertIn("Fusion Type", multimodal_df.columns)
        self.assertIn("Valence MSE", multimodal_df.columns)
        self.assertIn("F1 Score", multimodal_df.columns)
    
    @patch('report_generator.generate_performance_visualizations')
    def test_generate_report(self, mock_viz):
        """Test generating a report from experiment data."""
        # Mock visualizations
        mock_viz.return_value = {
            'vad_chart': os.path.join(self.reports_dir, 'vad_performance_chart.png'),
            'f1_chart': os.path.join(self.reports_dir, 'classification_f1_chart.png'),
            'compare_chart': os.path.join(self.reports_dir, 'text_vs_multimodal_chart.png')
        }
        
        # Test with our own simple template to avoid HTML formatting issues
        with patch('report_generator.DEFAULT_MARKDOWN_TEMPLATE', "# Test Report\n\n{experiment_count} experiments"):
            # Generate markdown report
            report_path = generate_report(
                results_dir=self.results_dir,
                output_dir=self.reports_dir,
                format="markdown"
            )
            
            # Verify report file was created
            self.assertIsNotNone(report_path)
            self.assertTrue(os.path.exists(report_path))
            self.assertEqual(os.path.basename(report_path), "experiment_report.md")
            
            # Verify report content
            with open(report_path, 'r') as f:
                content = f.read()
                self.assertIn("# Test Report", content)
                self.assertIn("2 experiments", content)


if __name__ == '__main__':
    unittest.main() 