#!/usr/bin/env python3
"""
Centralized Modal Setup Module for EMOD

This module provides utilities for Modal authentication, resource configuration,
and volume management for the EMOD project. It consolidates common Modal setup
code that was previously duplicated across multiple files.
"""

import os
import sys
import subprocess
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple

try:
    import modal
except ImportError:
    # For testing without Modal
    print("Warning: Modal package not found. Using mock implementation.")
    modal = None

# Global volume name for storing results
RESULTS_VOLUME_NAME = "emod-results-vol"

# Default resource settings
DEFAULT_GPU_TYPE = "T4"
DEFAULT_CPU_COUNT = 4
DEFAULT_MEMORY = 16

class ModalSetup:
    """
    Centralized class for Modal setup and resource management.
    
    This class handles:
    1. Authentication and volume setup
    2. GPU and resource configuration
    3. Experiment directory management
    """
    
    @staticmethod
    def authenticate() -> bool:
        """
        Ensure Modal authentication is active.
        
        Returns:
            bool: True if authentication is successful, False otherwise
        """
        try:
            # Try a simple command that requires authentication
            result = subprocess.run(
                ["modal", "volume", "list"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            return True
        except subprocess.CalledProcessError as e:
            if "Please run 'modal token new'" in e.stderr.decode():
                print("Modal authentication required.")
                print("Running 'modal token new'...")
                try:
                    subprocess.run(["modal", "token", "new"], check=True)
                    return True
                except subprocess.CalledProcessError:
                    print("Failed to authenticate with Modal.")
                    return False
            else:
                print(f"Error checking Modal authentication: {e}")
                return False
    
    @staticmethod
    def get_results_volume():
        """
        Get or create the Modal volume for storing experiment results.
        
        Returns:
            modal.Volume: The Modal volume object
        """
        if modal is None:
            return None
        return modal.Volume(RESULTS_VOLUME_NAME, create_if_missing=True)
    
    @staticmethod
    def get_gpu_config(gpu_type: str = DEFAULT_GPU_TYPE, count: int = 1):
        """
        Get GPU configuration for Modal.
        
        Args:
            gpu_type: Type of GPU to use ("T4", "A10G", "A100", etc.)
            count: Number of GPUs to use
            
        Returns:
            str or dict: GPU configuration for Modal
        """
        valid_gpu_types = ["T4", "A10G", "A100"]
        
        if gpu_type not in valid_gpu_types:
            print(f"Warning: GPU type {gpu_type} not recognized. Defaulting to T4.")
            gpu_type = "T4"
        
        # Use string-based GPU configuration (new Modal API)
        if count > 1:
            return {"gpu_count": count, "gpu_type": gpu_type}
        else:
            return gpu_type
    
    @staticmethod
    def create_experiment_image(
        requirements: List[str] = None,
        gpu_type: str = DEFAULT_GPU_TYPE
    ):
        """
        Create a Modal image for running experiments.
        
        Args:
            requirements: Additional Python requirements beyond the base image
            gpu_type: Type of GPU to use
            
        Returns:
            modal.Image: The Modal image object
        """
        if modal is None:
            return None
            
        # Start with the Python 3.9 CUDA base image
        image = modal.Image.debian_slim(python_version="3.9").pip_install(
            "torch==2.0.1",
            "transformers==4.30.2",
            "pandas==2.0.2",
            "numpy==1.25.0",
            "scikit-learn==1.2.2",
            "tqdm==4.65.0"
        )
        
        # Add custom requirements if provided
        if requirements:
            image = image.pip_install(*requirements)
        
        # Add CUDA support for GPU
        if gpu_type in ["T4", "A10G", "A100"]:
            image = image.apt_install("nvidia-cuda-toolkit")
        
        return image
    
    @staticmethod
    def create_experiment_stub(
        name: str,
        image=None,
        gpu=None,
        cpu: int = DEFAULT_CPU_COUNT,
        memory: int = DEFAULT_MEMORY
    ):
        """
        Create a Modal stub for running experiments.
        
        Args:
            name: Name of the experiment
            image: Modal image to use
            gpu: GPU configuration
            cpu: CPU count
            memory: Memory in GB
            
        Returns:
            modal.Stub: The Modal stub object
        """
        if modal is None:
            return None
            
        # Create defaults if not provided
        if image is None:
            image = ModalSetup.create_experiment_image()
        
        if gpu is None:
            gpu = ModalSetup.get_gpu_config()
        
        # Create the stub
        stub = modal.Stub(name)
        
        # Define the function for running experiments
        @stub.function(
            image=image,
            gpu=gpu,
            cpu=cpu,
            memory=memory,
            volumes={RESULTS_VOLUME_NAME: modal.Volume(RESULTS_VOLUME_NAME)}
        )
        def run_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
            """
            Run an experiment with the given configuration.
            
            This is a placeholder that will be implemented by the specific experiment module.
            
            Args:
                config: Experiment configuration
                
            Returns:
                Dict[str, Any]: Experiment results
            """
            return {"status": "not_implemented", "error": "Placeholder function"}
        
        return stub
    
    @staticmethod
    def generate_experiment_dir(
        experiment_type: str,
        text_model: str,
        audio_feature: Optional[str] = None,
        fusion_type: Optional[str] = None
    ) -> str:
        """
        Generate a unique directory name for an experiment.
        
        Args:
            experiment_type: Either 'text' or 'multimodal'
            text_model: Name of the text model
            audio_feature: Name of the audio feature (for multimodal)
            fusion_type: Type of fusion (for multimodal)
            
        Returns:
            str: Directory name for the experiment
        """
        # Clean up model name for file paths
        clean_model_name = text_model.replace('/', '_')
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate directory name
        if experiment_type == "text":
            return f"text_model_{clean_model_name}_{timestamp}"
        else:
            return f"multimodal_{clean_model_name}_{audio_feature}_{fusion_type}_{timestamp}"
    
    @staticmethod
    def save_results_to_volume(
        results: Dict[str, Any],
        volume,
        experiment_dir: str
    ) -> bool:
        """
        Save experiment results to Modal volume.
        
        Args:
            results: Experiment results
            volume: Modal volume
            experiment_dir: Directory name for the experiment
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create local temporary directory
            local_dir = Path(f"/tmp/{experiment_dir}")
            logs_dir = local_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            
            # Save final results
            if "vad_metrics" in results:
                with open(logs_dir / "final_results.json", "w") as f:
                    json.dump(results["vad_metrics"], f, indent=2)
            
            # Save training log
            if "training_log" in results:
                with open(logs_dir / "training_log.json", "w") as f:
                    json.dump(results["training_log"], f, indent=2)
            
            # Save ML classifier results
            if "classifier_metrics" in results:
                with open(local_dir / "ml_classifier_results.json", "w") as f:
                    json.dump(results["classifier_metrics"], f, indent=2)
            
            # Copy to volume (if not None)
            if volume is not None:
                volume.put(str(local_dir), f"/{experiment_dir}")
            
            return True
        except Exception as e:
            print(f"Error saving results to Modal volume: {e}")
            return False


# Singleton instance for convenience
modal_setup = ModalSetup()


def authenticate_modal() -> bool:
    """
    Ensure Modal CLI is installed and authenticated.
    
    Returns:
        bool: True if authentication is successful, False otherwise
    """
    # Check Modal CLI is installed
    try:
        subprocess.run(["modal", "--version"], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE, 
                      check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Modal CLI not found or not properly installed.")
        print("Please install it with: pip install modal")
        print("Then run: modal token new")
        return False
    
    # Authenticate
    return ModalSetup.authenticate()


def get_results_volume():
    """
    Get the Modal volume for results storage.
    
    Returns:
        modal.Volume: The Modal volume object
    """
    return ModalSetup.get_results_volume()


def create_experiment_stub(name: str, gpu_type: str = DEFAULT_GPU_TYPE, requirements: List[str] = None):
    """
    Create a Modal stub for running experiments.
    
    Args:
        name: Name of the experiment
        gpu_type: Type of GPU to use
        requirements: Additional Python requirements
        
    Returns:
        modal.Stub: The Modal stub object
    """
    # Create image
    image = ModalSetup.create_experiment_image(requirements, gpu_type)
    
    # Create GPU config
    gpu = ModalSetup.get_gpu_config(gpu_type)
    
    # Create and return stub
    return ModalSetup.create_experiment_stub(name, image, gpu)


# Simple test for the module
if __name__ == "__main__":
    # Test authentication
    auth_success = authenticate_modal()
    print(f"Authentication: {'Success' if auth_success else 'Failed'}")
    
    # Test directory generation
    text_dir = ModalSetup.generate_experiment_dir("text", "roberta-base")
    print(f"Text experiment directory: {text_dir}")
    
    multimodal_dir = ModalSetup.generate_experiment_dir("multimodal", "roberta-base", "mfcc", "early")
    print(f"Multimodal experiment directory: {multimodal_dir}") 