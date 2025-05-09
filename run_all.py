#!/usr/bin/env python3
"""
Wrapper script to run all EMOD experiments using Modal
"""

import sys
import os
import modal

# Add experiments directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "experiments"))

# Import the app from run_comprehensive_experiments
from experiments.run_comprehensive_experiments import app, train_model, TEXT_MODELS, AUDIO_FEATURES, FUSION_TYPES

@app.local_entrypoint()
def run_all_experiments():
    """Run all experiments (text and multimodal) on all datasets"""
    print("Running full EMOD experiment suite (all text models and multimodal combinations)")
    
    # Define datasets to use
    datasets = ["IEMOCAP_Final", "IEMOCAP_Filtered"]
    
    # Run full experiment grid
    completed_experiments = []
    
    # Text-only experiments
    for dataset in datasets:
        for text_model in TEXT_MODELS:
            print(f"Launching text experiment with {text_model} on {dataset}...")
            result = train_model.remote(
                text_model_name=text_model,
                dataset_name=dataset,
                num_epochs=40
            )
            completed_experiments.append({
                "type": "text",
                "model": text_model,
                "dataset": dataset,
                "result": "launched"
            })
    
    # Multimodal experiments
    for dataset in datasets:
        for text_model in TEXT_MODELS:
            for audio_feature in AUDIO_FEATURES:
                for fusion_type in FUSION_TYPES:
                    print(f"Launching multimodal experiment with {text_model}, {audio_feature}, {fusion_type} on {dataset}...")
                    result = train_model.remote(
                        text_model_name=text_model,
                        dataset_name=dataset,
                        audio_feature_type=audio_feature,
                        fusion_type=fusion_type,
                        num_epochs=40
                    )
                    completed_experiments.append({
                        "type": "multimodal",
                        "model": text_model,
                        "audio_feature": audio_feature,
                        "fusion_type": fusion_type,
                        "dataset": dataset,
                        "result": "launched"
                    })
    
    print(f"\nLaunched {len(completed_experiments)} experiments on Modal")
    print("All experiments are running on a persistent Modal instance.")
    print("Results will be saved to the Modal volume and can be downloaded later.") 