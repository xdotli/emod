"""
Main script for training and evaluating the emotion recognition pipeline.
"""

import os
import argparse
import logging
import pandas as pd
import numpy as np
import torch
import json
from datetime import datetime

from data.data_loader import (
    load_iemocap_data,
    scale_vad_values,
    get_emotion_mapping,
    save_data_splits
)
from data.data_utils import create_dataloaders
from models.vad_predictor import BARTZeroShotVADPredictor, evaluate_vad_predictor
from models.emotion_classifier import VADEmotionClassifier, evaluate_emotion_classifier
from models.pipeline import EmotionRecognitionPipeline
from utils.metrics import log_metrics
from utils.visualization import (
    plot_confusion_matrix,
    plot_vad_distribution,
    plot_vad_by_emotion,
    plot_tsne_vad
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train and evaluate emotion recognition pipeline')

    parser.add_argument('--data_path', type=str, default='IEMOCAP_Final.csv',
                        help='Path to IEMOCAP_Final.csv')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--vad_model', type=str, default='facebook/bart-large-mnli',
                        choices=['roberta-base', 'facebook/bart-large-mnli'],
                        help='Model for VAD prediction')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for processing')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='Number of trees in Random Forest')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data for testing')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Proportion of data for validation')
    parser.add_argument('--save_data', action='store_true',
                        help='Save data splits to CSV files')
    parser.add_argument('--skip_vad_eval', action='store_true',
                        help='Skip VAD evaluation (faster)')

    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()

    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    # Save arguments
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Load and preprocess data
    logger.info("Loading and preprocessing data")
    data_splits = load_iemocap_data(
        args.data_path,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state
    )

    # Save data splits if requested
    if args.save_data:
        save_data_splits(data_splits, os.path.join(output_dir, 'data'))

    # Scale VAD values
    logger.info("Scaling VAD values")
    scaler_path = os.path.join(output_dir, 'vad_scaler.pkl')
    train_scaled, val_scaled, test_scaled, vad_scaler = scale_vad_values(
        data_splits['train'],
        data_splits['val'],
        data_splits['test'],
        scaler_path
    )

    # Get emotion mappings
    emotion_to_idx, idx_to_emotion = get_emotion_mapping(data_splits['train'])

    # Save emotion mappings
    with open(os.path.join(output_dir, 'emotion_mappings.json'), 'w') as f:
        json.dump({
            'emotion_to_idx': emotion_to_idx,
            'idx_to_emotion': idx_to_emotion
        }, f, indent=2)

    # Create dataloaders
    logger.info("Creating dataloaders")
    dataloaders = create_dataloaders(
        {
            'train': train_scaled,
            'val': val_scaled,
            'test': test_scaled
        },
        emotion_to_idx,
        batch_size=args.batch_size
    )

    # Initialize VAD predictor
    logger.info(f"Initializing VAD predictor with {args.vad_model}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if 'bart' in args.vad_model.lower():
        vad_predictor = BARTZeroShotVADPredictor(args.vad_model, device)
    else:
        from models.vad_predictor import ZeroShotVADPredictor
        vad_predictor = ZeroShotVADPredictor(args.vad_model, device)

    # Evaluate VAD predictor on validation set
    if not args.skip_vad_eval:
        logger.info("Evaluating VAD predictor on validation set")
        vad_metrics = evaluate_vad_predictor(vad_predictor, dataloaders['val'], vad_scaler)

        # Log VAD metrics
        log_metrics({'vad_metrics': vad_metrics}, output_dir, prefix='vad_')

        # Plot VAD distributions
        plot_vad_distribution(
            vad_metrics['true_vad'],
            vad_metrics['pred_vad'],
            os.path.join(output_dir, 'plots')
        )

    # Prepare data for emotion classifier
    logger.info("Preparing data for emotion classifier")
    X_train = train_scaled[['valence', 'arousal', 'dominance']].values
    y_train = train_scaled['emotion'].map(emotion_to_idx).values

    X_val = val_scaled[['valence', 'arousal', 'dominance']].values
    y_val = val_scaled['emotion'].map(emotion_to_idx).values

    X_test = test_scaled[['valence', 'arousal', 'dominance']].values
    y_test = test_scaled['emotion'].map(emotion_to_idx).values

    # Train emotion classifier
    logger.info("Training emotion classifier")
    emotion_classifier = VADEmotionClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state
    )
    emotion_classifier.fit(X_train, y_train, emotion_to_idx, idx_to_emotion)

    # Evaluate emotion classifier on validation set
    logger.info("Evaluating emotion classifier on validation set")
    emotion_metrics = evaluate_emotion_classifier(
        emotion_classifier,
        X_val,
        y_val,
        idx_to_emotion
    )

    # Log emotion metrics
    log_metrics({'emotion_metrics': emotion_metrics}, output_dir, prefix='emotion_')

    # Plot confusion matrix
    y_val_pred = emotion_classifier.predict(X_val)
    plot_confusion_matrix(
        [idx_to_emotion[idx] for idx in y_val],
        [idx_to_emotion[idx] for idx in y_val_pred],
        os.path.join(output_dir, 'plots', 'confusion_matrix.png')
    )

    # Plot VAD by emotion
    plot_vad_by_emotion(
        X_val,
        [idx_to_emotion[idx] for idx in y_val],
        os.path.join(output_dir, 'plots')
    )

    # Plot t-SNE visualization
    plot_tsne_vad(
        X_val,
        [idx_to_emotion[idx] for idx in y_val],
        os.path.join(output_dir, 'plots', 'tsne_vad.png')
    )

    # Create and evaluate pipeline
    logger.info("Creating and evaluating pipeline")
    pipeline = EmotionRecognitionPipeline(vad_predictor, emotion_classifier, vad_scaler)

    # Evaluate pipeline on test set
    logger.info("Evaluating pipeline on test set")
    pipeline_metrics = pipeline.evaluate(dataloaders['test'])

    # Log pipeline metrics
    log_metrics(pipeline_metrics, output_dir, prefix='pipeline_')

    # Save pipeline
    logger.info("Saving pipeline")
    pipeline.save(output_dir)

    logger.info(f"All results saved to {output_dir}")

if __name__ == '__main__':
    main()
