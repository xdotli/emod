#!/usr/bin/env python3
"""
Script to run emotion recognition with reduced categories.
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run emotion recognition with reduced categories')

    parser.add_argument('--data_path', type=str, default='IEMOCAP_Reduced.csv',
                        help='Path to the reduced IEMOCAP data')
    parser.add_argument('--output_dir', type=str, default='results/reduced_categories',
                        help='Directory to save results')
    parser.add_argument('--vad_model', type=str, default='facebook/bart-large-mnli',
                        choices=['roberta-base', 'facebook/bart-large-mnli'],
                        help='Model for VAD prediction')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for processing')
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='Number of trees in Random Forest')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data for testing')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Proportion of data for validation')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
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

    # Import necessary modules
    from data.data_loader import scale_vad_values, get_emotion_mapping
    from data.data_utils import create_dataloaders
    from models.vad_predictor import BARTZeroShotVADPredictor, ZeroShotVADPredictor
    from models.emotion_classifier import VADEmotionClassifier, evaluate_emotion_classifier
    from models.pipeline import EmotionRecognitionPipeline
    from utils.metrics import log_metrics
    from utils.visualization import plot_confusion_matrix, plot_vad_by_emotion, plot_tsne_vad
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # Load data
    logger.info(f"Loading data from {args.data_path}")
    df = pd.read_csv(args.data_path)

    # Split data
    logger.info("Splitting data into train, validation, and test sets")
    train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=args.random_state)
    train_df, val_df = train_test_split(train_df, test_size=args.val_size/(1-args.test_size), random_state=args.random_state)

    logger.info(f"Data split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Scale VAD values
    logger.info("Scaling VAD values")
    scaler_path = os.path.join(output_dir, 'vad_scaler.pkl')

    # Extract VAD values
    train_vad = train_df[['valence', 'arousal', 'dominance']].values
    val_vad = val_df[['valence', 'arousal', 'dominance']].values
    test_vad = test_df[['valence', 'arousal', 'dominance']].values

    # Fit scaler on training data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    train_vad_scaled = scaler.fit_transform(train_vad)

    # Transform validation and test data
    val_vad_scaled = scaler.transform(val_vad)
    test_vad_scaled = scaler.transform(test_vad)

    # Create new DataFrames with scaled values
    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    test_scaled = test_df.copy()

    train_scaled[['valence', 'arousal', 'dominance']] = train_vad_scaled
    val_scaled[['valence', 'arousal', 'dominance']] = val_vad_scaled
    test_scaled[['valence', 'arousal', 'dominance']] = test_vad_scaled

    # Save the scaler
    import pickle
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    # Get emotion mappings
    unique_emotions = sorted(df['emotion'].unique())
    emotion_to_idx = {emotion: idx for idx, emotion in enumerate(unique_emotions)}
    idx_to_emotion = {idx: emotion for emotion, idx in emotion_to_idx.items()}

    # Save emotion mappings
    with open(os.path.join(output_dir, 'emotion_mappings.json'), 'w') as f:
        json.dump({
            'emotion_to_idx': emotion_to_idx,
            'idx_to_emotion': idx_to_emotion
        }, f, indent=2)

    # Create dataloaders
    logger.info("Creating dataloaders")
    data_splits = {
        'train': train_scaled,
        'val': val_scaled,
        'test': test_scaled
    }

    dataloaders = create_dataloaders(
        data_splits,
        emotion_to_idx,
        batch_size=args.batch_size
    )

    # Initialize VAD predictor
    logger.info(f"Initializing VAD predictor with {args.vad_model}")
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if 'bart' in args.vad_model.lower():
        vad_predictor = BARTZeroShotVADPredictor(args.vad_model, device)
    else:
        vad_predictor = ZeroShotVADPredictor(args.vad_model, device)

    # Evaluate VAD predictor on validation set
    if not args.skip_vad_eval:
        logger.info("Evaluating VAD predictor on validation set")
        from models.vad_predictor import evaluate_vad_predictor
        vad_metrics = evaluate_vad_predictor(vad_predictor, dataloaders['val'], scaler)

        # Log VAD metrics
        log_metrics({'vad_metrics': vad_metrics}, output_dir, prefix='vad_')

        # Plot VAD distributions
        from utils.visualization import plot_vad_distribution
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
    pipeline = EmotionRecognitionPipeline(vad_predictor, emotion_classifier, scaler)

    # Evaluate pipeline on test set
    logger.info("Evaluating pipeline on test set")
    pipeline_metrics = pipeline.evaluate(dataloaders['test'])

    # Log pipeline metrics
    log_metrics(pipeline_metrics, output_dir, prefix='pipeline_')

    # Save pipeline
    logger.info("Saving pipeline")
    pipeline.save(output_dir)

    logger.info(f"All results saved to {output_dir}")

    # Print summary
    print("\nSummary of Results:")
    print("-------------------")
    print(f"Stage 2 (VAD to Emotion) Accuracy: {emotion_metrics['accuracy']:.4f}")
    print(f"End-to-End (Text to Emotion) Accuracy: {pipeline_metrics['emotion_metrics']['accuracy']:.4f}")
    print("\nPerformance by Emotion:")
    for emotion, metrics in emotion_metrics['classification_report'].items():
        if isinstance(metrics, dict) and 'f1-score' in metrics:
            print(f"  {emotion}: F1 = {metrics['f1-score']:.4f}, Support = {metrics['support']}")

if __name__ == '__main__':
    main()
