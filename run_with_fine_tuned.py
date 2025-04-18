#!/usr/bin/env python3
"""
Script to train and evaluate an emotion recognition model using a fine-tuned VAD predictor.
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
    parser = argparse.ArgumentParser(description='Train and evaluate an emotion recognition model using a fine-tuned VAD predictor')
    
    parser.add_argument('--data_path', type=str, default='IEMOCAP_Final.csv',
                        help='Path to IEMOCAP_Final.csv')
    parser.add_argument('--output_dir', type=str, default='results/fine_tuned',
                        help='Directory to save results')
    parser.add_argument('--fine_tuned_model_dir', type=str, required=True,
                        help='Directory containing the fine-tuned VAD predictor')
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
    from src.data.data_loader import load_iemocap_data, scale_vad_values, get_emotion_mapping
    from src.data.data_utils import create_dataloaders
    from src.models.vad_fine_tuner import VADFineTuner
    from src.models.emotion_classifier import VADEmotionClassifier, evaluate_emotion_classifier
    from src.models.pipeline import EmotionRecognitionPipeline
    from src.utils.metrics import log_metrics
    from src.utils.visualization import plot_confusion_matrix, plot_vad_distribution, plot_vad_by_emotion, plot_tsne_vad
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data")
    data_splits = load_iemocap_data(
        args.data_path,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state
    )
    
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
    
    # Load fine-tuned VAD predictor
    logger.info(f"Loading fine-tuned VAD predictor from {args.fine_tuned_model_dir}")
    vad_predictor = VADFineTuner.load(os.path.join(args.fine_tuned_model_dir, 'model'))
    
    # Extract texts and VAD values for training emotion classifier
    logger.info("Extracting texts and VAD values for training emotion classifier")
    train_texts = train_scaled['Transcript'].tolist()
    train_emotions = train_scaled['emotion'].map(emotion_to_idx).values
    
    # Predict VAD values using fine-tuned model
    logger.info("Predicting VAD values using fine-tuned model")
    train_vad_pred = vad_predictor.predict(train_texts)
    
    # Train emotion classifier
    logger.info("Training emotion classifier")
    emotion_classifier = VADEmotionClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state
    )
    emotion_classifier.fit(train_vad_pred, train_emotions, emotion_to_idx, idx_to_emotion)
    
    # Evaluate emotion classifier on validation set
    logger.info("Evaluating emotion classifier on validation set")
    val_texts = val_scaled['Transcript'].tolist()
    val_emotions = val_scaled['emotion'].map(emotion_to_idx).values
    
    # Predict VAD values for validation set
    val_vad_pred = vad_predictor.predict(val_texts)
    
    # Evaluate emotion classifier
    emotion_metrics = evaluate_emotion_classifier(
        emotion_classifier,
        val_vad_pred,
        val_emotions,
        idx_to_emotion
    )
    
    # Log emotion metrics
    log_metrics({'emotion_metrics': emotion_metrics}, output_dir, prefix='emotion_')
    
    # Plot confusion matrix
    y_val_pred = emotion_classifier.predict(val_vad_pred)
    plot_confusion_matrix(
        [idx_to_emotion[idx] for idx in val_emotions],
        [idx_to_emotion[idx] for idx in y_val_pred],
        os.path.join(output_dir, 'plots', 'confusion_matrix.png')
    )
    
    # Plot VAD by emotion
    plot_vad_by_emotion(
        val_vad_pred,
        [idx_to_emotion[idx] for idx in val_emotions],
        os.path.join(output_dir, 'plots')
    )
    
    # Plot t-SNE visualization
    plot_tsne_vad(
        val_vad_pred,
        [idx_to_emotion[idx] for idx in val_emotions],
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
    
    # Create a summary file
    with open(os.path.join(output_dir, 'summary.md'), 'w') as f:
        f.write(f"# Fine-tuned Model Results\n\n")
        f.write(f"## Configuration\n\n")
        f.write(f"- Fine-tuned model: {args.fine_tuned_model_dir}\n")
        f.write(f"- Batch size: {args.batch_size}\n")
        f.write(f"- Random Forest estimators: {args.n_estimators}\n\n")
        
        f.write(f"## Emotion Classification Metrics\n\n")
        f.write(f"- Accuracy: {emotion_metrics['accuracy']:.4f}\n")
        f.write(f"- F1 (macro): {emotion_metrics['f1_macro']:.4f}\n")
        f.write(f"- F1 (weighted): {emotion_metrics['f1_weighted']:.4f}\n\n")
        
        f.write(f"### Classification Report\n\n")
        f.write(f"```\n")
        for class_name, metrics in emotion_metrics['classification_report'].items():
            if isinstance(metrics, dict):
                f.write(f"{class_name}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}, Support={metrics['support']}\n")
        f.write(f"```\n\n")
        
        f.write(f"## Pipeline Metrics\n\n")
        f.write(f"- Accuracy: {pipeline_metrics['emotion_metrics']['accuracy']:.4f}\n")
        f.write(f"- F1 (macro): {pipeline_metrics['emotion_metrics']['f1_macro']:.4f}\n")
        f.write(f"- F1 (weighted): {pipeline_metrics['emotion_metrics']['f1_weighted']:.4f}\n\n")
        
        f.write(f"### VAD Metrics\n\n")
        f.write(f"- MSE: {pipeline_metrics['vad_metrics']['mse']:.4f}\n")
        f.write(f"- RMSE: {pipeline_metrics['vad_metrics']['rmse']:.4f}\n")
        f.write(f"- MAE: {pipeline_metrics['vad_metrics']['mae']:.4f}\n")
    
    logger.info(f"All results saved to {output_dir}")

if __name__ == '__main__':
    main()
