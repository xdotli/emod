#!/usr/bin/env python3
"""
Script to run the multimodal emotion recognition system.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Multimodal Emotion Recognition System')

    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'predict'],
                        help='Mode to run the pipeline in')
    parser.add_argument('--data_path', type=str, default='IEMOCAP_Final.csv',
                        help='Path to IEMOCAP_Final.csv')
    parser.add_argument('--output_dir', type=str, default='results/multimodal',
                        help='Directory to save results')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='Directory containing saved model (for predict mode)')
    parser.add_argument('--vad_model', type=str, default='facebook/bart-large-mnli',
                        choices=['roberta-base', 'facebook/bart-large-mnli'],
                        help='Model for text VAD prediction')
    parser.add_argument('--audio_feature_type', type=str, default='mfcc',
                        choices=['mfcc', 'spectral', 'wav2vec'],
                        help='Type of audio features to extract')
    parser.add_argument('--fusion_type', type=str, default='early',
                        choices=['early', 'late'],
                        help='Type of multimodal fusion')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for processing')
    parser.add_argument('--skip_vad_eval', action='store_true',
                        help='Skip VAD evaluation (faster)')

    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()

    # Create directories if they don't exist
    os.makedirs("results/multimodal", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    if args.mode == 'train':
        # Import necessary modules
        import pandas as pd
        import numpy as np
        import torch
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        import pickle

        from data.data_loader import (
            load_iemocap_data,
            scale_vad_values,
            get_emotion_mapping
        )
        from data.data_utils import create_dataloaders
        from models.vad_predictor import BARTZeroShotVADPredictor, ZeroShotVADPredictor
        from models.audio_processor import AudioVADPredictor
        from models.multimodal_fusion import EarlyFusion, LateFusion
        from models.multimodal_pipeline import MultimodalEmotionRecognitionPipeline
        from utils.metrics import log_metrics
        from utils.visualization import plot_confusion_matrix

        # Load and preprocess data
        logger.info("Loading and preprocessing data")
        data_splits = load_iemocap_data(
            args.data_path,
            test_size=0.2,
            val_size=0.1,
            random_state=42
        )

        # Scale VAD values
        logger.info("Scaling VAD values")
        scaler_path = os.path.join(args.output_dir, 'vad_scaler.pkl')
        train_scaled, val_scaled, test_scaled, vad_scaler = scale_vad_values(
            data_splits['train'],
            data_splits['val'],
            data_splits['test'],
            scaler_path
        )

        # Get emotion mappings
        emotion_to_idx, idx_to_emotion = get_emotion_mapping(data_splits['train'])

        # Create dataloaders
        logger.info("Creating dataloaders")
        dataloaders = create_dataloaders(
            {
                'train': train_scaled,
                'val': val_scaled,
                'test': test_scaled
            },
            emotion_to_idx,
            include_audio=True,
            batch_size=args.batch_size
        )

        # Initialize text VAD predictor
        logger.info(f"Initializing text VAD predictor with {args.vad_model}")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if 'bart' in args.vad_model.lower():
            text_vad_predictor = BARTZeroShotVADPredictor(args.vad_model, device)
        else:
            text_vad_predictor = ZeroShotVADPredictor(args.vad_model, device)

        # Initialize audio VAD predictor
        logger.info(f"Initializing audio VAD predictor with {args.audio_feature_type} features")
        audio_vad_predictor = AudioVADPredictor(feature_type=args.audio_feature_type)

        # Extract audio paths and VAD values for training audio predictor
        logger.info("Extracting audio paths and VAD values for training audio predictor")
        train_audio_paths = train_scaled['Audio_Uttrance_Path'].tolist()
        train_vad_values = train_scaled[['valence', 'arousal', 'dominance']].values

        # Train audio VAD predictor
        logger.info("Training audio VAD predictor")
        audio_vad_predictor.train(
            train_audio_paths,
            train_vad_values,
            model_type='rf',
            n_estimators=100,
            random_state=42
        )

        # Initialize fusion model
        logger.info(f"Initializing {args.fusion_type} fusion model")
        if args.fusion_type == 'early':
            fusion_model = EarlyFusion(classifier_type='rf', random_state=42)
        else:
            fusion_model = LateFusion(text_weight=0.7, audio_weight=0.3)

        # Extract features for training fusion model
        logger.info("Extracting features for training fusion model")
        train_texts = train_scaled['Transcript'].tolist()
        train_emotions = train_scaled['emotion'].map(emotion_to_idx).values

        # Predict text VAD values
        logger.info("Predicting text VAD values")
        train_text_vad = text_vad_predictor.predict_vad(train_texts)

        # Predict audio VAD values
        logger.info("Predicting audio VAD values")
        train_audio_vad = audio_vad_predictor.predict(train_audio_paths)

        # Train fusion model
        logger.info("Training fusion model")
        fusion_model.fit(
            train_text_vad,
            train_audio_vad,
            train_emotions,
            emotion_to_idx,
            idx_to_emotion
        )

        # Create multimodal pipeline
        logger.info("Creating multimodal pipeline")
        pipeline = MultimodalEmotionRecognitionPipeline(
            text_vad_predictor,
            audio_vad_predictor,
            fusion_model,
            vad_scaler
        )

        # Evaluate pipeline on validation set
        logger.info("Evaluating pipeline on validation set")
        val_metrics = pipeline.evaluate(dataloaders['val'])

        # Log validation metrics
        log_metrics(val_metrics, args.output_dir, prefix='val_')

        # Plot confusion matrix
        plot_confusion_matrix(
            val_metrics['true_emotion_labels'],
            val_metrics['pred_emotion_labels'],
            os.path.join(args.output_dir, 'val_confusion_matrix.png')
        )

        # Evaluate pipeline on test set
        logger.info("Evaluating pipeline on test set")
        test_metrics = pipeline.evaluate(dataloaders['test'])

        # Log test metrics
        log_metrics(test_metrics, args.output_dir, prefix='test_')

        # Plot confusion matrix
        plot_confusion_matrix(
            test_metrics['true_emotion_labels'],
            test_metrics['pred_emotion_labels'],
            os.path.join(args.output_dir, 'test_confusion_matrix.png')
        )

        # Save pipeline
        logger.info("Saving pipeline")
        pipeline.save(args.output_dir)

        logger.info(f"All results saved to {args.output_dir}")

    elif args.mode == 'predict':
        # Check if model directory is provided
        if not args.model_dir:
            logger.error("Model directory must be provided in predict mode")
            sys.exit(1)

        # Import necessary modules
        import torch
        import pandas as pd
        import numpy as np
        from models.multimodal_pipeline import MultimodalEmotionRecognitionPipeline

        # Load the pipeline
        logger.info(f"Loading pipeline from {args.model_dir}")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pipeline = MultimodalEmotionRecognitionPipeline.load(args.model_dir, args.vad_model, device)

        # Load test data
        logger.info(f"Loading test data from {args.data_path}")
        df = pd.read_csv(args.data_path)

        # Use a sample of the data for prediction
        sample_size = min(100, len(df))
        sample_df = df.sample(sample_size, random_state=42)

        # Extract texts and audio paths
        texts = sample_df['Transcript'].tolist()
        audio_paths = sample_df['Audio_Uttrance_Path'].tolist()

        # Predict emotions
        logger.info("Predicting emotions")
        emotions, text_vad, audio_vad = pipeline.predict(texts, audio_paths, batch_size=args.batch_size)

        # Create results DataFrame
        results_df = pd.DataFrame({
            'text': texts,
            'audio_path': audio_paths,
            'predicted_emotion': emotions,
            'text_valence': text_vad[:, 0],
            'text_arousal': text_vad[:, 1],
            'text_dominance': text_vad[:, 2],
            'audio_valence': audio_vad[:, 0],
            'audio_arousal': audio_vad[:, 1],
            'audio_dominance': audio_vad[:, 2]
        })

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(args.output_dir, f'predict_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)

        results_path = os.path.join(output_dir, 'predictions.csv')
        results_df.to_csv(results_path, index=False)

        logger.info(f"Predictions saved to {results_path}")

        # Print sample predictions
        print("\nSample predictions:")
        for i in range(min(5, len(results_df))):
            print(f"Text: {results_df.iloc[i]['text']}")
            print(f"Emotion: {results_df.iloc[i]['predicted_emotion']}")
            print(f"Text VAD: ({results_df.iloc[i]['text_valence']:.2f}, {results_df.iloc[i]['text_arousal']:.2f}, {results_df.iloc[i]['text_dominance']:.2f})")
            print(f"Audio VAD: ({results_df.iloc[i]['audio_valence']:.2f}, {results_df.iloc[i]['audio_arousal']:.2f}, {results_df.iloc[i]['audio_dominance']:.2f})")
            print()

if __name__ == '__main__':
    main()
