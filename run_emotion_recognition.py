#!/usr/bin/env python3
"""
Script to run the two-stage emotion recognition system.
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
    parser = argparse.ArgumentParser(description='Two-Stage Emotion Recognition System')
    
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'predict'],
                        help='Mode to run the pipeline in')
    parser.add_argument('--data_path', type=str, default='IEMOCAP_Final.csv',
                        help='Path to IEMOCAP_Final.csv')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='Directory containing saved model (for predict mode)')
    parser.add_argument('--vad_model', type=str, default='facebook/bart-large-mnli',
                        choices=['roberta-base', 'facebook/bart-large-mnli'],
                        help='Model for VAD prediction')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for processing')
    parser.add_argument('--skip_vad_eval', action='store_true',
                        help='Skip VAD evaluation (faster)')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Create directories if they don't exist
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    if args.mode == 'train':
        # Import and run the training script
        from src.main import main as train_main
        sys.argv = [
            'src/main.py',
            f'--data_path={args.data_path}',
            f'--output_dir={args.output_dir}',
            f'--vad_model={args.vad_model}',
            f'--batch_size={args.batch_size}'
        ]
        if args.skip_vad_eval:
            sys.argv.append('--skip_vad_eval')
        train_main()
    elif args.mode == 'predict':
        # Check if model directory is provided
        if not args.model_dir:
            logger.error("Model directory must be provided in predict mode")
            sys.exit(1)
        
        # Import necessary modules
        import torch
        import pandas as pd
        from src.models.pipeline import EmotionRecognitionPipeline
        
        # Load the pipeline
        logger.info(f"Loading pipeline from {args.model_dir}")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pipeline = EmotionRecognitionPipeline.load(args.model_dir, args.vad_model, device)
        
        # Load test data
        logger.info(f"Loading test data from {args.data_path}")
        df = pd.read_csv(args.data_path)
        
        # Use a sample of the data for prediction
        sample_size = min(100, len(df))
        sample_df = df.sample(sample_size, random_state=42)
        
        # Extract texts
        texts = sample_df['Transcript'].tolist()
        
        # Predict emotions
        logger.info("Predicting emotions")
        emotions, vad_values = pipeline.predict(texts, batch_size=args.batch_size)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'text': texts,
            'predicted_emotion': emotions,
            'valence': vad_values[:, 0],
            'arousal': vad_values[:, 1],
            'dominance': vad_values[:, 2]
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
            print(f"VAD: ({results_df.iloc[i]['valence']:.2f}, {results_df.iloc[i]['arousal']:.2f}, {results_df.iloc[i]['dominance']:.2f})")
            print()

if __name__ == '__main__':
    main()
