#!/usr/bin/env python3
"""
Script to fine-tune a pre-trained language model for VAD prediction.
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
    parser = argparse.ArgumentParser(description='Fine-tune a pre-trained language model for VAD prediction')

    parser.add_argument('--data_path', type=str, default='IEMOCAP_Final.csv',
                        help='Path to IEMOCAP_Final.csv')
    parser.add_argument('--output_dir', type=str, default='results/fine_tuning',
                        help='Directory to save results')
    parser.add_argument('--model_name', type=str, default='roberta-base',
                        choices=['roberta-base', 'bert-base-uncased', 'distilbert-base-uncased'],
                        help='Name of the pre-trained model')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate for optimization')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length')
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
    from data.data_loader import load_iemocap_data, scale_vad_values
    from models.vad_fine_tuner import VADFineTuner

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

    # Extract texts and VAD values
    train_texts = train_scaled['Transcript'].tolist()
    train_vad = train_scaled[['valence', 'arousal', 'dominance']].values

    val_texts = val_scaled['Transcript'].tolist()
    val_vad = val_scaled[['valence', 'arousal', 'dominance']].values

    test_texts = test_scaled['Transcript'].tolist()
    test_vad = test_scaled[['valence', 'arousal', 'dominance']].values

    # Initialize fine-tuner
    logger.info(f"Initializing fine-tuner with {args.model_name}")
    fine_tuner = VADFineTuner(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_length=args.max_length
    )

    # Train the model
    logger.info("Training the model")
    history = fine_tuner.train(
        train_texts,
        train_vad,
        val_texts,
        val_vad,
        epochs=args.epochs,
        output_dir=output_dir
    )

    # Evaluate on test set
    logger.info("Evaluating on test set")
    test_preds = fine_tuner.predict(test_texts)

    # Calculate test metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import numpy as np

    test_mse = mean_squared_error(test_vad, test_preds)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(test_vad, test_preds)

    # Calculate metrics for each dimension
    dim_metrics = {}
    for i, dim in enumerate(['valence', 'arousal', 'dominance']):
        dim_mse = mean_squared_error(test_vad[:, i], test_preds[:, i])
        dim_rmse = np.sqrt(dim_mse)
        dim_mae = mean_absolute_error(test_vad[:, i], test_preds[:, i])

        dim_metrics[dim] = {
            'mse': float(dim_mse),
            'rmse': float(dim_rmse),
            'mae': float(dim_mae)
        }

    # Save test metrics
    test_metrics = {
        'mse': float(test_mse),
        'rmse': float(test_rmse),
        'mae': float(test_mae),
        'dim_metrics': dim_metrics
    }

    with open(os.path.join(output_dir, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)

    # Print test metrics
    logger.info(f"Test MSE: {test_mse:.4f}")
    logger.info(f"Test RMSE: {test_rmse:.4f}")
    logger.info(f"Test MAE: {test_mae:.4f}")

    for dim, metrics in dim_metrics.items():
        logger.info(f"{dim.capitalize()} MSE: {metrics['mse']:.4f}")
        logger.info(f"{dim.capitalize()} RMSE: {metrics['rmse']:.4f}")
        logger.info(f"{dim.capitalize()} MAE: {metrics['mae']:.4f}")

    # Save the model
    logger.info("Saving the model")
    fine_tuner.save(os.path.join(output_dir, 'model'))

    # Save test predictions
    import pandas as pd

    test_df = pd.DataFrame({
        'text': test_texts,
        'true_valence': test_vad[:, 0],
        'true_arousal': test_vad[:, 1],
        'true_dominance': test_vad[:, 2],
        'pred_valence': test_preds[:, 0],
        'pred_arousal': test_preds[:, 1],
        'pred_dominance': test_preds[:, 2]
    })

    test_df.to_csv(os.path.join(output_dir, 'test_predictions.csv'), index=False)

    logger.info(f"All results saved to {output_dir}")

    # Create a summary file
    with open(os.path.join(output_dir, 'summary.md'), 'w') as f:
        f.write(f"# Fine-tuning Results\n\n")
        f.write(f"## Configuration\n\n")
        f.write(f"- Model: {args.model_name}\n")
        f.write(f"- Batch size: {args.batch_size}\n")
        f.write(f"- Learning rate: {args.learning_rate}\n")
        f.write(f"- Epochs: {args.epochs}\n")
        f.write(f"- Max length: {args.max_length}\n\n")

        f.write(f"## Test Metrics\n\n")
        f.write(f"- MSE: {test_mse:.4f}\n")
        f.write(f"- RMSE: {test_rmse:.4f}\n")
        f.write(f"- MAE: {test_mae:.4f}\n\n")

        f.write(f"### Dimension-specific Metrics\n\n")
        for dim, metrics in dim_metrics.items():
            f.write(f"#### {dim.capitalize()}\n\n")
            f.write(f"- MSE: {metrics['mse']:.4f}\n")
            f.write(f"- RMSE: {metrics['rmse']:.4f}\n")
            f.write(f"- MAE: {metrics['mae']:.4f}\n\n")

        f.write(f"## Training History\n\n")
        f.write(f"See `history.json` for detailed training history.\n\n")
        f.write(f"### Final Validation Metrics\n\n")
        f.write(f"- Loss: {history['val_loss'][-1]:.4f}\n")
        f.write(f"- MSE: {history['val_mse'][-1]:.4f}\n")
        f.write(f"- RMSE: {history['val_rmse'][-1]:.4f}\n")
        f.write(f"- MAE: {history['val_mae'][-1]:.4f}\n")

if __name__ == '__main__':
    main()
