#!/usr/bin/env python3
"""
Script to evaluate emotion classification with reduced categories using true VAD values.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function."""
    # Load data
    logger.info("Loading data from IEMOCAP_Reduced.csv")
    df = pd.read_csv('IEMOCAP_Reduced.csv')
    
    # Create output directory
    output_dir = 'results/reduced_categories_eval'
    os.makedirs(output_dir, exist_ok=True)
    
    # Split data
    logger.info("Splitting data into train, validation, and test sets")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1/(1-0.2), random_state=42)
    
    logger.info(f"Data split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
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
    
    # Prepare data
    X_train = train_df[['valence', 'arousal', 'dominance']].values
    y_train = train_df['emotion'].map(emotion_to_idx).values
    
    X_val = val_df[['valence', 'arousal', 'dominance']].values
    y_val = val_df['emotion'].map(emotion_to_idx).values
    
    X_test = test_df[['valence', 'arousal', 'dominance']].values
    y_test = test_df['emotion'].map(emotion_to_idx).values
    
    # Train classifier
    logger.info("Training Random Forest classifier")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # Get feature importances
    importances = clf.feature_importances_
    logger.info(f"Feature importances: valence={importances[0]:.4f}, arousal={importances[1]:.4f}, dominance={importances[2]:.4f}")
    
    # Evaluate on validation set
    logger.info("Evaluating on validation set")
    y_val_pred = clf.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_report = classification_report(y_val, y_val_pred, target_names=unique_emotions, output_dict=True)
    
    logger.info(f"Validation accuracy: {val_accuracy:.4f}")
    
    # Save validation metrics
    with open(os.path.join(output_dir, 'validation_metrics.json'), 'w') as f:
        json.dump({
            'accuracy': val_accuracy,
            'classification_report': val_report
        }, f, indent=2)
    
    # Plot confusion matrix
    cm = confusion_matrix(y_val, y_val_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=unique_emotions, yticklabels=unique_emotions)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Validation Set)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_val.png'))
    plt.close()
    
    # Evaluate on test set
    logger.info("Evaluating on test set")
    y_test_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_report = classification_report(y_test, y_test_pred, target_names=unique_emotions, output_dict=True)
    
    logger.info(f"Test accuracy: {test_accuracy:.4f}")
    
    # Save test metrics
    with open(os.path.join(output_dir, 'test_metrics.json'), 'w') as f:
        json.dump({
            'accuracy': test_accuracy,
            'classification_report': test_report
        }, f, indent=2)
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=unique_emotions, yticklabels=unique_emotions)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Test Set)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_test.png'))
    plt.close()
    
    # Print summary
    print("\nSummary of Results with Reduced Categories (4 emotions):")
    print("-------------------------------------------------------")
    print(f"Stage 2 (VAD to Emotion) Validation Accuracy: {val_accuracy:.4f}")
    print(f"Stage 2 (VAD to Emotion) Test Accuracy: {test_accuracy:.4f}")
    print("\nPerformance by Emotion (Validation Set):")
    for emotion, metrics in val_report.items():
        if isinstance(metrics, dict) and 'f1-score' in metrics:
            print(f"  {emotion}: F1 = {metrics['f1-score']:.4f}, Support = {metrics['support']}")
    
    # Compare with random guessing and majority class baseline
    random_baseline = 1 / len(unique_emotions)
    majority_class = df['emotion'].value_counts(normalize=True).max()
    
    print("\nBaselines:")
    print(f"  Random Guessing: {random_baseline:.4f}")
    print(f"  Majority Class: {majority_class:.4f}")
    
    # Calculate improvement
    improvement_over_random = (test_accuracy - random_baseline) / random_baseline * 100
    improvement_over_majority = (test_accuracy - majority_class) / majority_class * 100
    
    print("\nImprovement:")
    print(f"  Over Random Guessing: {improvement_over_random:.2f}%")
    print(f"  Over Majority Class: {improvement_over_majority:.2f}%")

if __name__ == '__main__':
    main()
