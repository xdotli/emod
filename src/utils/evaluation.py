"""
Evaluation utilities for the EMOD project.

This module provides functions for evaluating emotion recognition models.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, mean_squared_error,
    mean_absolute_error, r2_score
)

def calculate_classification_metrics(y_true, y_pred, average='weighted'):
    """
    Calculate classification metrics for emotion prediction.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        average (str): Averaging method for f1, precision, and recall
        
    Returns:
        dict: Dictionary containing various metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'precision': precision_score(y_true, y_pred, average=average),
        'recall': recall_score(y_true, y_pred, average=average),
    }
    
    # Convert classification report to dict
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics['classification_report'] = report
    
    return metrics

def calculate_vad_metrics(y_true, y_pred):
    """
    Calculate regression metrics for VAD prediction.
    
    Args:
        y_true (array-like): True VAD values (shape: [n_samples, 3])
        y_pred (array-like): Predicted VAD values (shape: [n_samples, 3])
        
    Returns:
        dict: Dictionary containing MSE, RMSE, MAE, and R2 for each VAD dimension
    """
    # Ensure numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate metrics for each dimension
    mse = []
    rmse = []
    mae = []
    r2 = []
    
    for i in range(y_true.shape[1]):
        mse.append(mean_squared_error(y_true[:, i], y_pred[:, i]))
        rmse.append(np.sqrt(mse[-1]))
        mae.append(mean_absolute_error(y_true[:, i], y_pred[:, i]))
        r2.append(r2_score(y_true[:, i], y_pred[:, i]))
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def evaluate_emotion_classifier(classifier, X_test, y_test):
    """
    Evaluate an emotion classifier on test data.
    
    Args:
        classifier: Trained classifier with predict method
        X_test (array-like): Test features
        y_test (array-like): True emotion labels
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Predict emotions
    y_pred = classifier.predict(X_test)
    
    # Calculate metrics
    metrics = calculate_classification_metrics(y_test, y_pred)
    
    return metrics

def evaluate_vad_predictor(predictor, X_test, y_test):
    """
    Evaluate a VAD predictor on test data.
    
    Args:
        predictor: Trained predictor with predict method
        X_test (array-like): Test features
        y_test (array-like): True VAD values
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Predict VAD values
    y_pred = predictor.predict(X_test)
    
    # Calculate metrics
    metrics = calculate_vad_metrics(y_test, y_pred)
    
    return metrics

def evaluate_multimodal_system(vad_predictor, emotion_classifier, X_test, vad_test, emotion_test):
    """
    Evaluate a complete multimodal emotion recognition system.
    
    Args:
        vad_predictor: Trained VAD predictor
        emotion_classifier: Trained emotion classifier
        X_test (array-like): Test features
        vad_test (array-like): True VAD values
        emotion_test (array-like): True emotion labels
        
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    # Predict VAD values
    vad_pred = vad_predictor.predict(X_test)
    
    # Predict emotions from VAD values
    emotion_pred = emotion_classifier.predict(vad_pred)
    
    # Calculate VAD metrics
    vad_metrics = calculate_vad_metrics(vad_test, vad_pred)
    
    # Calculate emotion classification metrics
    emotion_metrics = calculate_classification_metrics(emotion_test, emotion_pred)
    
    # Combine all metrics
    metrics = {
        **emotion_metrics,
        'vad_metrics': vad_metrics,
        'vad_pred': vad_pred,
        'emotion_pred': emotion_pred
    }
    
    return metrics

def cross_validate_model(model, X, y, cv=5, random_state=42):
    """
    Perform cross-validation for a model.
    
    Args:
        model: Scikit-learn compatible model with fit and predict methods
        X (array-like): Features
        y (array-like): Target values
        cv (int): Number of folds
        random_state (int): Random seed for reproducibility
        
    Returns:
        dict: Dictionary containing cross-validation results
    """
    from sklearn.model_selection import cross_validate
    
    # Define scoring metrics
    if len(np.array(y).shape) > 1 and np.array(y).shape[1] > 1:
        # For VAD prediction (regression)
        scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
    else:
        # For emotion classification
        scoring = ['accuracy', 'f1_weighted', 'f1_macro', 'precision_weighted', 'recall_weighted']
    
    # Perform cross-validation
    cv_results = cross_validate(
        model, X, y, 
        cv=cv, 
        scoring=scoring,
        return_train_score=True,
        random_state=random_state
    )
    
    # Process and return results
    processed_results = {}
    for key, values in cv_results.items():
        processed_results[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values.tolist()
        }
    
    return processed_results

def confusion_matrix_analysis(y_true, y_pred, class_names=None):
    """
    Analyze confusion matrix to identify the most confused emotion pairs.
    
    Args:
        y_true (array-like): True emotion labels
        y_pred (array-like): Predicted emotion labels
        class_names (list, optional): List of class names
        
    Returns:
        dict: Dictionary containing confusion analysis results
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Get class names if not provided
    if class_names is None:
        class_names = sorted(set(y_true))
    
    # Normalize by row (true labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create a list of most confused pairs
    confused_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j:
                confused_pairs.append({
                    'true': class_names[i],
                    'pred': class_names[j],
                    'count': int(cm[i, j]),
                    'percentage': float(cm_norm[i, j])
                })
    
    # Sort by percentage (most confused first)
    confused_pairs.sort(key=lambda x: x['percentage'], reverse=True)
    
    # Calculate per-class accuracy
    class_accuracy = {}
    for i, name in enumerate(class_names):
        correct = cm[i, i]
        total = cm[i, :].sum()
        class_accuracy[name] = float(correct / total)
    
    return {
        'confusion_matrix': cm.tolist(),
        'normalized_matrix': cm_norm.tolist(),
        'class_names': class_names,
        'most_confused_pairs': confused_pairs[:10],  # Top 10 confused pairs
        'class_accuracy': class_accuracy
    }

def calculate_vad_prediction_accuracy(vad_true, vad_pred, emotion_true, 
                                     emotion_mapping_func, tolerance=0.2):
    """
    Calculate how often the VAD predictions map to the correct emotion.
    
    Args:
        vad_true (array-like): True VAD values
        vad_pred (array-like): Predicted VAD values
        emotion_true (array-like): True emotion labels
        emotion_mapping_func (callable): Function to map VAD to emotion labels
        tolerance (float): Tolerance for VAD values to be considered correct
        
    Returns:
        dict: Dictionary containing VAD-to-emotion evaluation results
    """
    # Map predicted VAD to emotions
    emotion_from_vad = emotion_mapping_func(vad_pred)
    
    # Calculate direct emotion prediction accuracy
    direct_accuracy = accuracy_score(emotion_true, emotion_from_vad)
    direct_f1 = f1_score(emotion_true, emotion_from_vad, average='weighted')
    
    # Calculate VAD-based correctness
    vad_true = np.array(vad_true)
    vad_pred = np.array(vad_pred)
    
    # Check if each dimension is within tolerance
    v_correct = np.abs(vad_true[:, 0] - vad_pred[:, 0]) <= tolerance
    a_correct = np.abs(vad_true[:, 1] - vad_pred[:, 1]) <= tolerance
    d_correct = np.abs(vad_true[:, 2] - vad_pred[:, 2]) <= tolerance
    
    # Check if all dimensions are correct simultaneously
    all_correct = v_correct & a_correct & d_correct
    
    # Calculate percentages
    v_pct = np.mean(v_correct) * 100
    a_pct = np.mean(a_correct) * 100
    d_pct = np.mean(d_correct) * 100
    all_pct = np.mean(all_correct) * 100
    
    return {
        'emotion_accuracy': direct_accuracy,
        'emotion_f1': direct_f1,
        'valence_accuracy': float(v_pct),
        'arousal_accuracy': float(a_pct),
        'dominance_accuracy': float(d_pct),
        'all_dimensions_accuracy': float(all_pct),
        'confusion_matrix': confusion_matrix(emotion_true, emotion_from_vad).tolist(),
        'tolerance': tolerance
    } 