"""
Emotion Classification Module

This module handles the second stage of the emotion recognition system:
classifying emotions based on VAD (Valence-Arousal-Dominance) values.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmotionClassifier:
    """Classifier for mapping VAD values to emotion categories."""
    
    def __init__(self, classifier_type='ensemble', random_state=42):
        """
        Initialize the emotion classifier.
        
        Args:
            classifier_type (str): The type of classifier to use ('ensemble', 'rf', 'nb', 'svm')
            random_state (int): Random seed for reproducibility
        """
        self.classifier_type = classifier_type
        self.random_state = random_state
        self.model = None
        
        # Initialize the classifier based on the specified type
        if classifier_type == 'ensemble':
            # Ensemble of Random Forest and Naive Bayes
            self.model = VotingClassifier(
                estimators=[
                    ('rf', RandomForestClassifier(n_estimators=100, random_state=random_state)),
                    ('nb', GaussianNB())
                ],
                voting='soft'
            )
        elif classifier_type == 'rf':
            # Random Forest classifier
            self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        elif classifier_type == 'nb':
            # Naive Bayes classifier
            self.model = GaussianNB()
        elif classifier_type == 'svm':
            # Support Vector Machine classifier
            self.model = SVC(probability=True, random_state=random_state)
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")
    
    def fit(self, X, y):
        """
        Train the classifier.
        
        Args:
            X (ndarray): Input features (VAD values)
            y (ndarray): Target emotion labels
            
        Returns:
            self: The trained classifier
        """
        logger.info(f"Training {self.classifier_type} classifier on {len(X)} samples")
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """
        Predict emotion labels from VAD values.
        
        Args:
            X (ndarray): Input features (VAD values)
            
        Returns:
            ndarray: Predicted emotion labels
        """
        if self.model is None:
            raise ValueError("Classifier not trained. Call fit() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict emotion probabilities from VAD values.
        
        Args:
            X (ndarray): Input features (VAD values)
            
        Returns:
            ndarray: Predicted emotion probabilities
        """
        if self.model is None:
            raise ValueError("Classifier not trained. Call fit() first.")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X, y_true):
        """
        Evaluate the classifier performance.
        
        Args:
            X (ndarray): Input features (VAD values)
            y_true (ndarray): True emotion labels
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Classifier not trained. Call fit() first.")
        
        y_pred = self.predict(X)
        
        accuracy = accuracy_score(y_true, y_pred)
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Weighted F1 Score: {f1_weighted:.4f}")
        logger.info(f"Macro F1 Score: {f1_macro:.4f}")
        
        # Create classification report
        report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
        
        # Return metrics
        metrics = {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'classification_report': report
        }
        
        return metrics
    
    def save(self, path):
        """
        Save the trained classifier to disk.
        
        Args:
            path (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("Classifier not trained. Call fit() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path):
        """
        Load a trained classifier from disk.
        
        Args:
            path (str): Path to the saved model
            
        Returns:
            EmotionClassifier: Loaded classifier
        """
        # Create a new instance
        instance = cls()
        
        # Load the model
        instance.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        
        return instance

def get_emotion_from_vad(vad_values, thresholds=None):
    """
    Map VAD values to emotions based on simple thresholds.
    This is a rule-based approach that does not require training.
    
    Args:
        vad_values (ndarray): VAD values (shape: [n_samples, 3])
        thresholds (dict, optional): Custom thresholds for emotion mapping
        
    Returns:
        list: Predicted emotion labels
    """
    if thresholds is None:
        # Default thresholds based on literature
        thresholds = {
            'valence_mid': 3.0,
            'arousal_mid': 3.0,
            'dominance_mid': 3.0
        }
    
    # Ensure vad_values is a numpy array
    vad_values = np.array(vad_values)
    
    # Get dimensions
    valence = vad_values[:, 0]
    arousal = vad_values[:, 1]
    dominance = vad_values[:, 2]
    
    # Map to emotions
    emotions = []
    for v, a, d in zip(valence, arousal, dominance):
        if v > thresholds['valence_mid'] and a > thresholds['arousal_mid']:
            emotions.append('happy')
        elif v < thresholds['valence_mid'] and a > thresholds['arousal_mid']:
            emotions.append('angry')
        elif v < thresholds['valence_mid'] and a < thresholds['arousal_mid']:
            emotions.append('sad')
        else:
            emotions.append('neutral')
    
    return emotions

def plot_feature_importance(model, feature_names=None):
    """
    Plot feature importance for RandomForest-based classifiers.
    
    Args:
        model: Trained classifier model
        feature_names (list, optional): Names of the features
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    import matplotlib.pyplot as plt
    
    # Check if the model has feature_importances_ attribute
    if not hasattr(model, 'feature_importances_'):
        if hasattr(model, 'estimators_') and 'rf' in model.estimators_:
            # For VotingClassifier, get the RandomForest estimator
            model = model.estimators_['rf']
        else:
            raise ValueError("Model does not support feature importance visualization")
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Set default feature names if not provided
    if feature_names is None:
        feature_names = ['Valence', 'Arousal', 'Dominance']
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(range(len(importances)), importances[indices], align='center')
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    ax.set_title('Feature Importance for Emotion Classification')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Importance')
    
    plt.tight_layout()
    
    return fig

def analyze_emotion_vad_distribution(vad_values, emotion_labels):
    """
    Analyze and visualize the distribution of VAD values for different emotions.
    
    Args:
        vad_values (ndarray): VAD values (shape: [n_samples, 3])
        emotion_labels (list): Corresponding emotion labels
        
    Returns:
        tuple: (DataFrame of statistics, matplotlib.figure.Figure)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create DataFrame
    df = pd.DataFrame({
        'Valence': vad_values[:, 0],
        'Arousal': vad_values[:, 1],
        'Dominance': vad_values[:, 2],
        'Emotion': emotion_labels
    })
    
    # Compute statistics
    stats = df.groupby('Emotion').agg({
        'Valence': ['mean', 'std', 'min', 'max'],
        'Arousal': ['mean', 'std', 'min', 'max'],
        'Dominance': ['mean', 'std', 'min', 'max']
    })
    
    # Create visualizations
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Valence distribution
    sns.boxplot(x='Emotion', y='Valence', data=df, ax=axes[0])
    axes[0].set_title('Valence by Emotion')
    
    # Arousal distribution
    sns.boxplot(x='Emotion', y='Arousal', data=df, ax=axes[1])
    axes[1].set_title('Arousal by Emotion')
    
    # Dominance distribution
    sns.boxplot(x='Emotion', y='Dominance', data=df, ax=axes[2])
    axes[2].set_title('Dominance by Emotion')
    
    plt.tight_layout()
    
    return stats, fig 