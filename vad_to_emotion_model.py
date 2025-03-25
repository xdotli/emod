#!/usr/bin/env python3
"""
VAD to Emotion Classifier

This script trains and evaluates several machine learning models to map VAD (valence-arousal-dominance)
tuples to emotion categories. It replaces the rule-based approach with a data-driven one.

The models include:
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Gradient Boosting
- XGBoost

Author: AI Assistant
Date: March 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import xgboost as xgb
from tabulate import tabulate

def load_data(file_path):
    """
    Load the dataset from a CSV file.
    
    Args:
        file_path: Path to the CSV file containing VAD values and emotion labels
        
    Returns:
        X: VAD features
        y: Emotion labels
        label_encoder: Label encoder for emotion categories
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Check if VAD columns exist
    if all(col in df.columns for col in ['valence', 'arousal', 'dominance', 'emotion']):
        X = df[['valence', 'arousal', 'dominance']].values
        
        # Encode emotion labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df['emotion'].values)
        
        # Print statistics
        print(f"Dataset contains {len(df)} samples")
        print("\nEmotion distribution:")
        emotion_counts = df['emotion'].value_counts()
        print(tabulate(
            [[emotion, count, f"{count/len(df)*100:.2f}%", label_encoder.transform([emotion])[0]] 
             for emotion, count in emotion_counts.items()],
            headers=["Emotion", "Count", "Percentage", "Encoded"],
            tablefmt="grid"
        ))
        
        # Print VAD statistics
        print("\nVAD statistics:")
        vad_stats = df[['valence', 'arousal', 'dominance']].describe()
        print(tabulate(
            vad_stats,
            headers="keys",
            tablefmt="grid",
            floatfmt=".4f"
        ))
        
        return X, y, label_encoder, df['emotion'].values
    else:
        raise ValueError("Dataset does not contain required VAD and emotion columns")

def visualize_vad_distribution(X, y_labels):
    """
    Visualize the distribution of emotions in VAD space.
    
    Args:
        X: VAD features
        y_labels: Emotion labels (strings)
    """
    try:
        # Create a DataFrame for plotting
        df = pd.DataFrame(X, columns=['valence', 'arousal', 'dominance'])
        df['emotion'] = y_labels
        
        # 3D scatter plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each emotion category with different colors
        for emotion in df['emotion'].unique():
            subset = df[df['emotion'] == emotion]
            ax.scatter(
                subset['valence'], 
                subset['arousal'], 
                subset['dominance'],
                label=emotion,
                alpha=0.7
            )
        
        ax.set_xlabel('Valence')
        ax.set_ylabel('Arousal')
        ax.set_zlabel('Dominance')
        ax.set_title('Emotion Distribution in VAD Space')
        ax.legend()
        
        # Set limits
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        
        plt.savefig('vad_emotion_distribution.png')
        print("Visualization saved as 'vad_emotion_distribution.png'")
        
        # Also create 2D projections for better visualization
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        # Valence-Arousal plot
        for emotion in df['emotion'].unique():
            subset = df[df['emotion'] == emotion]
            axs[0].scatter(subset['valence'], subset['arousal'], label=emotion, alpha=0.7)
        axs[0].set_xlabel('Valence')
        axs[0].set_ylabel('Arousal')
        axs[0].set_title('Valence-Arousal Projection')
        axs[0].grid(True)
        axs[0].legend()
        
        # Valence-Dominance plot
        for emotion in df['emotion'].unique():
            subset = df[df['emotion'] == emotion]
            axs[1].scatter(subset['valence'], subset['dominance'], label=emotion, alpha=0.7)
        axs[1].set_xlabel('Valence')
        axs[1].set_ylabel('Dominance')
        axs[1].set_title('Valence-Dominance Projection')
        axs[1].grid(True)
        
        # Arousal-Dominance plot
        for emotion in df['emotion'].unique():
            subset = df[df['emotion'] == emotion]
            axs[2].scatter(subset['arousal'], subset['dominance'], label=emotion, alpha=0.7)
        axs[2].set_xlabel('Arousal')
        axs[2].set_ylabel('Dominance')
        axs[2].set_title('Arousal-Dominance Projection')
        axs[2].grid(True)
        
        plt.tight_layout()
        plt.savefig('vad_2d_projections.png')
        print("2D projections saved as 'vad_2d_projections.png'")
    except Exception as e:
        print(f"Error creating visualizations: {e}")

def train_models(X, y, label_encoder):
    """
    Train multiple models on the VAD-to-emotion data and select the best one.
    
    Args:
        X: VAD features
        y: Encoded emotion labels (integers)
        label_encoder: Label encoder for emotion categories
        
    Returns:
        best_model: The best performing model
        best_model_name: Name of the best model
        X_test: Test features
        y_test: Test labels (encoded)
        scaler: Feature scaler
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    }
    
    # Parameters for grid search
    param_grids = {
        'Decision Tree': {
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5, 10],
            'criterion': ['gini', 'entropy']
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10]
        },
        'SVM': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'kernel': ['rbf', 'linear', 'poly']
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    }
    
    # Train and evaluate models
    results = []
    best_model = None
    best_accuracy = 0
    best_model_name = ""
    
    print("\nTraining and evaluating models...")
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Use smaller param grid for faster execution
        param_grid = {
            k: v[:2] for k, v in param_grids[name].items()
        }
        
        try:
            # Train with grid search
            grid_search = GridSearchCV(
                model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train_scaled, y_train)
            
            # Get best model
            best_params = grid_search.best_params_
            trained_model = grid_search.best_estimator_
            
            # Evaluate
            y_pred = trained_model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Save results
            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'Best Parameters': best_params
            })
            
            print(f"{name} Accuracy: {accuracy:.4f}")
            print(f"Best parameters: {best_params}")
            
            # Keep track of best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = trained_model
                best_model_name = name
        except Exception as e:
            print(f"Error training {name}: {e}")
    
    # Print results
    print("\nModel Comparison:")
    results_df = pd.DataFrame(results)
    print(tabulate(
        results_df[['Model', 'Accuracy']],
        headers='keys',
        tablefmt='grid',
        floatfmt='.4f'
    ))
    
    print(f"\nBest model: {best_model_name} with accuracy: {best_accuracy:.4f}")
    
    # Save scaler with the model
    joblib.dump(scaler, 'vad_scaler.pkl')
    
    return best_model, best_model_name, X_test_scaled, y_test, scaler, label_encoder

def evaluate_model(model, model_name, X_test, y_test, label_encoder):
    """
    Evaluate the model and create visualizations.
    
    Args:
        model: Trained model
        model_name: Name of the model
        X_test: Test features
        y_test: Test labels (encoded)
        label_encoder: Label encoder for emotion categories
    """
    # Get emotion labels
    class_names = label_encoder.classes_
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Convert encoded predictions and labels back to strings for better reporting
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    y_test_labels = label_encoder.inverse_transform(y_test)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test_labels, y_pred_labels))
    
    # Confusion matrix
    cm = confusion_matrix(y_test_labels, y_pred_labels)
    
    # Print tabular confusion matrix
    print("\nConfusion Matrix:")
    print(tabulate(
        cm,
        headers=class_names,
        showindex=class_names,
        tablefmt="grid"
    ))
    
    # Try to create a visual confusion matrix
    try:
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True Emotion')
        plt.xlabel('Predicted Emotion')
        plt.savefig('confusion_matrix_vad2emotion.png')
        print("Confusion matrix visualization saved as 'confusion_matrix_vad2emotion.png'")
    except Exception as e:
        print(f"Could not create confusion matrix visualization: {e}")
    
    # Visualize decision boundaries if it's a tree-based model
    if model_name == 'Decision Tree':
        try:
            plt.figure(figsize=(20, 10))
            plot_tree(model, filled=True, feature_names=['Valence', 'Arousal', 'Dominance'], 
                    class_names=list(class_names), rounded=True, proportion=False)
            plt.savefig('decision_tree_vad2emotion.png')
            print("Decision tree visualization saved as 'decision_tree_vad2emotion.png'")
        except Exception as e:
            print(f"Could not create decision tree visualization: {e}")
    
    # Save feature importances for tree-based models
    if model_name in ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'XGBoost']:
        try:
            importances = model.feature_importances_
            features = ['Valence', 'Arousal', 'Dominance']
            
            plt.figure(figsize=(10, 6))
            plt.bar(features, importances)
            plt.title(f'Feature Importance - {model_name}')
            plt.ylabel('Importance')
            plt.xlabel('Feature')
            plt.savefig('feature_importance_vad2emotion.png')
            print("Feature importance visualization saved as 'feature_importance_vad2emotion.png'")
            
            # Also print feature importances
            print("\nFeature Importances:")
            for feature, importance in zip(features, importances):
                print(f"{feature}: {importance:.4f}")
        except Exception as e:
            print(f"Could not create feature importance visualization: {e}")

def save_model(model, model_name, scaler, label_encoder):
    """
    Save the trained model and scaler for later use.
    
    Args:
        model: Trained model
        model_name: Name of the model
        scaler: Feature scaler
        label_encoder: Label encoder for emotion categories
    """
    # Save the model
    model_filename = 'vad_to_emotion_model.pkl'
    joblib.dump(model, model_filename)
    print(f"\nModel saved as '{model_filename}'")
    
    # Save label encoder
    joblib.dump(label_encoder, 'vad_label_encoder.pkl')
    print(f"Label encoder saved as 'vad_label_encoder.pkl'")
    
    # Save a model info file with metadata
    model_info = {
        'model_name': model_name,
        'features': ['valence', 'arousal', 'dominance'],
        'target': 'emotion',
        'classes': list(label_encoder.classes_),
        'date_trained': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save as json
    import json
    with open('vad_to_emotion_model_info.json', 'w') as f:
        json.dump(model_info, f, indent=4)
    print("Model metadata saved as 'vad_to_emotion_model_info.json'")

def predict_emotion_from_vad(vad_values, model=None, scaler=None, label_encoder=None):
    """
    Predict emotion from VAD values using the trained model.
    
    Args:
        vad_values: List or array of [valence, arousal, dominance] values
        model: Pre-loaded model (optional)
        scaler: Pre-loaded scaler (optional)
        label_encoder: Pre-loaded label encoder (optional)
        
    Returns:
        emotion: Predicted emotion category
        probabilities: Probability distribution over all emotions
    """
    if model is None:
        try:
            model = joblib.load('vad_to_emotion_model.pkl')
        except:
            raise ValueError("Model not found. Please train a model first.")
    
    if scaler is None:
        try:
            scaler = joblib.load('vad_scaler.pkl')
        except:
            raise ValueError("Scaler not found. Please train a model first.")
    
    if label_encoder is None:
        try:
            label_encoder = joblib.load('vad_label_encoder.pkl')
        except:
            raise ValueError("Label encoder not found. Please train a model first.")
    
    # Reshape input for single prediction
    vad_values = np.array(vad_values).reshape(1, -1)
    
    # Scale the input
    vad_values_scaled = scaler.transform(vad_values)
    
    # Predict
    emotion_encoded = model.predict(vad_values_scaled)[0]
    emotion = label_encoder.inverse_transform([emotion_encoded])[0]
    
    # Get probabilities if the model supports it
    try:
        probabilities = model.predict_proba(vad_values_scaled)[0]
        prob_dict = dict(zip(label_encoder.classes_, probabilities))
    except:
        prob_dict = {emotion: 1.0}
    
    return emotion, prob_dict

def main():
    """Main function to train and evaluate VAD-to-emotion models."""
    print("=" * 80)
    print("VAD TO EMOTION MODEL TRAINING")
    print("=" * 80)
    
    # Path to dataset
    data_path = "processed_data.csv"
    
    try:
        # Load data
        X, y, label_encoder, y_labels = load_data(data_path)
        
        # Visualize data
        visualize_vad_distribution(X, y_labels)
        
        # Train models
        best_model, best_model_name, X_test, y_test, scaler, label_encoder = train_models(X, y, label_encoder)
        
        # Evaluate best model
        evaluate_model(best_model, best_model_name, X_test, y_test, label_encoder)
        
        # Save model
        save_model(best_model, best_model_name, scaler, label_encoder)
        
        # Test with sample values
        print("\nTesting model with sample VAD values:")
        
        test_vad_values = [
            [0.8, 0.7, 0.6],   # Happy
            [-0.8, -0.5, -0.6], # Sad
            [-0.6, 0.8, 0.7],  # Angry
            [0.0, 0.0, 0.0]    # Neutral
        ]
        
        for vad in test_vad_values:
            emotion, probs = predict_emotion_from_vad(vad, best_model, scaler, label_encoder)
            print(f"VAD {vad} → Emotion: {emotion}")
            
            # Print top 3 probabilities
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
            for emotion, prob in sorted_probs:
                print(f"  {emotion}: {prob:.4f}")
        
        print("\nVAD-to-Emotion model training completed!")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error: {e}")

# Function to be imported by main.py to replace the rule-based approach
def get_vad_to_emotion_predictor():
    """
    Return a callable function that predicts emotions from VAD values.
    
    Returns:
        function: A function that takes a tuple of (valence, arousal, dominance) and returns an emotion.
    """
    try:
        # Load the model
        model = joblib.load('vad_to_emotion_model.pkl')
        scaler = joblib.load('vad_scaler.pkl')
        label_encoder = joblib.load('vad_label_encoder.pkl')
        
        # Define the predictor function
        def predict(valence, arousal, dominance):
            """Predict emotion from VAD values"""
            vad_values = [valence, arousal, dominance]
            emotion, _ = predict_emotion_from_vad(vad_values, model, scaler, label_encoder)
            return emotion
        
        print("Loaded machine learning VAD-to-emotion model!")
        return predict
    except Exception as e:
        print(f"Could not load ML model: {e}")
        print("Falling back to rule-based VAD-to-emotion mapping")
        
        # Return a version of the original rule-based approach if model loading fails
        def rule_based_vad_to_emotion(valence, arousal, dominance):
            """Rule-based VAD to emotion mapping (fallback)"""
            if valence > 0:
                if arousal > 0:
                    if dominance > 0:
                        return "happy"
                    else:
                        return "excited"
                else:
                    if dominance > 0:
                        return "content"
                    else:
                        return "relaxed"
            else:
                if arousal > 0:
                    if dominance > 0:
                        return "angry"
                    else:
                        return "fearful"
                else:
                    if dominance > 0:
                        return "disgusted"
                    else:
                        return "sad"
        
        return rule_based_vad_to_emotion

if __name__ == "__main__":
    main() 