"""
Emotion classifier based on VAD values.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

class VADEmotionDataset(Dataset):
    """
    Dataset for emotion classification from VAD values.
    """
    def __init__(self, vad_values, emotion_labels):
        """
        Initialize the dataset.
        
        Args:
            vad_values: Array of VAD values (valence, arousal, dominance)
            emotion_labels: Array of emotion labels
        """
        self.vad_values = vad_values
        self.emotion_labels = emotion_labels
    
    def __len__(self):
        return len(self.vad_values)
    
    def __getitem__(self, idx):
        return {
            'vad_values': torch.tensor(self.vad_values[idx], dtype=torch.float),
            'emotion': self.emotion_labels[idx]
        }

class EmotionClassifier(nn.Module):
    """
    Emotion classifier based on VAD values.
    """
    def __init__(self, num_classes=4, dropout=0.3):
        """
        Initialize the model.
        
        Args:
            num_classes: Number of emotion classes
            dropout: Dropout rate
        """
        super(EmotionClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(3, 64),  # 3 VAD values
            nn.LayerNorm(64),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input VAD values
            
        Returns:
            Emotion class logits
        """
        return self.classifier(x)

class RuleBasedEmotionClassifier:
    """
    Rule-based emotion classifier based on VAD values.
    """
    def __init__(self):
        """
        Initialize the classifier with VAD thresholds for each emotion.
        """
        # Define VAD thresholds for each emotion
        # Format: (valence_min, valence_max, arousal_min, arousal_max, dominance_min, dominance_max)
        self.emotion_thresholds = {
            'happy': (3.5, 5.0, 3.0, 5.0, 3.0, 5.0),  # High valence, high arousal, high dominance
            'angry': (1.0, 2.5, 3.5, 5.0, 3.5, 5.0),  # Low valence, high arousal, high dominance
            'sad': (1.0, 2.5, 1.0, 2.5, 1.0, 2.5),    # Low valence, low arousal, low dominance
            'neutral': (2.5, 3.5, 2.5, 3.5, 2.5, 3.5)  # Medium valence, medium arousal, medium dominance
        }
    
    def predict(self, vad_values):
        """
        Predict emotion from VAD values.
        
        Args:
            vad_values: Array of VAD values (valence, arousal, dominance)
            
        Returns:
            Predicted emotion labels
        """
        predictions = []
        
        for vad in vad_values:
            valence, arousal, dominance = vad
            
            # Calculate distance to each emotion prototype
            distances = {}
            for emotion, thresholds in self.emotion_thresholds.items():
                v_min, v_max, a_min, a_max, d_min, d_max = thresholds
                
                # Check if VAD values are within thresholds
                v_match = v_min <= valence <= v_max
                a_match = a_min <= arousal <= a_max
                d_match = d_min <= dominance <= d_max
                
                # Calculate distance to center of threshold range
                v_center = (v_min + v_max) / 2
                a_center = (a_min + a_max) / 2
                d_center = (d_min + d_max) / 2
                
                distance = np.sqrt((valence - v_center)**2 + (arousal - a_center)**2 + (dominance - d_center)**2)
                distances[emotion] = distance
            
            # Predict the emotion with the smallest distance
            predicted_emotion = min(distances, key=distances.get)
            predictions.append(predicted_emotion)
        
        return np.array(predictions)

class EmotionClassifierTrainer:
    """
    Trainer for the emotion classifier.
    """
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the trainer.
        
        Args:
            model: EmotionClassifier instance
            device: Device to use for training
        """
        self.model = model
        self.device = device
        self.model.to(device)
    
    def train(self, train_loader, test_loader, num_epochs=20, learning_rate=1e-3):
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training data
            test_loader: DataLoader for testing data
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            
        Returns:
            Trained model
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                vad_values = batch['vad_values'].to(self.device)
                targets = batch['emotion'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(vad_values)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(train_loader):.4f}")
            
            # Evaluation
            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                self.evaluate(test_loader)
        
        return self.model
    
    def evaluate(self, test_loader):
        """
        Evaluate the model.
        
        Args:
            test_loader: DataLoader for testing data
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for batch in test_loader:
                vad_values = batch['vad_values'].to(self.device)
                targets = batch['emotion']
                
                outputs = self.model(vad_values)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Compute metrics
        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted')
        
        print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(all_targets, all_preds))
        
        return {
            'accuracy': accuracy,
            'f1': f1
        }
    
    def predict(self, vad_values):
        """
        Predict emotions from VAD values.
        
        Args:
            vad_values: Array of VAD values (valence, arousal, dominance)
            
        Returns:
            Predicted emotion labels
        """
        self.model.eval()
        
        # Convert to tensor
        vad_tensor = torch.tensor(vad_values, dtype=torch.float).to(self.device)
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(vad_tensor)
            _, predicted = torch.max(outputs, 1)
        
        return predicted.cpu().numpy()
