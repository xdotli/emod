"""
Audio-based VAD prediction model.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

class AudioVADDataset(Dataset):
    """
    Dataset for audio-based VAD prediction.
    """
    def __init__(self, audio_paths, vad_values, sample_rate=16000, max_length=5):
        """
        Initialize the dataset.
        
        Args:
            audio_paths: List of paths to audio files
            vad_values: Array of VAD values (valence, arousal, dominance)
            sample_rate: Target sample rate
            max_length: Maximum audio length in seconds
        """
        self.audio_paths = audio_paths
        self.vad_values = vad_values
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.max_samples = sample_rate * max_length
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        vad = self.vad_values[idx]
        
        # Load audio file
        if os.path.exists(audio_path):
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if necessary
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                waveform = resampler(waveform)
            
            # Pad or truncate to max_samples
            if waveform.shape[1] < self.max_samples:
                # Pad with zeros
                padding = torch.zeros(1, self.max_samples - waveform.shape[1])
                waveform = torch.cat([waveform, padding], dim=1)
            elif waveform.shape[1] > self.max_samples:
                # Truncate
                waveform = waveform[:, :self.max_samples]
        else:
            # If file doesn't exist, return zeros
            waveform = torch.zeros(1, self.max_samples)
        
        # Extract features (mel spectrogram)
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )(waveform)
        
        # Convert to decibels
        mel_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
        
        return {
            'features': mel_spectrogram,
            'vad_values': torch.tensor(vad, dtype=torch.float)
        }

class AudioVADModel(nn.Module):
    """
    Audio-based VAD prediction model.
    """
    def __init__(self, dropout=0.3):
        """
        Initialize the model.
        
        Args:
            dropout: Dropout rate
        """
        super(AudioVADModel, self).__init__()
        
        # CNN layers for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Calculate the size of the flattened features
        # For a 64x157 mel spectrogram (5 seconds of audio)
        # After 4 max pooling layers: 64/16 x 157/16 = 4 x 9 = 36 features
        self.fc_input_size = 256 * 4 * 9
        
        # Shared fully connected layers
        self.shared_layer = nn.Sequential(
            nn.Linear(self.fc_input_size, 512),
            nn.LayerNorm(512),
            nn.Dropout(dropout),
            nn.ReLU(),
        )
        
        # Separate branches for VAD
        self.valence_branch = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.arousal_branch = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.dominance_branch = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input mel spectrogram
            
        Returns:
            VAD predictions (valence, arousal, dominance)
        """
        # CNN feature extraction
        x = self.conv_layers(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Shared layer
        shared = self.shared_layer(x)
        
        # VAD branches
        valence = self.valence_branch(shared)
        arousal = self.arousal_branch(shared)
        dominance = self.dominance_branch(shared)
        
        return torch.cat([valence, arousal, dominance], dim=1)

class AudioVADTrainer:
    """
    Trainer for the audio-based VAD prediction model.
    """
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the trainer.
        
        Args:
            model: AudioVADModel instance
            device: Device to use for training
        """
        self.model = model
        self.device = device
        self.model.to(device)
    
    def train(self, train_loader, test_loader, num_epochs=20, learning_rate=1e-4):
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
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                features = batch['features'].to(self.device)
                targets = batch['vad_values'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(features)
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
                features = batch['features'].to(self.device)
                targets = batch['vad_values'].to(self.device)
                
                outputs = self.model(features)
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # Stack all predictions and targets
        preds = np.vstack(all_preds)
        targets = np.vstack(all_targets)
        
        # Compute metrics per VAD dimension
        mse = mean_squared_error(targets, preds, multioutput='raw_values')
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, preds, multioutput='raw_values')
        r2 = r2_score(targets, preds, multioutput='raw_values')
        
        # Print metrics
        vad_labels = ['Valence', 'Arousal', 'Dominance']
        for i in range(3):
            print(f"{vad_labels[i]} - MSE: {mse[i]:.4f}, RMSE: {rmse[i]:.4f}, MAE: {mae[i]:.4f}, R²: {r2[i]:.4f}")
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def predict(self, audio_paths, sample_rate=16000, max_length=5):
        """
        Predict VAD values for new audio files.
        
        Args:
            audio_paths: List of paths to audio files
            sample_rate: Target sample rate
            max_length: Maximum audio length in seconds
            
        Returns:
            VAD predictions (valence, arousal, dominance)
        """
        self.model.eval()
        max_samples = sample_rate * max_length
        all_features = []
        
        for audio_path in audio_paths:
            if os.path.exists(audio_path):
                waveform, sr = torchaudio.load(audio_path)
                
                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Resample if necessary
                if sr != sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, sample_rate)
                    waveform = resampler(waveform)
                
                # Pad or truncate to max_samples
                if waveform.shape[1] < max_samples:
                    # Pad with zeros
                    padding = torch.zeros(1, max_samples - waveform.shape[1])
                    waveform = torch.cat([waveform, padding], dim=1)
                elif waveform.shape[1] > max_samples:
                    # Truncate
                    waveform = waveform[:, :max_samples]
            else:
                # If file doesn't exist, return zeros
                waveform = torch.zeros(1, max_samples)
            
            # Extract features (mel spectrogram)
            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=1024,
                hop_length=512,
                n_mels=64
            )(waveform)
            
            # Convert to decibels
            mel_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
            all_features.append(mel_spectrogram)
        
        # Stack features
        features = torch.stack(all_features).to(self.device)
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(features)
        
        return outputs.cpu().numpy()
