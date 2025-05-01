"""
Main script for the emotion recognition system.
"""
import os
import argparse
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from data import IEMOCAPDataProcessor
from models import (
    TextVADModel, TextVADDataset, TextVADTrainer,
    AudioVADModel, AudioVADDataset, AudioVADTrainer,
    MultimodalVADModel, MultimodalVADDataset, MultimodalVADTrainer,
    EmotionClassifier, RuleBasedEmotionClassifier, VADEmotionDataset, EmotionClassifierTrainer
)
from utils import plot_vad_distribution, plot_confusion_matrix, plot_vad_predictions, plot_emotion_distribution

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Emotion Recognition System')
    parser.add_argument('--data_path', type=str, default='IEMOCAP_Final.csv',
                        help='Path to the IEMOCAP_Final.csv file')
    parser.add_argument('--mode', type=str, default='text',
                        choices=['text', 'audio', 'multimodal'],
                        help='Mode of operation (text, audio, multimodal)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory for models and results')

    return parser.parse_args()

def set_seed(seed):
    """
    Set random seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_text_vad_model(data_processor, args):
    """
    Train the text-based VAD prediction model.

    Args:
        data_processor: IEMOCAPDataProcessor instance
        args: Command line arguments

    Returns:
        Trained model and tokenizer
    """
    print("Training text-based VAD prediction model...")

    # Prepare data
    df_model = data_processor.prepare_data_for_vad_prediction()
    texts = df_model['Transcript'].values
    vad = df_model[['valence', 'arousal', 'dominance']].values

    # Split data
    X_train, X_test, y_train, y_test, _, _ = data_processor.split_data(df_model, test_size=0.2, random_state=args.seed)

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    model = TextVADModel(model_name='roberta-base', dropout=0.1)

    # Create datasets and dataloaders
    train_dataset = TextVADDataset(X_train, y_train, tokenizer)
    test_dataset = TextVADDataset(X_test, y_test, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Train model
    trainer = TextVADTrainer(model, tokenizer)
    model = trainer.train(train_loader, test_loader, num_epochs=args.epochs, learning_rate=args.learning_rate)

    # Evaluate model
    metrics = trainer.evaluate(test_loader)

    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'text_vad_model.pt'))

    return model, tokenizer, metrics

def train_audio_vad_model(data_processor, args):
    """
    Train the audio-based VAD prediction model.

    Args:
        data_processor: IEMOCAPDataProcessor instance
        args: Command line arguments

    Returns:
        Trained model
    """
    print("Training audio-based VAD prediction model...")

    # Prepare data
    df_model = data_processor.prepare_data_for_vad_prediction()
    audio_paths = df_model['Audio_Uttrance_Path'].values
    vad = df_model[['valence', 'arousal', 'dominance']].values

    # Split data
    X_train, X_test, y_train, y_test, _, _ = data_processor.split_data(df_model, test_size=0.2, random_state=args.seed)

    # Initialize model
    model = AudioVADModel(dropout=0.3)

    # Create datasets and dataloaders
    train_dataset = AudioVADDataset(X_train, y_train)
    test_dataset = AudioVADDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Train model
    trainer = AudioVADTrainer(model)
    model = trainer.train(train_loader, test_loader, num_epochs=args.epochs, learning_rate=1e-4)

    # Evaluate model
    metrics = trainer.evaluate(test_loader)

    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'audio_vad_model.pt'))

    return model, metrics

def train_multimodal_vad_model(text_model, audio_model, data_processor, args):
    """
    Train the multimodal VAD prediction model.

    Args:
        text_model: Trained text VAD model
        audio_model: Trained audio VAD model
        data_processor: IEMOCAPDataProcessor instance
        args: Command line arguments

    Returns:
        Trained model
    """
    print("Training multimodal VAD prediction model...")

    # Prepare data
    df_model = data_processor.prepare_data_for_vad_prediction()

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    # Tokenize texts
    texts = df_model['Transcript'].values
    encodings = tokenizer(
        texts,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Get audio paths
    audio_paths = df_model['Audio_Uttrance_Path'].values

    # Get VAD values
    vad = df_model[['valence', 'arousal', 'dominance']].values

    # Split data
    X_train_ids, X_test_ids, X_train_mask, X_test_mask, X_train_audio, X_test_audio, y_train, y_test = train_test_split(
        encodings['input_ids'], encodings['attention_mask'], audio_paths, vad,
        test_size=0.2, random_state=args.seed
    )

    # Initialize model
    model = MultimodalVADModel(text_model, audio_model, fusion_dropout=0.3)

    # Create datasets and dataloaders
    train_dataset = MultimodalVADDataset(
        {'input_ids': X_train_ids, 'attention_mask': X_train_mask},
        X_train_audio,
        y_train
    )
    test_dataset = MultimodalVADDataset(
        {'input_ids': X_test_ids, 'attention_mask': X_test_mask},
        X_test_audio,
        y_test
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Train model
    trainer = MultimodalVADTrainer(model)
    model = trainer.train(train_loader, test_loader, num_epochs=10, learning_rate=1e-4)

    # Evaluate model
    metrics = trainer.evaluate(test_loader)

    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'multimodal_vad_model.pt'))

    return model, metrics

def train_emotion_classifier(vad_model, data_processor, args, mode='text'):
    """
    Train the emotion classifier based on VAD predictions.

    Args:
        vad_model: Trained VAD prediction model
        data_processor: IEMOCAPDataProcessor instance
        args: Command line arguments
        mode: Mode of operation (text, audio, multimodal)

    Returns:
        Trained model
    """
    print(f"Training {mode}-based emotion classifier...")

    # Prepare data
    df_model = data_processor.prepare_data_for_vad_prediction()

    # Get VAD values and emotion labels
    vad = df_model[['valence', 'arousal', 'dominance']].values
    emotions = df_model['Mapped_Emotion'].values

    # Encode emotion labels
    label_encoder = LabelEncoder()
    encoded_emotions = label_encoder.fit_transform(emotions)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        vad, encoded_emotions, test_size=0.2, random_state=args.seed
    )

    # Initialize model
    model = EmotionClassifier(num_classes=len(label_encoder.classes_), dropout=0.3)

    # Create datasets and dataloaders
    train_dataset = VADEmotionDataset(X_train, y_train)
    test_dataset = VADEmotionDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Train model
    trainer = EmotionClassifierTrainer(model)
    model = trainer.train(train_loader, test_loader, num_epochs=args.epochs, learning_rate=1e-3)

    # Evaluate model
    metrics = trainer.evaluate(test_loader)

    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, f'{mode}_emotion_classifier.pt'))

    return model, label_encoder, metrics

def main():
    """
    Main function.
    """
    # Parse arguments
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Initialize data processor
    data_processor = IEMOCAPDataProcessor(args.data_path)

    # Load and process data
    data_processor.load_data()
    data_processor.extract_vad_values()
    data_processor.process_emotion_labels()

    # Train models based on mode
    if args.mode == 'text':
        # Train text-based VAD prediction model
        text_vad_model, tokenizer, text_vad_metrics = train_text_vad_model(data_processor, args)

        # Train emotion classifier
        emotion_classifier, label_encoder, emotion_metrics = train_emotion_classifier(
            text_vad_model, data_processor, args, mode='text'
        )

    elif args.mode == 'audio':
        # Train audio-based VAD prediction model
        audio_vad_model, audio_vad_metrics = train_audio_vad_model(data_processor, args)

        # Train emotion classifier
        emotion_classifier, label_encoder, emotion_metrics = train_emotion_classifier(
            audio_vad_model, data_processor, args, mode='audio'
        )

    elif args.mode == 'multimodal':
        # Train text-based VAD prediction model
        text_vad_model, tokenizer, text_vad_metrics = train_text_vad_model(data_processor, args)

        # Train audio-based VAD prediction model
        audio_vad_model, audio_vad_metrics = train_audio_vad_model(data_processor, args)

        # Train multimodal VAD prediction model
        multimodal_vad_model, multimodal_vad_metrics = train_multimodal_vad_model(
            text_vad_model, audio_vad_model, data_processor, args
        )

        # Train emotion classifier
        emotion_classifier, label_encoder, emotion_metrics = train_emotion_classifier(
            multimodal_vad_model, data_processor, args, mode='multimodal'
        )

    print(f"Training complete. Models saved to {args.output_dir}")

if __name__ == '__main__':
    main()
