"""
Run evaluation on the trained model and output numerical results.
"""
import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from models import TextVADModel, RuleBasedEmotionClassifier
from data import IEMOCAPDataProcessor
from utils.vad_emotion_mapping import batch_vad_to_emotion

def main():
    # Load the IEMOCAP dataset
    data_processor = IEMOCAPDataProcessor('../IEMOCAP_Final.csv')
    data_processor.load_data()
    data_processor.extract_vad_values()
    data_processor.process_emotion_labels()
    df_model = data_processor.prepare_data_for_vad_prediction()
    
    # Get a sample of the data for quick evaluation
    sample_size = 100
    df_sample = df_model.sample(n=sample_size, random_state=42)
    
    # Get true VAD values and emotions
    true_vad = df_sample[['valence', 'arousal', 'dominance']].values
    true_emotions = df_sample['Mapped_Emotion'].values
    
    # Load the trained model
    model_dir = 'output'
    text_vad_model = TextVADModel(model_name='roberta-base')
    text_vad_model.load_state_dict(torch.load(os.path.join(model_dir, 'text_vad_model.pt')))
    text_vad_model.eval()
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_vad_model = text_vad_model.to(device)
    
    # Predict VAD values
    texts = df_sample['Transcript'].values
    pred_vad = []
    
    for text in texts:
        # Tokenize text
        encoding = tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move inputs to device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Predict VAD values
        with torch.no_grad():
            vad_pred = text_vad_model(input_ids, attention_mask)
        
        # Convert to numpy
        vad_values = vad_pred.cpu().numpy()[0]
        pred_vad.append(vad_values)
    
    pred_vad = np.array(pred_vad)
    
    # Calculate VAD prediction metrics
    vad_mse = np.mean((true_vad - pred_vad) ** 2, axis=0)
    vad_rmse = np.sqrt(vad_mse)
    vad_mae = np.mean(np.abs(true_vad - pred_vad), axis=0)
    
    # Predict emotions using rule-based classifier
    rule_based_classifier = RuleBasedEmotionClassifier()
    pred_emotions = rule_based_classifier.predict(pred_vad)
    
    # Calculate emotion classification metrics
    accuracy = accuracy_score(true_emotions, pred_emotions)
    f1 = f1_score(true_emotions, pred_emotions, average='weighted')
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    print("\nVAD Prediction Metrics:")
    print(f"MSE: Valence={vad_mse[0]:.4f}, Arousal={vad_mse[1]:.4f}, Dominance={vad_mse[2]:.4f}, Average={np.mean(vad_mse):.4f}")
    print(f"RMSE: Valence={vad_rmse[0]:.4f}, Arousal={vad_rmse[1]:.4f}, Dominance={vad_rmse[2]:.4f}, Average={np.mean(vad_rmse):.4f}")
    print(f"MAE: Valence={vad_mae[0]:.4f}, Arousal={vad_mae[1]:.4f}, Dominance={vad_mae[2]:.4f}, Average={np.mean(vad_mae):.4f}")
    
    print("\nEmotion Classification Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(true_emotions, pred_emotions))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_emotions, pred_emotions)
    labels = np.unique(np.concatenate([true_emotions, pred_emotions]))
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)
    
    print("\nSample Predictions:")
    print("-"*80)
    for i in range(min(10, sample_size)):
        print(f"Text: {texts[i]}")
        print(f"True VAD: Valence={true_vad[i][0]:.2f}, Arousal={true_vad[i][1]:.2f}, Dominance={true_vad[i][2]:.2f}")
        print(f"Pred VAD: Valence={pred_vad[i][0]:.2f}, Arousal={pred_vad[i][1]:.2f}, Dominance={pred_vad[i][2]:.2f}")
        print(f"True Emotion: {true_emotions[i]}")
        print(f"Pred Emotion: {pred_emotions[i]}")
        print("-"*80)

if __name__ == '__main__':
    main()
