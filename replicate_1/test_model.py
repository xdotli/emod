"""
Simple test script for the emotion recognition model.
"""
import os
import torch
from transformers import AutoTokenizer
from models import TextVADModel, RuleBasedEmotionClassifier

def main():
    # Load the model
    model_dir = 'output'
    text_vad_model = TextVADModel(model_name='roberta-base')
    text_vad_model.load_state_dict(torch.load(os.path.join(model_dir, 'text_vad_model.pt')))
    text_vad_model.eval()
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    
    # Test text
    text = "I am feeling very happy today!"
    
    # Tokenize text
    encoding = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move model and inputs to device
    text_vad_model = text_vad_model.to(device)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Predict VAD values
    with torch.no_grad():
        vad_pred = text_vad_model(input_ids, attention_mask)
    
    # Convert to numpy
    vad_values = vad_pred.cpu().numpy()[0]
    
    # Use rule-based classification
    rule_based_classifier = RuleBasedEmotionClassifier()
    emotion = rule_based_classifier.predict(vad_values.reshape(1, -1))[0]
    
    # Print results
    print(f"Text: {text}")
    print(f"VAD values: Valence={vad_values[0]:.2f}, Arousal={vad_values[1]:.2f}, Dominance={vad_values[2]:.2f}")
    print(f"Predicted emotion: {emotion}")

if __name__ == '__main__':
    main()
