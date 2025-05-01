"""
Test the emotion recognition model with example texts.
"""
import os
import torch
from transformers import AutoTokenizer
from models import TextVADModel, RuleBasedEmotionClassifier

def predict_emotion(text, text_vad_model, tokenizer, device):
    """
    Predict emotion from text.
    
    Args:
        text: Input text
        text_vad_model: Trained text VAD model
        tokenizer: Tokenizer for text encoding
        device: Device to use for inference
        
    Returns:
        Predicted VAD values and emotion
    """
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
    
    # Use rule-based classification
    rule_based_classifier = RuleBasedEmotionClassifier()
    emotion = rule_based_classifier.predict(vad_values.reshape(1, -1))[0]
    
    return vad_values, emotion

def main():
    # Load the model
    model_dir = 'output'
    text_vad_model = TextVADModel(model_name='roberta-base')
    text_vad_model.load_state_dict(torch.load(os.path.join(model_dir, 'text_vad_model.pt')))
    text_vad_model.eval()
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_vad_model = text_vad_model.to(device)
    
    # Example texts
    examples = [
        "I am feeling very happy today!",
        "I am so angry right now!",
        "I feel sad and depressed.",
        "I'm just feeling normal, nothing special.",
        "That was the best day of my life!",
        "I hate when people lie to me.",
        "I miss my family so much.",
        "It's just another ordinary day."
    ]
    
    # Predict emotions for each example
    print("Testing emotion recognition model with example texts:\n")
    for text in examples:
        vad_values, emotion = predict_emotion(text, text_vad_model, tokenizer, device)
        print(f"Text: {text}")
        print(f"VAD values: Valence={vad_values[0]:.2f}, Arousal={vad_values[1]:.2f}, Dominance={vad_values[2]:.2f}")
        print(f"Predicted emotion: {emotion}")
        print("-" * 80)

if __name__ == '__main__':
    main()
