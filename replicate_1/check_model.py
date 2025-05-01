"""
Check the model structure.
"""
import os
import torch
from models import TextVADModel

def main():
    # Load the model
    model_dir = 'output'
    model_path = os.path.join(model_dir, 'text_vad_model.pt')
    
    # Check if the model file exists
    if os.path.exists(model_path):
        print(f"Model file exists at {model_path}")
        
        # Try to load the model
        try:
            state_dict = torch.load(model_path)
            print("Successfully loaded model state dict")
            print(f"State dict keys: {state_dict.keys()}")
            
            # Initialize the model
            model = TextVADModel(model_name='roberta-base')
            print("Successfully initialized model")
            
            # Load the state dict
            model.load_state_dict(state_dict)
            print("Successfully loaded state dict into model")
            
            # Print model structure
            print("\nModel structure:")
            print(model)
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Model file does not exist at {model_path}")

if __name__ == '__main__':
    main()
