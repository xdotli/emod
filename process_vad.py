import numpy as np
import pandas as pd

def vad_to_emotion(valence, arousal, dominance):
    """
    Convert VAD (Valence-Arousal-Dominance) values to basic emotions.
    
    Parameters:
    - valence: Float between -1 and 1, representing negative to positive feelings
    - arousal: Float between -1 and 1, representing calm to excited
    - dominance: Float between -1 and 1, representing submissive to dominant
    
    Returns:
    - emotion_label: String representing the emotion category
    """
    # Define emotion categories based on VAD space quadrants
    # This is a simplified mapping based on common emotion models
    
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

def process_vad_from_dataset(dataset_path):
    """
    Process VAD values from a dataset and convert to emotion labels.
    
    Parameters:
    - dataset_path: Path to CSV file containing VAD values
    
    Returns:
    - DataFrame with original data plus emotion labels
    """
    # Load the dataset
    try:
        data = pd.read_csv(dataset_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    # Check if VAD columns exist
    required_columns = ['valence', 'arousal', 'dominance']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        # Try alternate column names
        column_alternatives = {
            'valence': ['val', 'v'],
            'arousal': ['aro', 'a'],
            'dominance': ['dom', 'd']
        }
        
        for missing_col in missing_columns:
            for alt in column_alternatives[missing_col]:
                if alt in data.columns:
                    data[missing_col] = data[alt]
                    missing_columns.remove(missing_col)
                    break
    
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return None
    
    # Normalize VAD values if they're not in the -1 to 1 range
    for col in required_columns:
        col_min, col_max = data[col].min(), data[col].max()
        if col_min < -1 or col_max > 1:
            # Assuming the values are in a different range and need normalization
            data[col] = 2 * ((data[col] - col_min) / (col_max - col_min)) - 1
    
    # Apply the VAD to emotion mapping
    data['emotion'] = data.apply(
        lambda row: vad_to_emotion(row['valence'], row['arousal'], row['dominance']), 
        axis=1
    )
    
    return data

def visualize_vad_emotions(data):
    """
    Plot the VAD values colored by emotion category.
    
    Parameters:
    - data: DataFrame with valence, arousal, dominance and emotion columns
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Define colors for each emotion
        emotion_colors = {
            'happy': 'yellow',
            'excited': 'orange',
            'content': 'green',
            'relaxed': 'lightgreen',
            'angry': 'red',
            'fearful': 'purple',
            'disgusted': 'brown',
            'sad': 'blue'
        }
        
        # Plot each point
        for emotion in data['emotion'].unique():
            subset = data[data['emotion'] == emotion]
            ax.scatter(
                subset['valence'], 
                subset['arousal'], 
                subset['dominance'],
                c=emotion_colors.get(emotion, 'gray'),
                label=emotion,
                alpha=0.7
            )
        
        ax.set_xlabel('Valence')
        ax.set_ylabel('Arousal')
        ax.set_zlabel('Dominance')
        ax.set_title('Emotion Categories in VAD Space')
        ax.legend()
        
        plt.show()
    except ImportError:
        print("Matplotlib is required for visualization")

# Example usage
if __name__ == "__main__":
    # Example path - replace with actual dataset path from the project
    example_dataset_path = "Datasets/IEMOCAP_full_release/Session1/dialog/wav/features.csv"
    
    # Process the dataset
    processed_data = process_vad_from_dataset(example_dataset_path)
    
    if processed_data is not None:
        # Print summary of emotions found
        print("Emotion distribution:")
        print(processed_data['emotion'].value_counts())
        
        # Visualize the data
        visualize_vad_emotions(processed_data)
