"""
Data processing module for the IEMOCAP dataset.
"""
import pandas as pd
import numpy as np
import ast
from collections import Counter
from sklearn.model_selection import train_test_split

class IEMOCAPDataProcessor:
    """
    Processes the IEMOCAP dataset for emotion recognition tasks.
    """
    def __init__(self, csv_path):
        """
        Initialize the data processor.
        
        Args:
            csv_path: Path to the IEMOCAP_Final.csv file
        """
        self.csv_path = csv_path
        self.df = None
        self.df_final = None
        self.emotion_mapping = {
            'neutral': 'neutral',
            'frustration': 'angry',
            'anger': 'angry',
            'surprise': None,
            'disgust': None,
            'other': None,
            'sadness': 'sad',
            'fear': None,
            'happiness': 'happy',
            'excited': 'happy'
        }
    
    def load_data(self):
        """
        Load the IEMOCAP dataset.
        """
        self.df = pd.read_csv(self.csv_path)
        return self.df
    
    def extract_vad_values(self):
        """
        Extract VAD (Valence, Arousal, Dominance) values from the dimension column.
        """
        if self.df is None:
            self.load_data()
        
        # Function to extract VAD values from the dimension column
        def extract_vad(dimension_str):
            try:
                dimension_dict = ast.literal_eval(dimension_str)[0]
                return {
                    'valence': dimension_dict['valence'],
                    'arousal': dimension_dict['arousal'],
                    'dominance': dimension_dict['dominance']
                }
            except (ValueError, SyntaxError, KeyError, IndexError):
                return {'valence': np.nan, 'arousal': np.nan, 'dominance': np.nan}
        
        # Apply the function to extract VAD values
        vad_values = self.df['dimension'].apply(extract_vad)
        
        # Create new columns for VAD values
        self.df['valence'] = vad_values.apply(lambda x: x['valence'])
        self.df['arousal'] = vad_values.apply(lambda x: x['arousal'])
        self.df['dominance'] = vad_values.apply(lambda x: x['dominance'])
        
        # Drop rows with NaN VAD values
        self.df = self.df.dropna(subset=['valence', 'arousal', 'dominance'])
        
        return self.df
    
    def process_emotion_labels(self):
        """
        Process emotion labels from the category column.
        """
        if self.df is None:
            self.load_data()
        
        # Create a copy of the dataframe with relevant columns
        df_processed = self.df[['Speaker_id', 'Transcript', 'dimension', 'category']].copy()
        
        # Convert category to lowercase
        df_processed['category'] = df_processed['category'].astype(str).str.lower()
        
        # Drop duplicate transcripts
        df_processed = df_processed.drop_duplicates(subset='Transcript')
        
        # Function to get the majority emotion label
        def get_majority_label(label_str):
            try:
                labels = ast.literal_eval(label_str)  # convert string to list
                count = Counter(labels)
                most_common = count.most_common(1)[0]
                if most_common[1] >= 2:
                    return most_common[0]
                else:
                    return None
            except (ValueError, SyntaxError):
                return None
        
        # Create a new column for final emotion if majority exists
        df_processed['Emotion'] = df_processed['category'].apply(get_majority_label)
        
        # Drop rows where no majority
        df_filtered = df_processed.dropna(subset=['Emotion'])
        
        # Map the Emotion column
        df_filtered['Mapped_Emotion'] = df_filtered['Emotion'].map(self.emotion_mapping)
        
        # Drop rows where mapping returns None
        self.df_final = df_filtered.dropna(subset=['Mapped_Emotion'])
        
        return self.df_final
    
    def prepare_data_for_vad_prediction(self):
        """
        Prepare data for VAD prediction.
        """
        if self.df_final is None:
            self.process_emotion_labels()
            self.extract_vad_values()
        
        # Merge the processed emotion labels with VAD values
        df_model = self.df_final.merge(
            self.df[['Speaker_id', 'valence', 'arousal', 'dominance']], 
            on='Speaker_id', 
            how='inner'
        )
        
        return df_model
    
    def split_data(self, df, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets.
        
        Args:
            df: DataFrame to split
            test_size: Proportion of the dataset to include in the test split
            random_state: Random state for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        X = df['Transcript'].values
        y_vad = df[['valence', 'arousal', 'dominance']].values
        y_emotion = df['Mapped_Emotion'].values
        
        X_train, X_test, y_vad_train, y_vad_test, y_emotion_train, y_emotion_test = train_test_split(
            X, y_vad, y_emotion, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_vad_train, y_vad_test, y_emotion_train, y_emotion_test
