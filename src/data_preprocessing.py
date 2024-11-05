import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

class DataPreprocessor:
    def __init__(self, scaler_path='models/scaler.pkl'):
        self.scaler_path = scaler_path
        self.scaler = self.load_scaler() if os.path.exists(scaler_path) else StandardScaler()

    def load_data(self, file_path):
        """
        Load raw data from a CSV file.
        
        Args:
            file_path (str): Path to the CSV file.
            
        Returns:
            pd.DataFrame: Loaded data as a pandas DataFrame.
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def preprocess_data(self, raw_data):
        """
        Preprocess raw data for training and prediction.
        
        Args:
            raw_data (pd.DataFrame): DataFrame containing raw data.
            
        Returns:
            pd.DataFrame: Preprocessed data.
        """
        df = raw_data.copy()

        # Check if necessary columns exist
        required_columns = ['timestamp', 'location_x', 'location_y', 'passenger_count', 
                            'traffic_speed', 'weather_conditions', 'is_holiday', 'is_weekend']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

        # Extract time features
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month

        # Create cyclical time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Scale numerical features
        numerical_cols = ['location_x', 'location_y', 'traffic_speed']
        df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])

        # Save the scaler for future use
        joblib.dump(self.scaler, self.scaler_path)

        return df

    def load_scaler(self):
        """
        Load an existing scaler from disk.
        """
        try:
            scaler = joblib.load(self.scaler_path)
            print("Scaler loaded successfully.")
            return scaler
        except FileNotFoundError:
            print("Scaler file not found. Initializing a new scaler.")
            return StandardScaler()

# Example usage
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    data = preprocessor.load_data('data/historical_data.csv')
    if data is not None:
        processed_data = preprocessor.preprocess_data(data)
        print("Preprocessing complete. Sample data:")
        print(processed_data.head())
