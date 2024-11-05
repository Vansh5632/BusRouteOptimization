import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import os
import tensorflow as tf

class DemandModel:
    def __init__(self, input_shape):
        self.model = self.build_model(input_shape)
        self.scaler = None
    
    def build_model(self, input_shape):
        """
        Build the LSTM model for demand prediction.
        
        Args:
            input_shape: Tuple representing the shape of the input data.
            
        Returns:
            Compiled LSTM model.
        """
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def train_model(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2):
        """
        Train the LSTM model on the provided training data.
        
        Args:
            X_train: Array of input features for training.
            y_train: Array of target labels for training.
            epochs: Number of epochs to train the model.
            batch_size: Batch size for training.
            validation_split: Fraction of data to use for validation.
        """
        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        try:
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[early_stopping],
                verbose=1
            )
            return history
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise
    
    def save_model(self, model_path='models/demand_model.keras', scaler_path='models/scaler.pkl'):
        """
        Save the trained model and scaler to files.
        
        Args:
            model_path: Path to save the model.
            scaler_path: Path to save the scaler.
        """
        if not os.path.exists('models'):
            os.makedirs('models')
        
        try:
            self.model.save(model_path)
            if self.scaler is not None:
                joblib.dump(self.scaler, scaler_path)
            print(f"Model saved to {model_path}")
            if self.scaler is not None:
                print(f"Scaler saved to {scaler_path}")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            raise

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the data from a CSV file.
    
    Args:
        file_path: Path to the CSV file containing the data.
        
    Returns:
        Processed feature and target arrays, and the fitted scaler.
    """
    try:
        data = pd.read_csv(file_path)
        
        # Ensure the timestamp is in datetime format
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Create features for hour, day, etc.
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        data['month'] = data['timestamp'].dt.month
        
        # Cyclical encoding of time features
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        
        # One-hot encode weather conditions
        encoder = OneHotEncoder(sparse_output=False)  # Updated argument
        weather_encoded = encoder.fit_transform(data[['weather_conditions']])
        weather_encoded_df = pd.DataFrame(weather_encoded, columns=encoder.get_feature_names_out(['weather_conditions']))
        data = pd.concat([data, weather_encoded_df], axis=1)
        data.drop('weather_conditions', axis=1, inplace=True)
        
        # Initialize and fit the scaler
        scaler = StandardScaler()
        numerical_cols = ['location_x', 'location_y', 'traffic_speed']
        data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
        
        # Prepare sequences
        sequence_length = 24  # 24-hour history
        feature_columns = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 
                           'location_x', 'location_y', 'traffic_speed',
                           'is_holiday', 'is_weekend'] + list(weather_encoded_df.columns)
        
        features = data[feature_columns].values
        target = data['passenger_count'].values
        
        X, y = [], []
        for i in range(len(features) - sequence_length):
            X.append(features[i:i + sequence_length])
            y.append(target[i + sequence_length])
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), scaler
    
    except Exception as e:
        print(f"Error in data preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Load and preprocess data
        file_path = 'data/historical_data.csv'
        X, y, scaler = load_and_preprocess_data(file_path)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize and train the model
        input_shape = (X_train.shape[1], X_train.shape[2])
        demand_model = DemandModel(input_shape)
        demand_model.scaler = scaler  # Store the scaler in the model instance
        
        # Train the model
        history = demand_model.train_model(X_train, y_train)
        
        # Save the model and scaler
        demand_model.save_model()
        
        # Print final metrics
        print("\nTraining completed successfully!")
        print(f"Final training loss: {history.history['loss'][-1]:.4f}")
        print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
