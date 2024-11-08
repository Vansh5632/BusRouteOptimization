import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class TrafficModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.required_columns = ['timestamp', 'location_x', 'location_y', 'traffic_speed']

    def build_model(self):
        """
        Build and initialize the RandomForestRegressor model.
        """
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        print("[INFO] Traffic model initialized.")

    def extract_time_features(self, data):
        """
        Extract time-based features from timestamp column.
        
        Args:
            data (DataFrame): Input data with timestamp column
            
        Returns:
            DataFrame: Data with added time-based features
        """
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Extract time components
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        data['month'] = data['timestamp'].dt.month
        
        # Create cyclical features
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        
        return data

    def preprocess_data(self, data):
        """
        Preprocess the data for model training and prediction.

        Args:
            data (DataFrame): Raw data with required columns:
                - timestamp: datetime or string
                - location_x: float
                - location_y: float
                - traffic_speed: float

        Returns:
            X (DataFrame): Preprocessed features
            y (Series): Target variable
        """
        # Verify required columns
        missing_cols = [col for col in self.required_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Handle missing values
        data = data.dropna(subset=self.required_columns)

        # Extract time-based features
        data = self.extract_time_features(data)

        # Select features for model
        features = ['location_x', 'location_y', 'hour_sin', 'hour_cos', 
                   'day_sin', 'day_cos', 'month']
        X = data[features]
        y = data['traffic_speed']

        # Scale features
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        return X_scaled, y

    def train_model(self, X_train, y_train):
        """Train the Random Forest model on training data."""
        if self.model is None:
            self.build_model()

        print("[INFO] Training traffic model...")
        self.model.fit(X_train, y_train)
        print("[INFO] Traffic model training complete.")

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model using test data and print performance metrics.
        """
        if self.model is None:
            raise ValueError("Model is not trained yet!")

        print("[INFO] Evaluating traffic model...")
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)

        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")

        return mse, mae

    def predict(self, X):
        """Predict traffic speed using the trained model."""
        if self.model is None:
            raise ValueError("Model is not trained yet!")
        
        # If X is a DataFrame, extract features and scale
        if isinstance(X, pd.DataFrame):
            X = self.extract_time_features(X)
            features = ['location_x', 'location_y', 'hour_sin', 'hour_cos', 
                       'day_sin', 'day_cos', 'month']
            X = X[features]
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def save_model(self, model_path='models/traffic_model.pkl', scaler_path='models/scaler.pkl'):
        """Save the trained model and scaler to disk."""
        if self.model is None:
            raise ValueError("Model is not trained yet!")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"[INFO] Traffic model saved to {model_path} and scaler to {scaler_path}")

    def load_model(self, model_path='models/traffic_model.pkl', scaler_path='models/scaler.pkl'):
        """Load a trained model and scaler from disk."""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        print(f"[INFO] Traffic model loaded from {model_path} and scaler from {scaler_path}")


def example_usage():
    """Example of how to use the TrafficModel class."""
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
    n_locations = 100
    
    data = pd.DataFrame({
        'timestamp': np.repeat(dates, n_locations),
        'location_x': np.random.uniform(0, 10, len(dates) * n_locations),
        'location_y': np.random.uniform(0, 10, len(dates) * n_locations),
        'traffic_speed': np.random.uniform(20, 60, len(dates) * n_locations)
    })

    # Initialize model and preprocess data
    traffic_model = TrafficModel()
    X, y = traffic_model.preprocess_data(data)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate the model
    traffic_model.train_model(X_train, y_train)
    traffic_model.evaluate_model(X_test, y_test)

    # Save the model
    traffic_model.save_model()


if __name__ == "__main__":
    example_usage()