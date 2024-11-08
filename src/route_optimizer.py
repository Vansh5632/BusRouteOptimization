import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime

class RouteOptimizer:
    def __init__(self):
        """
        Initialize the RouteOptimizer class by loading the trained model and scaler.
        """
        self.model = joblib.load('models/traffic_model.pkl')
        self.scaler = joblib.load('models/scaler.pkl')
        print("[INFO] Traffic model and scaler loaded.")

    def preprocess_input(self, data):
        """
        Preprocess input data for predictions.

        Args:
            data (DataFrame): Input data with columns ['location_x', 'location_y', 'hour', 'day_of_week', 'month'].

        Returns:
            numpy array: Scaled features for prediction.
        """
        # Cyclical encoding for hour and day only
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)

        # Use features exactly as they were during training
        features = ['location_x', 'location_y', 'hour_sin', 'hour_cos', 
                   'day_sin', 'day_cos', 'month']
        scaled_features = self.scaler.transform(data[features])
        return scaled_features

    def predict_traffic_speed(self, data):
        """
        Predict traffic speed for the given data using the trained model.

        Args:
            data (DataFrame): Input data.

        Returns:
            numpy array: Predicted traffic speeds.
        """
        preprocessed_data = self.preprocess_input(data)
        predictions = self.model.predict(preprocessed_data)
        return predictions

    def optimize_route(self, locations, timestamp=None):
        """
        Optimize the route based on predicted traffic speeds.

        Args:
            locations (list): List of (x, y) coordinates representing the route.
            timestamp (datetime, optional): Timestamp for the route. Defaults to current time.

        Returns:
            list: Optimized route based on predicted traffic speeds.
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Create a DataFrame with route data
        route_df = pd.DataFrame(locations, columns=['location_x', 'location_y'])
        route_df['hour'] = timestamp.hour
        route_df['day_of_week'] = timestamp.weekday()
        route_df['month'] = timestamp.month  # Keep month as raw numeric value

        # Predict traffic speeds for each location
        route_df['traffic_speed'] = self.predict_traffic_speed(route_df)

        # Create a graph with weighted edges based on traffic speed
        G = nx.DiGraph()
        for i in range(len(locations) - 1):
            loc1, loc2 = locations[i], locations[i + 1]
            speed = route_df['traffic_speed'][i]
            distance = np.linalg.norm(np.array(loc1) - np.array(loc2))
            time_cost = distance / speed if speed > 0 else float('inf')
            G.add_edge(i, i + 1, weight=time_cost)

        # Find the shortest path based on traffic conditions
        optimized_path = nx.shortest_path(G, source=0, target=len(locations) - 1, weight='weight')
        optimized_route = [locations[i] for i in optimized_path]

        print("[INFO] Route optimization complete.")
        return optimized_route

    def visualize_route(self, original_route, optimized_route):
        """
        Visualize the original and optimized routes.

        Args:
            original_route (list): Original route coordinates.
            optimized_route (list): Optimized route coordinates.
        """
        plt.figure(figsize=(10, 6))
        original_route = np.array(original_route)
        optimized_route = np.array(optimized_route)

        plt.plot(original_route[:, 0], original_route[:, 1], 'ro-', label='Original Route')
        plt.plot(optimized_route[:, 0], optimized_route[:, 1], 'go-', label='Optimized Route')

        plt.title("Route Optimization")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.grid(True)
        plt.show()

def example_usage():
    """
    Example usage of the RouteOptimizer class.
    """
    # Define a list of coordinates representing a sample route
    locations = [(77.5946, 12.9716), (77.5976, 12.9766), (77.6006, 12.9806), (77.6056, 12.9856)]

    # Initialize the Route Optimizer
    optimizer = RouteOptimizer()

    # Use current timestamp for optimization
    timestamp = datetime.now()
    
    # Optimize the route
    optimized_route = optimizer.optimize_route(locations, timestamp)

    # Visualize the original and optimized routes
    optimizer.visualize_route(locations, optimized_route)

    print("Optimized Route:", optimized_route)

if __name__ == "__main__":
    example_usage()