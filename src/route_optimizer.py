import numpy as np
import joblib
from traffic_model import TrafficModel

class RouteOptimizer:
    def __init__(self):
        # Load the trained traffic model
        self.model = TrafficModel()
        self.model.load_model('models/traffic_model.pkl')

    def optimize_route(self, locations, current_hour, day_of_week):
        """
        Optimize route based on traffic predictions.

        Args:
            locations: List of tuples with coordinates [(x1, y1), (x2, y2), ...].
            current_hour: Current hour of the day (0-23).
            day_of_week: Current day of the week (0-6).

        Returns:
            List of optimized coordinates [(x1, y1), (x2, y2), ...].
        """
        if not locations:
            return []

        # Prepare features for prediction
        features = []
        for loc in locations:
            x, y = loc
            hour_sin = np.sin(2 * np.pi * current_hour / 24)
            hour_cos = np.cos(2 * np.pi * current_hour / 24)
            day_sin = np.sin(2 * np.pi * day_of_week / 7)
            day_cos = np.cos(2 * np.pi * day_of_week / 7)
            features.append([x, y, hour_sin, hour_cos, day_sin, day_cos])

        features = np.array(features)

        # Predict traffic speeds using the trained model
        traffic_speeds = self.model.predict(features)

        # Sort locations by predicted traffic speed (lower speed = higher priority)
        sorted_indices = np.argsort(traffic_speeds)
        optimized_route = [locations[i] for i in sorted_indices]

        return optimized_route
