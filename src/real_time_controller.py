import numpy as np
from flask import Flask, request, jsonify
from route_optimizer import RouteOptimizer
from datetime import datetime

app = Flask(__name__)

# Initialize the RouteOptimizer
optimizer = RouteOptimizer()

@app.route('/optimize_route', methods=['POST'])
def optimize_route():
    """
    API endpoint to optimize bus routes in real-time.
    
    Request JSON Format:
    {
        "locations": [[x1, y1], [x2, y2], ...],
        "current_time": "HH:MM",
        "day_of_week": 0-6 (0=Sunday, 6=Saturday)
    }

    Response JSON Format:
    {
        "optimized_route": [[x1, y1], [x2, y2], ...],
        "status": "success"
    }
    """
    try:
        # Validate input data
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Extract and validate required fields
        locations = data.get('locations')
        current_time_str = data.get('current_time')
        day_of_week = data.get('day_of_week')

        if not all([locations, current_time_str, day_of_week is not None]):
            return jsonify({
                "error": "Missing required fields. Please provide locations, current_time, and day_of_week"
            }), 400

        # Validate locations format
        if not isinstance(locations, list) or not locations:
            return jsonify({"error": "Invalid locations format. Must be a non-empty list of coordinates"}), 400

        # Validate day_of_week
        try:
            day_of_week = int(day_of_week)
            if not 0 <= day_of_week <= 6:
                raise ValueError
        except ValueError:
            return jsonify({"error": "day_of_week must be an integer between 0 and 6"}), 400

        # Validate and parse time
        try:
            hour, minute = map(int, current_time_str.split(':'))
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                raise ValueError
        except (ValueError, IndexError):
            return jsonify({"error": "Invalid time format. Use HH:MM (24-hour format)"}), 400

        # Convert locations to the correct format if needed
        try:
            locations = [(float(x), float(y)) for x, y in locations]
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid coordinate format. Each location should be [x, y]"}), 400

        # Get current timestamp
        current_date = datetime.now()
        
        # Optimize the route with all required parameters
        optimized_route = optimizer.optimize_route(
            locations=locations,
            current_time_str=current_time_str,
            day_of_week=day_of_week
        )

        return jsonify({
            "optimized_route": optimized_route,
            "status": "success"
        }), 200

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)