from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Load your AQI ML model
try:
    model = pickle.load(open("aqi.pkl", "rb"))
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model:", e)
    model = None


@app.route("/")
def home():
    return "EcoGuard AI ML Backend is Running"


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()

        # Extract pollutant features
        pm2_5 = float(data.get("pm2_5", 0))
        pm10  = float(data.get("pm10", 0))
        no2   = float(data.get("no2", 0))
        o3    = float(data.get("o3", 0))
        so2   = float(data.get("so2", 0))
        nh3   = float(data.get("nh3", 0))
        no_   = float(data.get("no", 0))
        co    = float(data.get("co", 0))

        # Arrange features
        features = np.array([[pm2_5, pm10, no2, o3, so2, nh3, no_, co]])

        # Model Prediction
        prediction = model.predict(features)[0]

        # Confidence (DecisionTree does not have confidence → use dummy value)
        confidence = 1.0

        # Generate simple analysis
        analysis_msg = f"The model predicts the air quality category as '{prediction}'."

        response = {
            "prediction": str(prediction),
            "confidence": confidence,
            "analysis": analysis_msg
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
