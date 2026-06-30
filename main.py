import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
CORS(app)

class IncidentPredictor:
    def __init__(self):
        self.model = LinearRegression()
    
    def predict(self, historical_data, periods=3):
        result = {
            "status": "insufficient_data",
            "message": "Need at least 3 data points",
            "predicted_values": [],
            "confidence": 0,
            "trend_direction": "unknown",
            "avg_predicted": 0
        }
        
        if len(historical_data) < 3:
            return result
        
        X = np.array(range(len(historical_data))).reshape(-1, 1)
        y = np.array(historical_data, dtype=float)
        
        self.model.fit(X, y)
        
        r_squared = self.model.score(X, y)
        confidence = round(r_squared * 100, 1)
        
        future_X = np.array(
            range(len(historical_data), len(historical_data) + periods)
        ).reshape(-1, 1)
        predictions = self.model.predict(future_X)
        
        predicted_values = [max(0, round(float(p))) for p in predictions]
        
        recent_trend = historical_data[-1] - historical_data[-2]
        if recent_trend > 0:
            direction = "increasing"
        elif recent_trend < 0:
            direction = "decreasing"
        else:
            direction = "stable"
        
        result = {
            "status": "success",
            "predicted_values": predicted_values,
            "confidence": confidence,
            "trend_direction": direction,
            "avg_predicted": round(np.mean(predicted_values)),
            "current_avg": round(np.mean(historical_data[-3:])) if len(historical_data) >= 3 else historical_data[-1],
            "model_used": "Linear Regression"
        }
        
        return result

predictor = IncidentPredictor()

def generate_ai_message(historical_data, prediction_result, top_incident_type):
    if prediction_result["status"] != "success":
        return f"Collecting more data... Need at least 3 months of records. Currently have {len(historical_data)} month(s)."
    
    current_avg = prediction_result["current_avg"]
    predicted_avg = prediction_result["avg_predicted"]
    confidence = prediction_result["confidence"]
    direction = prediction_result["trend_direction"]
    predicted_values = prediction_result["predicted_values"]
    
    lines = []
    lines.append(f"<strong>Trend Analysis:</strong>")
    lines.append(f"Current 3-month avg: <strong>{current_avg}</strong> incidents")
    lines.append(f"Predicted 3-month avg: <strong>{predicted_avg}</strong> incidents")
    lines.append(f"Model confidence: <strong>{confidence}%</strong>")
    lines.append("")
    
    if direction == "increasing":
        change = predicted_avg - current_avg
        pct = ((change / current_avg) * 100) if current_avg > 0 else 0
        lines.append(f"WARNING: Upward trend detected!")
        lines.append(f"Expected increase: ~{abs(round(pct))}%")
        lines.append(f"Primary concern: <strong>{top_incident_type}</strong>")
        lines.append(f"Recommend increasing patrols and community awareness")
    elif direction == "decreasing":
        change = current_avg - predicted_avg
        pct = ((change / current_avg) * 100) if current_avg > 0 else 0
        lines.append(f"POSITIVE: Downward trend detected!")
        lines.append(f"Expected decrease: ~{abs(round(pct))}%")
        lines.append(f"Current measures are effective. Continue monitoring.")
    else:
        lines.append(f"STABLE: No significant change expected")
        lines.append(f"Maintain current incident management strategies")
    
    lines.append("")
    lines.append(f"Forecast for next 3 periods: {predicted_values}")
    lines.append(f"Model: {prediction_result['model_used']}")
    
    return "<br>".join(lines)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
        
        historical_data = data.get("historical_data", [])
        top_incident_type = data.get("top_incident_type", "Incidents")
        periods = data.get("periods", 3)
        
        if not isinstance(historical_data, list) or len(historical_data) == 0:
            return jsonify({"status": "error", "message": "historical_data must be a non-empty array"}), 400
        
        historical_data = [float(x) for x in historical_data]
        
        result = predictor.predict(historical_data, periods)
        ai_message = generate_ai_message(historical_data, result, top_incident_type)
        
        return jsonify({
            "status": "success",
            "prediction": result,
            "ai_message": ai_message
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "service": "Barangay Incident Prediction API",
        "version": "1.0.0"
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
