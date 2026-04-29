import os
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'business_risk_model.pkl')
METRICS_PATH = os.path.join(BASE_DIR, 'models', 'metrics.json')

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

def get_risk_category(score):
    if score < 33:
        return "Low Risk"
    elif score < 66:
        return "Medium Risk"
    else:
        return "High Risk"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    global model
    if model is None:
        model = load_model()
        if model is None:
            return jsonify({'error': 'Model not trained. Call /train first.'}), 500
            
    try:
        data = request.json
        features = pd.DataFrame([{
            'revenue': float(data.get('revenue', 0)),
            'expenses': float(data.get('expenses', 0)),
            'profit_margin': float(data.get('profit_margin', 0)),
            'debt_ratio': float(data.get('debt_ratio', 0)),
            'cash_flow': float(data.get('cash_flow', 0)),
            'market_growth': float(data.get('market_growth', 0)),
            'years_in_business': float(data.get('years_in_business', 0)),
            'employee_count': float(data.get('employee_count', 0))
        }])
        
        prediction = model.predict(features)[0]
        score = max(0, min(100, float(prediction))) # keep within 0-100
        category = get_risk_category(score)
        
        return jsonify({
            'score': round(score, 2),
            'category': category
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/metrics', methods=['GET'])
def get_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
        return jsonify(metrics)
    return jsonify({'error': 'Metrics not found. Train the model first.'}), 404

@app.route('/train', methods=['POST'])
def retrain_model():
    try:
        from train_model import train_models
        train_models()
        global model
        model = load_model()
        return jsonify({'status': 'success', 'message': 'Model retrained successfully.'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
