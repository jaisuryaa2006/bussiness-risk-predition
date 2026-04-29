"""
predict.py - Standalone prediction helper script.
Can be used from CLI to test the model without the Flask server.

Usage:
  python predict.py --revenue 500000 --expenses 300000 --profit_margin 0.25
                    --debt_ratio 0.5 --cash_flow 150000 --market_growth 5.5
                    --years_in_business 10 --employee_count 50
"""
import os
import argparse
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'business_risk_model.pkl')

def get_risk_category(score: float) -> str:
    if score < 33:
        return "🟢 Low Risk"
    elif score < 66:
        return "🟡 Medium Risk"
    else:
        return "🔴 High Risk"

def predict(data: dict) -> dict:
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Run 'python train_model.py' first.")
        return {}
    
    model = joblib.load(MODEL_PATH)
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    score = round(max(0, min(100, float(prediction))), 2)
    
    return {
        'risk_score': score,
        'category': get_risk_category(score)
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Business Risk Score')
    parser.add_argument('--revenue',            type=float, required=True)
    parser.add_argument('--expenses',           type=float, required=True)
    parser.add_argument('--profit_margin',      type=float, required=True)
    parser.add_argument('--debt_ratio',         type=float, required=True)
    parser.add_argument('--cash_flow',          type=float, required=True)
    parser.add_argument('--market_growth',      type=float, required=True)
    parser.add_argument('--years_in_business',  type=int,   required=True)
    parser.add_argument('--employee_count',     type=int,   required=True)
    
    args = parser.parse_args()
    result = predict(vars(args))
    
    print("\n" + "="*40)
    print("  BUSINESS RISK PREDICTION RESULT")
    print("="*40)
    print(f"  Risk Score  : {result['risk_score']} / 100")
    print(f"  Category    : {result['category']}")
    print("="*40 + "\n")
