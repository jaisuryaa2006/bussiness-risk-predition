import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_PATH = os.path.join(DATA_DIR, 'business_risk_dataset.csv')
MODEL_PATH = os.path.join(MODEL_DIR, 'business_risk_model.pkl')
METRICS_PATH = os.path.join(MODEL_DIR, 'metrics.json')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def generate_synthetic_data(samples=1000):
    np.random.seed(42)
    revenue = np.random.uniform(50000, 5000000, samples)
    expenses = revenue * np.random.uniform(0.4, 1.2, samples)
    profit_margin = (revenue - expenses) / revenue
    debt_ratio = np.random.uniform(0, 1.5, samples)
    cash_flow = revenue - expenses - (np.random.uniform(1000, 50000, samples))
    market_growth = np.random.uniform(-5, 20, samples)
    years_in_business = np.random.randint(1, 50, samples)
    employee_count = np.random.randint(2, 500, samples)

    # Calculate synthetic risk score
    # Lower profit margin, higher debt ratio -> higher risk
    base_risk = 50
    risk = base_risk - (profit_margin * 50) + (debt_ratio * 20) - (market_growth * 0.5) - (np.log(years_in_business) * 2) - (cash_flow / 100000)
    risk = np.clip(risk, 0, 100) # Scale 0 to 100
    
    df = pd.DataFrame({
        'revenue': revenue,
        'expenses': expenses,
        'profit_margin': profit_margin,
        'debt_ratio': debt_ratio,
        'cash_flow': cash_flow,
        'market_growth': market_growth,
        'years_in_business': years_in_business,
        'employee_count': employee_count,
        'risk_score': risk
    })
    df.to_csv(DATA_PATH, index=False)
    print(f"Generated synthetic dataset at {DATA_PATH}")

def get_risk_category_for_eval(score):
    if score < 33: return 0
    elif score < 66: return 1
    else: return 2

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    y_test_cat = [get_risk_category_for_eval(y) for y in y_test]
    preds_cat = [get_risk_category_for_eval(p) for p in preds]
    return {
        'Accuracy': float(accuracy_score(y_test_cat, preds_cat) * 100),
        'R2': float(r2_score(y_test, preds)),
        'MAE': float(mean_absolute_error(y_test, preds)),
        'MSE': float(mean_squared_error(y_test, preds)),
        'RMSE': float(np.sqrt(mean_squared_error(y_test, preds)))
    }

def train_models():
    if not os.path.exists(DATA_PATH):
        generate_synthetic_data()
        
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    X = df.drop('risk_score', axis=1)
    y = df['risk_score']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Models to evaluate
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    metrics = {}
    fitted_estimators = []
    
    print("Training base models...")
    for name, model in models.items():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', model)
        ])
        pipeline.fit(X_train, y_train)
        metrics[name] = evaluate_model(pipeline, X_test, y_test)
        
        # for VotingRegressor
        # VotingRegressor expects base estimators. We'll standard-scale the data, then provide base estimators
        fitted_estimators.append((name, model)) 

    print("Training Ensemble (VotingRegressor)...")
    ensemble = VotingRegressor(estimators=[(k, v) for k, v in models.items()])
    
    final_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ensemble', ensemble)
    ])
    
    final_pipeline.fit(X_train, y_train)
    metrics['Ensemble (Voting)'] = evaluate_model(final_pipeline, X_test, y_test)
    
    # Save the metrics to a JSON file
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Save the pipeline
    joblib.dump(final_pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print("Training completed. Metrics saved.")

if __name__ == '__main__':
    train_models()
