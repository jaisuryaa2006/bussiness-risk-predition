# Business Risk Prediction using Ensemble Learning (Regression)

A complete end-to-end Machine Learning web application that predicts a **Business Risk Score (0–100)** based on key financial indicators using an ensemble of regression models.

---

## 🧠 ML Models Used

| Model | Type |
|---|---|
| Linear Regression | Baseline |
| Random Forest Regressor | Tree Ensemble |
| Gradient Boosting Regressor | Boosting |
| XGBoost Regressor | Boosting |
| **VotingRegressor (Ensemble)** | **Final Model** |

Metrics tracked: **R2 Score, MAE, MSE, RMSE**

---

## 🗂️ Project Structure

```
business-risk-prediction/
├── app.py                         # Flask application
├── train_model.py                 # Model training script
├── predict.py                     # CLI prediction helper
├── requirements.txt
├── render.yaml                    # Render deployment config
│
├── data/
│   └── business_risk_dataset.csv  # Auto-generated if missing
│
├── models/
│   ├── business_risk_model.pkl    # Saved best model
│   └── metrics.json               # Model evaluation results
│
├── templates/
│   ├── index.html                 # Landing page
│   └── predict.html               # Prediction form
│
├── static/
│   ├── css/style.css
│   └── js/main.js
│
└── .github/
    └── workflows/pipeline.yml    # CI/CD GitHub Actions
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

This will auto-generate a synthetic dataset if `data/business_risk_dataset.csv` is missing, train all models, and save the best one.

```bash
python train_model.py
```

### 3. Run the Flask App

```bash
python app.py
```

Visit: [http://localhost:5000](http://localhost:5000)

---

## 🖥️ API Reference

| Route | Method | Description |
|---|---|---|
| `/` | GET | Landing page with model charts |
| `/predict` | GET | Prediction form UI |
| `/api/predict` | POST | JSON API: returns risk score |
| `/metrics` | GET | Returns model comparison metrics |
| `/train` | POST | Triggers model retraining |

### `/api/predict` — Example Request

```json
POST /api/predict
Content-Type: application/json

{
  "revenue": 500000,
  "expenses": 300000,
  "profit_margin": 0.25,
  "debt_ratio": 0.5,
  "cash_flow": 150000,
  "market_growth": 5.5,
  "years_in_business": 10,
  "employee_count": 50
}
```

### Response

```json
{
  "score": 28.45,
  "category": "Low Risk"
}
```

---

## 📊 Risk Categories

| Score Range | Category |
|---|---|
| 0 – 32 | 🟢 Low Risk |
| 33 – 65 | 🟡 Medium Risk |
| 66 – 100 | 🔴 High Risk |

---

## 🐍 CLI Prediction (without Flask)

```bash
python predict.py \
  --revenue 500000 \
  --expenses 300000 \
  --profit_margin 0.25 \
  --debt_ratio 0.5 \
  --cash_flow 150000 \
  --market_growth 5.5 \
  --years_in_business 10 \
  --employee_count 50
```

---

## ☁️ Deployment

### Render (Auto)

The `render.yaml` file configures automatic deployment:

- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `gunicorn app:app`

### CI/CD (GitHub Actions)

The `.github/workflows/pipeline.yml` pipeline:
1. Installs dependencies
2. Runs `train_model.py`
3. Verifies model `.pkl` file was created
4. Triggers Render deploy via webhook (`RENDER_DEPLOY_HOOK_URL` secret)

---

## 🔐 GitHub Secrets Required

| Secret | Description |
|---|---|
| `RENDER_DEPLOY_HOOK_URL` | Render deploy hook URL for auto-deploy |

---

## 🖌️ UI Features

- **Dark glassmorphism** design with gradient accents
- **AOS scroll animations**
- **Chart.js** model comparison graphs (R2, MAE, RMSE)
- **Animated loading spinner** on prediction
- **Responsive** Bootstrap 5 layout
- Color-coded risk result card (**Green / Yellow / Red**)
