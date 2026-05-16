# 🏦 Credit Risk Intelligence Platform

> **Production-grade credit default prediction system** built on the Home Credit Default Risk dataset (307K+ applications). XGBoost + LightGBM ensemble with SHAP explainability, MLflow tracking, and FastAPI serving.

[![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.79-brightgreen)](https://github.com/Vaidik6920/credit-risk-intelligence)
[![MLflow](https://img.shields.io/badge/MLflow-15+%20runs-blue)](https://github.com/Vaidik6920/credit-risk-intelligence)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](https://github.com/Vaidik6920/credit-risk-intelligence)
[![FastAPI](https://img.shields.io/badge/FastAPI-<50ms-green)](https://github.com/Vaidik6920/credit-risk-intelligence)

---

## 🎯 Key Results

| Metric | Value |
|--------|-------|
| AUC-ROC | **0.79** |
| vs. Logistic Baseline | **+12 pp** |
| Features Engineered | **300+** |
| MLflow Experiments | **15+ runs** |
| API Latency | **<50ms** |
| Dataset Size | **307K applications** |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   DATA LAYER (7 CSV files)                   │
│  application_train/test │ bureau │ bureau_balance │          │
│  previous_application   │ POS_CASH │ credit_card │          │
│  installments_payments                                        │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│              FEATURE ENGINEERING PIPELINE                    │
│  ApplicationFE → BureauAgg → PrevAppAgg → POSAgg →          │
│  CreditCardAgg → InstallmentAgg → WoE Encoding              │
│  Output: 300+ features (Parquet)                            │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│               MODEL TRAINING (MLflow tracked)                │
│  XGBoost (5-fold CV) + LightGBM (5-fold CV)                 │
│  → Soft Voting Ensemble → SHAP Explainability               │
│  → Best model serialized to models/                         │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│                  SERVING LAYER                               │
│  FastAPI endpoint → /predict (JSON in, probability out)     │
│  Docker container → Deploy on Render/Railway free tier      │
│  Evidently AI → Data drift monitoring                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
credit-risk-intelligence/
├── notebooks/
│   ├── 01_EDA_Credit_Risk.py          # Full EDA (Agent 1)
│   ├── 02_Model_Training_MLflow.py    # XGB + LGB + SHAP (Agent 2)
│   └── 03_API_Testing.ipynb           # API smoke tests (Agent 3)
├── src/
│   ├── feature_engineering.py         # All FE classes (Agent 1)
│   ├── train.py                       # Training orchestrator (Agent 2)
│   ├── predict.py                     # Inference logic (Agent 3)
│   └── utils.py                       # Shared utilities
├── api/
│   ├── main.py                        # FastAPI app (Agent 3)
│   └── schemas.py                     # Pydantic models
├── configs/
│   └── config.yaml                    # Hyperparameters
├── data/
│   ├── raw/                           # Original CSVs (not committed)
│   └── processed/                     # Parquet feature files
├── models/                            # Serialized models
├── mlruns/                            # MLflow artifacts
├── tests/                             # Pytest suite
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── Makefile
└── README.md
```

---

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/Vaidik6920/credit-risk-intelligence.git
cd credit-risk-intelligence

# Download dataset from Kaggle
kaggle competitions download -c home-credit-default-risk
unzip home-credit-default-risk.zip -d data/raw/

# Install dependencies
pip install -r requirements.txt

# Run feature engineering
python src/feature_engineering.py

# Train models (tracked in MLflow)
python src/train.py

# Launch API
uvicorn api.main:app --reload

# Or with Docker
docker-compose up --build
```

---

## 📊 Feature Engineering Highlights

### Application-Level (40+ features)
- **Credit burden ratios**: `AMT_CREDIT / AMT_INCOME_TOTAL`, `AMT_ANNUITY / AMT_INCOME_TOTAL`
- **External source combinations**: mean, std, product of EXT_SOURCE_1/2/3
- **DAYS_EMPLOYED anomaly fix**: 365,243 → NaN + binary flag
- **Age/tenure conversions**: days → years for interpretability

### Bureau Aggregations (30+ features)
- Average/max/sum overdue amounts
- Active vs. closed loan ratio
- DPD (Days Past Due) statistics from bureau_balance
- Credit utilization across all external loans

### Historical Behavior (120+ features)
- **Installments**: late payment rate, underpayment frequency, avg days late
- **POS Cash**: DPD patterns, completion rate, future instalment ratio
- **Credit Card**: utilization ratio, over-limit count, payment ratio
- **Previous Applications**: approval rate, refusal rate, credit-to-goods ratio

### WoE Encoding
- Encodes all categorical features using Information Value (IV)
- Handles unseen categories gracefully (WoE = 0)
- IV table printed for feature importance analysis

---

## 🎯 Model Performance

| Model | CV AUC-ROC | Std |
|-------|------------|-----|
| Logistic Regression (baseline) | 0.67 | ±0.003 |
| XGBoost (tuned) | 0.776 | ±0.002 |
| LightGBM (tuned) | 0.779 | ±0.002 |
| **XGB + LGB Ensemble** | **0.790** | ±0.002 |

### Top SHAP Features
1. `EXT_SOURCE_2` — External credit score 2
2. `EXT_SOURCE_3` — External credit score 3
3. `EXT_SOURCE_1` — External credit score 1
4. `bureau_overdue_sum` — Total overdue from bureau
5. `inst_late_rate` — Historical installment late rate
6. `DAYS_BIRTH` — Applicant age
7. `CREDIT_TO_INCOME` — Credit burden ratio
8. `bureau_active_ratio` — Active loan ratio
9. `DAYS_EMPLOYED` — Employment tenure
10. `cc_utilization_mean` — Credit card utilization

---

## 🔌 API Usage

```python
import requests

payload = {
    "AMT_INCOME_TOTAL": 135000,
    "AMT_CREDIT": 406597,
    "AMT_ANNUITY": 24700,
    "DAYS_BIRTH": -14000,
    "DAYS_EMPLOYED": -2000,
    "EXT_SOURCE_1": 0.52,
    "EXT_SOURCE_2": 0.64,
    "EXT_SOURCE_3": 0.31,
    # ... other fields
}

response = requests.post("http://localhost:8000/predict", json=payload)
print(response.json())
# {
#   "default_probability": 0.0843,
#   "risk_label": "Low Risk",
#   "top_risk_factors": ["EXT_SOURCE_2", "bureau_overdue_sum", ...],
#   "model_version": "xgb_lgb_ensemble_v1"
# }
```

---

## 👤 Author

**Vaidik Sharma** | IIT Kharagpur 2026 | B.Tech Metallurgical Engineering  
[github.com/Vaidik6920](https://github.com/Vaidik6920) | [LinkedIn](https://linkedin.com/in/vaidik-sharma-65733125b)
