# 🏦 Credit Risk Intelligence Platform

> **Production-grade credit default prediction system** built on the Home Credit Default Risk dataset (307K+ applications). XGBoost + LightGBM + CatBoost ensemble with SHAP explainability, MLflow experiment tracking, and FastAPI serving.

[![Live Dashboard](https://img.shields.io/badge/Dashboard-Live%20on%20Streamlit-FF4B4B)](https://credit-risk-intelligence.streamlit.app)
[![Live API](https://img.shields.io/badge/API-Live%20on%20Render-brightgreen)](https://credit-risk-intelligence-xv4z.onrender.com/docs)
[![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.7899-blue)](https://github.com/Vaidik6920/credit-risk-intelligence)
[![MLflow](https://img.shields.io/badge/MLflow-30%20runs-orange)](https://github.com/Vaidik6920/credit-risk-intelligence)
[![Tests](https://img.shields.io/badge/Tests-36%2F36%20passing-brightgreen)](https://github.com/Vaidik6920/credit-risk-intelligence)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](https://github.com/Vaidik6920/credit-risk-intelligence)

**🎯 Live Dashboard:** https://credit-risk-intelligence.streamlit.app

**🔗 Live API Docs:** https://credit-risk-intelligence-xv4z.onrender.com/docs

---

## 🎯 Key Results

| Metric | Value |
|--------|-------|
| AUC-ROC | **0.7899** |
| vs. Logistic Baseline | **+12 pp** |
| Features Engineered | **271** |
| MLflow Experiments | **30 runs** |
| API Latency (no SHAP) | **~150ms** |
| Tests | **36/36 passing** |
| Dataset | **307K applications** |

---

## 🔌 Try It Live

```bash
# Health check
curl https://credit-risk-intelligence-xv4z.onrender.com/health

# Predict default probability
curl -X POST https://credit-risk-intelligence-xv4z.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "AMT_INCOME_TOTAL": 135000,
    "AMT_CREDIT": 406597,
    "AMT_ANNUITY": 24700,
    "DAYS_BIRTH": -14235,
    "DAYS_EMPLOYED": -2160,
    "EXT_SOURCE_1": 0.52,
    "EXT_SOURCE_2": 0.64,
    "EXT_SOURCE_3": 0.31
  }'
```

**Response:**
```json
{
  "default_probability": 0.0843,
  "risk_label": "Low Risk",
  "risk_score": 916,
  "recommended_action": "Approve",
  "model_version": "xgb_lgb_cat_ensemble_v1",
  "latency_ms": 148.3
}
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   DATA LAYER (8 CSV files)                   │
│  application_train/test · bureau · bureau_balance            │
│  previous_application · POS_CASH · credit_card              │
│  installments_payments                                        │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│              FEATURE ENGINEERING PIPELINE                    │
│  ApplicationFE → BureauAgg → PrevAppAgg → POSAgg →          │
│  CreditCardAgg → InstallmentAgg → WoE Encoding              │
│  Output: 271 features (Parquet)                             │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│               MODEL TRAINING (MLflow — 30 runs)              │
│  Logistic Baseline (0.73) → XGBoost (0.785) →              │
│  LightGBM (0.789) → Optuna tuning →                        │
│  CatBoost → Ensemble [XGB:0.55 LGB:0.45] → AUC 0.7899     │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│              SERVING LAYER                                   │
│  FastAPI /predict · /predict/batch · /health · /model/info  │
│  Optional SHAP · Evidently drift monitoring                 │
│  Streamlit Scorecard Dashboard (Live Ratio Calculator)      │
│  https://credit-risk-intelligence.streamlit.app             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Model Performance

| Model | OOF AUC-ROC |
|-------|------------|
| Logistic Regression (baseline) | 0.730 |
| XGBoost 5-fold CV | 0.785 |
| LightGBM 5-fold CV | 0.789 |
| **XGB + LGB + CatBoost Ensemble** | **0.7899** ⭐ |

### Top 10 SHAP Features
1. `EXT_SOURCE_2` — External credit score 2 (strongest predictor)
2. `EXT_SOURCE_3` — External credit score 3
3. `EXT_SOURCE_1` — External credit score 1
4. `inst_late_rate` — Historical installment late payment rate
5. `DAYS_BIRTH` — Applicant age
6. `bureau_overdue_sum` — Total overdue from bureau
7. `CREDIT_TO_INCOME` — Credit burden ratio
8. `DAYS_EMPLOYED` — Employment tenure
9. `cc_utilization_mean` — Credit card utilization
10. `prev_approval_rate` — Historical approval rate

---

## 📁 Project Structure

```
credit-risk-intelligence/
├── src/
│   ├── feature_engineering.py   271 features, 7 classes, WoE encoder
│   ├── train.py                 30 MLflow runs, XGB+LGB+CAT+Optuna+SHAP
│   └── evaluate.py              Metrics, threshold analysis, calibration
├── api/
│   ├── main.py                  FastAPI app, 8 endpoints
│   └── schemas.py               Pydantic v2 validation
├── streamlit_app.py             Live scoring dashboard (4 pages)
├── notebooks/
│   ├── 01_EDA_Credit_Risk.py
│   ├── 02_Model_Training_MLflow.py
│   └── 03_API_Testing.py
├── tests/                       36/36 tests passing
├── configs/config.yaml
├── Dockerfile
├── docker-compose.yml
└── render.yaml
```

---

## 🚀 Run Locally

```bash
git clone https://github.com/Vaidik6920/credit-risk-intelligence
cd credit-risk-intelligence
pip install -r requirements_render.txt

# Download dataset from kaggle.com/c/home-credit-default-risk
python src/feature_engineering.py
python src/train.py
uvicorn api.main:app --reload        # API → localhost:8000/docs
streamlit run streamlit_app.py       # Dashboard → localhost:8501
```

---

## 👤 Author

**Vaidik Sharma** | IIT Kharagpur 2026 | B.Tech Metallurgical Engineering
[github.com/Vaidik6920](https://github.com/Vaidik6920) · [LinkedIn](https://linkedin.com/in/vaidik-sharma-65733125b)
