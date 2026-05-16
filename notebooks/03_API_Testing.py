# =============================================================================
# CREDIT RISK INTELLIGENCE PLATFORM
# Notebook 03: API Testing, Live Demo & Monitoring
# Agent 3 | Vaidik Sharma | github.com/Vaidik6920
# =============================================================================

# ── CELL 1: Setup ─────────────────────────────────────────────────────────────
import requests
import json
import time
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt

# Set this to your deployed URL after running on Render/Railway
# Local:   http://localhost:8000
# Render:  https://credit-risk-api.onrender.com
BASE_URL = "http://localhost:8000"

print(f"Testing API at: {BASE_URL}")
print("Make sure the API is running: uvicorn api.main:app --reload")


# ── CELL 2: Health Check ──────────────────────────────────────────────────────
r = requests.get(f"{BASE_URL}/health")
print(f"\nGET /health  →  HTTP {r.status_code}")
print(json.dumps(r.json(), indent=2))
assert r.status_code == 200, "Health check failed!"
assert r.json()["model_loaded"], "Model not loaded!"
print("\n✅ API is healthy and model is loaded")


# ── CELL 3: Model Info ────────────────────────────────────────────────────────
r = requests.get(f"{BASE_URL}/model/info")
info = r.json()
print(f"\nGET /model/info  →  HTTP {r.status_code}")
print(json.dumps(info, indent=2))

print(f"\nModel version   : {info['model_version']}")
print(f"Training AUC    : {info['training_auc']}")
print(f"Threshold       : {info['optimal_threshold']}")
print(f"Feature count   : {info['feature_count']}")
print(f"Top features    : {', '.join(info['top_features'][:5])}")


# ── CELL 4: Single Prediction — Low Risk Applicant ───────────────────────────
LOW_RISK_APPLICANT = {
    "AMT_INCOME_TOTAL"   : 270000,   # Good income
    "AMT_CREDIT"         : 360000,   # Reasonable credit (1.33x income)
    "AMT_ANNUITY"        : 18000,    # Comfortable repayment
    "DAYS_BIRTH"         : -17520,   # 48 years old — established
    "DAYS_EMPLOYED"      : -5840,    # 16 years employment — stable
    "EXT_SOURCE_1"       : 0.78,     # High external scores
    "EXT_SOURCE_2"       : 0.82,
    "EXT_SOURCE_3"       : 0.71,
    "AMT_GOODS_PRICE"    : 337500,
    "NAME_CONTRACT_TYPE" : "Cash loans",
    "CODE_GENDER"        : "F",
    "FLAG_OWN_CAR"       : "Y",
    "FLAG_OWN_REALTY"    : "Y",
    "CNT_CHILDREN"       : 0,
    "CNT_FAM_MEMBERS"    : 2.0,
    "NAME_INCOME_TYPE"   : "Working",
    "NAME_EDUCATION_TYPE": "Higher education",
    "NAME_FAMILY_STATUS" : "Married",
    "NAME_HOUSING_TYPE"  : "House / apartment",
}

t0 = time.time()
r  = requests.post(f"{BASE_URL}/predict", json=LOW_RISK_APPLICANT)
latency_wall = (time.time() - t0) * 1000

print(f"\nPOST /predict (Low Risk)  →  HTTP {r.status_code}  ({latency_wall:.0f}ms wall)")
pred = r.json()
print(json.dumps(pred, indent=2))
assert r.status_code == 200
print(f"\n📊 Result: {pred['risk_label']}  |  P(default)={pred['default_probability']:.4f}  |  Score={pred['risk_score']}")
print(f"   Action: {pred['recommended_action']}")
print(f"   Top risk driver: {pred['top_risk_factors'][0]['feature'] if pred['top_risk_factors'] else 'N/A'}")


# ── CELL 5: Single Prediction — High Risk Applicant ──────────────────────────
HIGH_RISK_APPLICANT = {
    "AMT_INCOME_TOTAL"   : 67500,    # Low income
    "AMT_CREDIT"         : 675000,   # Massive credit (10x income!)
    "AMT_ANNUITY"        : 33750,    # Repayment = 50% of income — very high
    "DAYS_BIRTH"         : -9125,    # 25 years old — young
    "DAYS_EMPLOYED"      : -365,     # Only 1 year employment
    "EXT_SOURCE_1"       : 0.18,     # Very low external scores
    "EXT_SOURCE_2"       : 0.22,
    "EXT_SOURCE_3"       : 0.15,
    "AMT_GOODS_PRICE"    : 607500,
    "NAME_CONTRACT_TYPE" : "Cash loans",
    "CODE_GENDER"        : "M",
    "FLAG_OWN_CAR"       : "N",
    "FLAG_OWN_REALTY"    : "N",
    "CNT_CHILDREN"       : 3,
    "CNT_FAM_MEMBERS"    : 5.0,
    "NAME_INCOME_TYPE"   : "Working",
    "NAME_EDUCATION_TYPE": "Secondary / secondary special",
    "NAME_FAMILY_STATUS" : "Single / not married",
    "NAME_HOUSING_TYPE"  : "Rented apartment",
}

r = requests.post(f"{BASE_URL}/predict", json=HIGH_RISK_APPLICANT)
pred = r.json()
print(f"\nPOST /predict (High Risk)  →  HTTP {r.status_code}")
print(f"  P(default)         : {pred['default_probability']:.4f}")
print(f"  Risk Label         : {pred['risk_label']}")
print(f"  Risk Score (0-1000): {pred['risk_score']}")
print(f"  Recommended Action : {pred['recommended_action']}")
print(f"\n  Top Risk Factors:")
for factor in pred["top_risk_factors"]:
    arrow = "🔴" if factor["direction"] == "increases_risk" else "🟢"
    print(f"    {arrow} {factor['feature']:<35} SHAP={factor['shap_value']:+.4f}  — {factor['description']}")


# ── CELL 6: Batch Prediction ─────────────────────────────────────────────────
batch_payload = {
    "applications": [LOW_RISK_APPLICANT, HIGH_RISK_APPLICANT, LOW_RISK_APPLICANT],
    "return_shap": False,
}

t0 = time.time()
r  = requests.post(f"{BASE_URL}/predict/batch", json=batch_payload)
batch_wall = (time.time() - t0) * 1000

batch_resp = r.json()
print(f"\nPOST /predict/batch  →  HTTP {r.status_code}  ({batch_wall:.0f}ms for {batch_resp['total']} applications)")
for i, pred in enumerate(batch_resp["predictions"], 1):
    print(f"  Applicant {i}: P={pred['default_probability']:.4f}  Label={pred['risk_label']:20}  Score={pred['risk_score']}")


# ── CELL 7: Latency Benchmark ─────────────────────────────────────────────────
print("\n⏱  Latency Benchmark (20 sequential requests)")
latencies = []
for _ in range(20):
    t0 = time.time()
    requests.post(f"{BASE_URL}/predict", json=LOW_RISK_APPLICANT)
    latencies.append((time.time() - t0) * 1000)

print(f"  p50  : {np.percentile(latencies, 50):.1f}ms")
print(f"  p95  : {np.percentile(latencies, 95):.1f}ms")
print(f"  p99  : {np.percentile(latencies, 99):.1f}ms")
print(f"  max  : {max(latencies):.1f}ms")
print(f"  Target <50ms p95: {'✅' if np.percentile(latencies, 95) < 50 else '⚠️'}")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(latencies, "o-", color="#2563EB", ms=6, lw=1.5)
ax.axhline(50, color="red", linestyle="--", lw=1.5, label="50ms target")
ax.axhline(np.mean(latencies), color="green", linestyle="--", lw=1, label=f"Mean {np.mean(latencies):.1f}ms")
ax.set_xlabel("Request #"); ax.set_ylabel("Latency (ms)")
ax.set_title("API Latency — 20 Sequential Requests"); ax.legend()
plt.tight_layout()
plt.savefig("../data/plots/api_latency_benchmark.png", dpi=150, bbox_inches="tight")
plt.show()


# ── CELL 8: Drift Monitoring ─────────────────────────────────────────────────
r = requests.get(f"{BASE_URL}/monitoring/drift")
drift = r.json()
print(f"\nGET /monitoring/drift  →  HTTP {r.status_code}")
print(json.dumps(drift, indent=2))
print(f"\n  Overall PSI    : {drift['overall_psi']:.4f}")
print(f"  Status         : {drift['overall_status']}")
print(f"  Recommendation : {drift['recommendation']}")


# ── CELL 9: Score Distribution Simulation ────────────────────────────────────
# Simulate what a population of predictions looks like
np.random.seed(42)
n_low  = 9200  # ~92% non-defaulters
n_high = 800   # ~8% defaulters

# Simulate score distributions (mirror training distribution shape)
scores_good = np.random.beta(8, 2, n_low)   # skewed toward 1 (high scores)
scores_bad  = np.random.beta(2, 5, n_high)  # skewed toward 0 (low scores)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Score distribution
axes[0].hist(scores_good, bins=50, alpha=0.6, color="#2563EB", label=f"Non-Default ({n_low:,})", density=True)
axes[0].hist(scores_bad,  bins=50, alpha=0.6, color="#DC2626", label=f"Default ({n_high:,})", density=True)
axes[0].axvline(0.21, color="black", linestyle="--", lw=2, label="Optimal threshold 0.21")
axes[0].set_title("Simulated Score Distribution by True Class", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Predicted Default Probability"); axes[0].set_ylabel("Density")
axes[0].legend()

# Risk bucket breakdown
all_scores = np.concatenate([scores_good, scores_bad])
all_labels = np.concatenate([np.zeros(n_low), np.ones(n_high)])
buckets = ["Low (<10%)", "Medium (10-25%)", "High (25-50%)", "Very High (>50%)"]
counts  = [
    (all_scores < 0.10).sum(),
    ((all_scores >= 0.10) & (all_scores < 0.25)).sum(),
    ((all_scores >= 0.25) & (all_scores < 0.50)).sum(),
    (all_scores >= 0.50).sum(),
]
colors = ["#16A34A", "#D97706", "#DC2626", "#7C3AED"]
axes[1].bar(buckets, counts, color=colors, alpha=0.85, edgecolor="white", linewidth=1.5)
for i, (c, ct) in enumerate(zip(colors, counts)):
    axes[1].text(i, ct + 50, f"{ct:,}\n({ct/len(all_scores)*100:.1f}%)", ha="center", fontsize=10)
axes[1].set_title("Applications by Risk Bucket", fontsize=11, fontweight="bold")
axes[1].set_ylabel("Count")
axes[1].tick_params(axis="x", rotation=10)

plt.suptitle("Credit Risk Intelligence — Prediction Population Analysis", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("../data/plots/score_distribution_population.png", dpi=150, bbox_inches="tight")
plt.show()


# ── CELL 10: Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("📋 AGENT 3 COMPLETE — DELIVERABLES SUMMARY")
print("=" * 65)
print("""
  Files created:
    api/schemas.py          Pydantic v2 request/response models
    api/main.py             FastAPI app (8 endpoints, SHAP, logging)
    Dockerfile              Multi-stage, non-root, health check
    docker-compose.yml      API + MLflow UI services
    render.yaml             Render one-click deploy config
    .dockerignore           Clean build context
    tests/test_api.py       35+ pytest tests (mock-based, no GPU needed)
    .github/workflows/ci.yml  CI/CD: test → lint → docker build → push

  API Endpoints:
    GET  /              → service info
    GET  /health        → liveness + model status
    GET  /model/info    → version, weights, features, AUC
    POST /predict       → single prediction (<50ms target)
    POST /predict/batch → batch up to 500 applications
    GET  /monitoring/drift → PSI drift report (Evidently AI style)
    GET  /docs          → Swagger UI (auto-generated)
    GET  /redoc         → ReDoc API docs

  Deploy commands:
    # Local
    uvicorn api.main:app --reload

    # Docker
    docker-compose up --build

    # Render (after GitHub push)
    # Connect repo at dashboard.render.com → auto-deploys via render.yaml

  GitHub: github.com/Vaidik6920/credit-risk-intelligence
""")
