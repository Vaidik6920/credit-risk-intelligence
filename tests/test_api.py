"""
Test Suite — Credit Risk Intelligence Platform
Agent 3 | Vaidik Sharma | github.com/Vaidik6920

Run: pytest tests/ -v --tb=short
"""

import json
import pytest
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from api.main import app


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """TestClient with model cache mocked — no actual model files needed."""
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.92, 0.08]])

    mock_shap = MagicMock()
    mock_shap.shap_values.return_value = np.array([[
        -0.31, 0.04, -0.18, 0.02, -0.09, 0.01, 0.03, -0.12, 0.05, -0.07,
        0.01, 0.02, -0.04, 0.01, 0.03,
    ]])
    mock_shap.expected_value = -2.5

    with patch("api.main._cache", {
        "xgb"              : mock_model,
        "lgb"              : mock_model,
        "cat"              : mock_model,
        "feature_pipeline" : None,
        "shap_explainer"   : mock_shap,
        "inference_config" : {"optimal_threshold": 0.21, "ensemble_weights": [0.4, 0.4, 0.2]},
        "feature_names"    : None,
        "shap_top10"       : ["EXT_SOURCE_2", "inst_late_rate", "DAYS_BIRTH"],
        "startup_time"     : 1700000000.0,
        "total_predictions": 42,
    }):
        with TestClient(app) as c:
            yield c


VALID_PAYLOAD = {
    "AMT_INCOME_TOTAL"   : 135000,
    "AMT_CREDIT"         : 406597,
    "AMT_ANNUITY"        : 24700,
    "DAYS_BIRTH"         : -14235,
    "DAYS_EMPLOYED"      : -2160,
    "EXT_SOURCE_1"       : 0.52,
    "EXT_SOURCE_2"       : 0.64,
    "EXT_SOURCE_3"       : 0.31,
    "AMT_GOODS_PRICE"    : 351000,
    "NAME_CONTRACT_TYPE" : "Cash loans",
    "CODE_GENDER"        : "M",
    "FLAG_OWN_CAR"       : "Y",
    "FLAG_OWN_REALTY"    : "N",
    "CNT_CHILDREN"       : 1,
    "CNT_FAM_MEMBERS"    : 3.0,
    "NAME_INCOME_TYPE"   : "Working",
    "NAME_EDUCATION_TYPE": "Higher education",
    "NAME_FAMILY_STATUS" : "Married",
    "NAME_HOUSING_TYPE"  : "House / apartment",
}


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_schema(self, client):
        r = client.get("/health").json()
        assert "status"             in r
        assert "model_loaded"       in r
        assert "model_version"      in r
        assert "uptime_seconds"     in r
        assert "total_predictions"  in r

    def test_health_model_loaded(self, client):
        r = client.get("/health").json()
        assert r["model_loaded"] is True
        assert r["status"] == "ready"

    def test_health_total_predictions_is_int(self, client):
        r = client.get("/health").json()
        assert isinstance(r["total_predictions"], int)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL INFO ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

class TestModelInfo:
    def test_model_info_200(self, client):
        r = client.get("/model/info")
        assert r.status_code == 200

    def test_model_info_schema(self, client):
        r = client.get("/model/info").json()
        assert "model_version"     in r
        assert "ensemble_weights"  in r
        assert "optimal_threshold" in r
        assert "feature_count"     in r
        assert "training_auc"      in r
        assert "top_features"      in r

    def test_model_info_auc_range(self, client):
        r = client.get("/model/info").json()
        assert 0.5 <= r["training_auc"] <= 1.0

    def test_ensemble_weights_sum_to_one(self, client):
        r = client.get("/model/info").json()
        total = sum(r["ensemble_weights"].values())
        assert abs(total - 1.0) < 1e-3


# ─────────────────────────────────────────────────────────────────────────────
# PREDICT ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

class TestPredict:
    def test_predict_200(self, client):
        r = client.post("/predict", json=VALID_PAYLOAD)
        assert r.status_code == 200

    def test_predict_response_schema(self, client):
        r = client.post("/predict", json=VALID_PAYLOAD).json()
        assert "default_probability" in r
        assert "risk_label"          in r
        assert "risk_score"          in r
        assert "recommended_action"  in r
        assert "top_risk_factors"    in r
        assert "model_version"       in r
        assert "prediction_id"       in r
        assert "latency_ms"          in r

    def test_predict_probability_in_range(self, client):
        r = client.post("/predict", json=VALID_PAYLOAD).json()
        assert 0.0 <= r["default_probability"] <= 1.0

    def test_predict_risk_score_in_range(self, client):
        r = client.post("/predict", json=VALID_PAYLOAD).json()
        assert 0 <= r["risk_score"] <= 1000

    def test_predict_risk_label_valid(self, client):
        r = client.post("/predict", json=VALID_PAYLOAD).json()
        valid_labels = {"Low Risk", "Medium Risk", "High Risk", "Very High Risk"}
        assert r["risk_label"] in valid_labels

    def test_predict_action_valid(self, client):
        r = client.post("/predict", json=VALID_PAYLOAD).json()
        valid_actions = {"Approve", "Approve with conditions", "Manual review required", "Decline"}
        assert r["recommended_action"] in valid_actions

    def test_predict_risk_factors_list(self, client):
        r = client.post("/predict", json=VALID_PAYLOAD).json()
        assert isinstance(r["top_risk_factors"], list)

    def test_predict_risk_factor_schema(self, client):
        r = client.post("/predict", json=VALID_PAYLOAD).json()
        if r["top_risk_factors"]:
            factor = r["top_risk_factors"][0]
            assert "feature"     in factor
            assert "shap_value"  in factor
            assert "direction"   in factor
            assert "description" in factor
            assert factor["direction"] in ("increases_risk", "decreases_risk")

    def test_predict_latency_logged(self, client):
        r = client.post("/predict", json=VALID_PAYLOAD).json()
        assert r["latency_ms"] >= 0

    def test_predict_prediction_id_is_uuid(self, client):
        import uuid
        r = client.post("/predict", json=VALID_PAYLOAD).json()
        # Should not raise ValueError
        uuid.UUID(r["prediction_id"])

    def test_predict_missing_required_field(self, client):
        bad = {k: v for k, v in VALID_PAYLOAD.items() if k != "AMT_INCOME_TOTAL"}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_predict_invalid_income(self, client):
        bad = {**VALID_PAYLOAD, "AMT_INCOME_TOTAL": -1}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_predict_invalid_gender(self, client):
        bad = {**VALID_PAYLOAD, "CODE_GENDER": "X"}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_predict_ext_source_out_of_range(self, client):
        bad = {**VALID_PAYLOAD, "EXT_SOURCE_2": 1.5}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_predict_days_employed_positive_rejected(self, client):
        bad = {**VALID_PAYLOAD, "DAYS_EMPLOYED": 500}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_predict_minimum_required_fields(self, client):
        """Only the 5 required fields — should still succeed."""
        minimal = {
            "AMT_INCOME_TOTAL": 100000,
            "AMT_CREDIT":       300000,
            "AMT_ANNUITY":      20000,
            "DAYS_BIRTH":       -12000,
            "DAYS_EMPLOYED":    -1500,
        }
        r = client.post("/predict", json=minimal)
        assert r.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# BATCH PREDICT ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchPredict:
    def test_batch_predict_200(self, client):
        r = client.post("/predict/batch", json={"applications": [VALID_PAYLOAD]})
        assert r.status_code == 200

    def test_batch_predict_schema(self, client):
        r = client.post("/predict/batch", json={"applications": [VALID_PAYLOAD]}).json()
        assert "predictions"       in r
        assert "total"             in r
        assert "batch_latency_ms"  in r

    def test_batch_total_matches_input(self, client):
        payload = {"applications": [VALID_PAYLOAD, VALID_PAYLOAD, VALID_PAYLOAD]}
        r = client.post("/predict/batch", json=payload).json()
        assert r["total"] == 3
        assert len(r["predictions"]) == 3

    def test_batch_empty_list_rejected(self, client):
        r = client.post("/predict/batch", json={"applications": []})
        assert r.status_code == 422

    def test_batch_latency_positive(self, client):
        r = client.post("/predict/batch", json={"applications": [VALID_PAYLOAD]}).json()
        assert r["batch_latency_ms"] >= 0


# ─────────────────────────────────────────────────────────────────────────────
# MONITORING ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

class TestDriftReport:
    def test_drift_report_200(self, client):
        r = client.get("/monitoring/drift")
        assert r.status_code == 200

    def test_drift_report_schema(self, client):
        r = client.get("/monitoring/drift").json()
        assert "overall_psi"         in r
        assert "overall_status"      in r
        assert "drifted_features"    in r
        assert "report_generated_at" in r
        assert "recommendation"      in r

    def test_drift_psi_non_negative(self, client):
        r = client.get("/monitoring/drift").json()
        assert r["overall_psi"] >= 0

    def test_drift_status_valid(self, client):
        r = client.get("/monitoring/drift").json()
        assert r["overall_status"] in ("stable", "moderate", "critical")


# ─────────────────────────────────────────────────────────────────────────────
# ROOT ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

class TestRoot:
    def test_root_200(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_root_has_docs_link(self, client):
        r = client.get("/").json()
        assert "docs" in r


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE — latency smoke test (not a load test)
# ─────────────────────────────────────────────────────────────────────────────

class TestPerformance:
    def test_single_predict_under_200ms(self, client):
        """Wall-clock test; mocked models are near-instantaneous."""
        import time
        t0 = time.perf_counter()
        r  = client.post("/predict", json=VALID_PAYLOAD)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert r.status_code == 200
        # Mock is fast; 200ms gives generous headroom for real model
        assert elapsed_ms < 200, f"Latency too high: {elapsed_ms:.0f}ms"
