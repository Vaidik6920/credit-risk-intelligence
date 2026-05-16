"""
API Schemas — Credit Risk Intelligence Platform
Agent 3 | Vaidik Sharma | github.com/Vaidik6920

Pydantic v2 models for:
  - CreditApplicationRequest   (raw applicant fields → /predict)
  - PredictionResponse         (probability + label + SHAP drivers)
  - BatchPredictionRequest     (list of applications → /predict/batch)
  - HealthResponse             (GET /health)
  - ModelInfoResponse          (GET /model/info)
  - DriftReportResponse        (GET /monitoring/drift)
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────────────────────

class RiskLabel(str, Enum):
    LOW    = "Low Risk"
    MEDIUM = "Medium Risk"
    HIGH   = "High Risk"
    VERY_HIGH = "Very High Risk"


class ContractType(str, Enum):
    CASH_LOANS     = "Cash loans"
    REVOLVING_LOANS = "Revolving loans"


class GenderCode(str, Enum):
    M = "M"
    F = "F"
    XNA = "XNA"


class IncomeType(str, Enum):
    WORKING            = "Working"
    STATE_SERVANT      = "State servant"
    COMMERCIAL_ASSOCIATE = "Commercial associate"
    PENSIONER          = "Pensioner"
    UNEMPLOYED         = "Unemployed"
    STUDENT            = "Student"
    BUSINESSMAN        = "Businessman"
    MATERNITY_LEAVE    = "Maternity leave"


class EducationType(str, Enum):
    SECONDARY            = "Secondary / secondary special"
    HIGHER               = "Higher education"
    INCOMPLETE_HIGHER    = "Incomplete higher"
    LOWER_SECONDARY      = "Lower secondary"
    ACADEMIC_DEGREE      = "Academic degree"


class FamilyStatus(str, Enum):
    SINGLE             = "Single / not married"
    MARRIED            = "Married"
    CIVIL_MARRIAGE     = "Civil marriage"
    WIDOW              = "Widow"
    SEPARATED          = "Separated"


class HousingType(str, Enum):
    HOUSE_APARTMENT    = "House / apartment"
    RENTED_APARTMENT   = "Rented apartment"
    WITH_PARENTS       = "With parents"
    MUNICIPAL_APARTMENT = "Municipal apartment"
    OFFICE_APARTMENT   = "Office apartment"
    CO_OP_APARTMENT    = "Co-op apartment"


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST — single prediction
# ─────────────────────────────────────────────────────────────────────────────

class CreditApplicationRequest(BaseModel):
    """
    Applicant data for credit default probability prediction.
    Required fields are the minimum needed for the model.
    Optional fields improve accuracy — supply as many as available.
    """

    # ── Required core fields ──────────────────────────────────────────────────
    AMT_INCOME_TOTAL: float = Field(..., gt=0, description="Annual income (INR/USD)", example=135000)
    AMT_CREDIT:       float = Field(..., gt=0, description="Loan amount requested", example=406597)
    AMT_ANNUITY:      float = Field(..., gt=0, description="Annual loan repayment", example=24700)
    DAYS_BIRTH:       int   = Field(..., lt=0, description="Days before application (negative)", example=-14235)
    DAYS_EMPLOYED:    int   = Field(...,       description="Days employed (negative); 0 = unemployed", example=-2160)

    # ── External credit scores (most predictive) ──────────────────────────────
    EXT_SOURCE_1: Optional[float] = Field(None, ge=0, le=1, description="External credit score 1", example=0.52)
    EXT_SOURCE_2: Optional[float] = Field(None, ge=0, le=1, description="External credit score 2 (most important)", example=0.64)
    EXT_SOURCE_3: Optional[float] = Field(None, ge=0, le=1, description="External credit score 3", example=0.31)

    # ── Loan details ──────────────────────────────────────────────────────────
    AMT_GOODS_PRICE:    Optional[float] = Field(None, gt=0, description="Price of goods loan is for", example=351000)
    NAME_CONTRACT_TYPE: Optional[ContractType] = Field(None, example="Cash loans")

    # ── Demographic ───────────────────────────────────────────────────────────
    CODE_GENDER:        Optional[GenderCode]    = Field(None, example="M")
    FLAG_OWN_CAR:       Optional[str]           = Field(None, pattern="^[YN]$", example="Y")
    FLAG_OWN_REALTY:    Optional[str]           = Field(None, pattern="^[YN]$", example="N")
    CNT_CHILDREN:       Optional[int]           = Field(None, ge=0, le=20, example=1)
    CNT_FAM_MEMBERS:    Optional[float]         = Field(None, ge=1, le=20, example=3.0)
    NAME_INCOME_TYPE:   Optional[IncomeType]    = Field(None)
    NAME_EDUCATION_TYPE: Optional[EducationType] = Field(None)
    NAME_FAMILY_STATUS: Optional[FamilyStatus]  = Field(None)
    NAME_HOUSING_TYPE:  Optional[HousingType]   = Field(None)

    # ── Registration / document days ─────────────────────────────────────────
    DAYS_REGISTRATION: Optional[float] = Field(None, le=0, example=-4380.0)
    DAYS_ID_PUBLISH:   Optional[int]   = Field(None, le=0, example=-2922)

    # ── Contact flags ─────────────────────────────────────────────────────────
    FLAG_MOBIL:       Optional[int] = Field(None, ge=0, le=1, example=1)
    FLAG_EMP_PHONE:   Optional[int] = Field(None, ge=0, le=1, example=1)
    FLAG_WORK_PHONE:  Optional[int] = Field(None, ge=0, le=1, example=0)
    FLAG_PHONE:       Optional[int] = Field(None, ge=0, le=1, example=0)
    FLAG_EMAIL:       Optional[int] = Field(None, ge=0, le=1, example=0)

    # ── Region ────────────────────────────────────────────────────────────────
    REGION_POPULATION_RELATIVE: Optional[float] = Field(None, ge=0, le=1, example=0.0187)
    REGION_RATING_CLIENT:       Optional[int]   = Field(None, ge=1, le=3, example=2)

    # ── Credit bureau enquiries ───────────────────────────────────────────────
    AMT_REQ_CREDIT_BUREAU_MON:  Optional[float] = Field(None, ge=0, example=0.0)
    AMT_REQ_CREDIT_BUREAU_QRT:  Optional[float] = Field(None, ge=0, example=0.0)
    AMT_REQ_CREDIT_BUREAU_YEAR: Optional[float] = Field(None, ge=0, example=1.0)

    # ── Explanation flag ─────────────────────────────────────────────────────
    explain: bool = Field(False, description="Set true to return SHAP-based top_risk_factors (adds ~200ms)")

    @field_validator("DAYS_EMPLOYED")
    @classmethod
    def validate_days_employed(cls, v):
        # 0 = unemployed (will be treated as NaN in FE); positive not allowed
        if v > 0:
            raise ValueError("DAYS_EMPLOYED must be 0 (unemployed) or negative")
        return v

    @model_validator(mode="after")
    def validate_credit_sanity(self):
        if self.AMT_GOODS_PRICE and self.AMT_CREDIT:
            if self.AMT_CREDIT < self.AMT_GOODS_PRICE * 0.5:
                raise ValueError("AMT_CREDIT seems too low relative to AMT_GOODS_PRICE")
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "AMT_INCOME_TOTAL": 135000,
                "AMT_CREDIT":       406597,
                "AMT_ANNUITY":      24700,
                "DAYS_BIRTH":       -14235,
                "DAYS_EMPLOYED":    -2160,
                "EXT_SOURCE_1":     0.52,
                "EXT_SOURCE_2":     0.64,
                "EXT_SOURCE_3":     0.31,
                "AMT_GOODS_PRICE":  351000,
                "NAME_CONTRACT_TYPE": "Cash loans",
                "CODE_GENDER":      "M",
                "FLAG_OWN_CAR":     "Y",
                "FLAG_OWN_REALTY":  "N",
                "CNT_CHILDREN":     1,
                "CNT_FAM_MEMBERS":  3.0,
                "NAME_INCOME_TYPE": "Working",
                "NAME_EDUCATION_TYPE": "Higher education",
                "NAME_FAMILY_STATUS": "Married",
                "NAME_HOUSING_TYPE": "House / apartment",
            }
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE — single prediction
# ─────────────────────────────────────────────────────────────────────────────

class RiskFactor(BaseModel):
    feature:     str   = Field(..., description="Feature name")
    shap_value:  float = Field(..., description="SHAP contribution (positive = increases default risk)")
    direction:   str   = Field(..., description="'increases_risk' | 'decreases_risk'")
    description: str   = Field(..., description="Human-readable explanation")


class PredictionResponse(BaseModel):
    default_probability: float     = Field(..., ge=0, le=1, description="P(default) from ensemble model")
    risk_label:          RiskLabel = Field(..., description="Risk category")
    risk_score:          int       = Field(..., ge=0, le=1000, description="Score 0-1000 (lower = riskier)")
    recommended_action:  str       = Field(..., description="Approve / Review / Decline")
    top_risk_factors:    List[RiskFactor] = Field(..., description="Top 5 SHAP risk drivers")
    model_version:       str       = Field(..., description="Model version identifier")
    prediction_id:       str       = Field(..., description="UUID for audit trail")
    latency_ms:          float     = Field(..., description="Inference latency in milliseconds")

    model_config = {
        "json_schema_extra": {
            "example": {
                "default_probability": 0.0843,
                "risk_label":          "Low Risk",
                "risk_score":          916,
                "recommended_action":  "Approve",
                "top_risk_factors": [
                    {"feature": "EXT_SOURCE_2", "shap_value": -0.312, "direction": "decreases_risk",
                     "description": "Strong external credit score reduces default risk"},
                    {"feature": "inst_late_rate", "shap_value": 0.041, "direction": "increases_risk",
                     "description": "Some historical late payments add marginal risk"},
                ],
                "model_version":  "xgb_lgb_cat_ensemble_v1",
                "prediction_id":  "a3f7c821-1b4e-4d9a-8f2c-9e0d7b3a1f55",
                "latency_ms":     12.4,
            }
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE — batch prediction
# ─────────────────────────────────────────────────────────────────────────────

class BatchPredictionRequest(BaseModel):
    applications: List[CreditApplicationRequest] = Field(
        ..., min_length=1, max_length=500,
        description="List of credit applications (max 500 per batch)"
    )
    return_shap: bool = Field(False, description="Whether to return SHAP values (slower)")


class BatchPredictionResponse(BaseModel):
    predictions:  List[PredictionResponse]
    total:        int
    batch_latency_ms: float


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH & INFO
# ─────────────────────────────────────────────────────────────────────────────

class ModelStatus(str, Enum):
    READY   = "ready"
    LOADING = "loading"
    ERROR   = "error"


class HealthResponse(BaseModel):
    status:        ModelStatus
    model_loaded:  bool
    model_version: str
    uptime_seconds: float
    total_predictions: int


class ModelInfoResponse(BaseModel):
    model_version:    str
    ensemble_weights: Dict[str, float]
    optimal_threshold: float
    feature_count:    int
    training_auc:     float
    top_features:     List[str]
    deployed_at:      str


# ─────────────────────────────────────────────────────────────────────────────
# MONITORING
# ─────────────────────────────────────────────────────────────────────────────

class DriftStatus(str, Enum):
    STABLE   = "stable"
    MODERATE = "moderate"
    CRITICAL = "critical"


class FeatureDrift(BaseModel):
    feature:    str
    psi:        float
    status:     DriftStatus
    mean_train: Optional[float] = None
    mean_live:  Optional[float] = None


class DriftReportResponse(BaseModel):
    overall_psi:      float
    overall_status:   DriftStatus
    drifted_features: List[FeatureDrift]
    report_generated_at: str
    recommendation:   str
