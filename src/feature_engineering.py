"""
Feature Engineering Pipeline — Credit Risk Intelligence Platform
Agent 1 | Vaidik Sharma | github.com/Vaidik6920

Architecture:
  ApplicationFeatureEngineer   ← application_train/test level features
  BureauAggregator             ← bureau.csv + bureau_balance.csv
  PrevApplicationAggregator    ← previous_application.csv
  POSCashAggregator            ← POS_CASH_balance.csv
  CreditCardAggregator         ← credit_card_balance.csv
  InstallmentAggregator        ← installments_payments.csv
  WoEEncoder                   ← Weight-of-Evidence for categoricals
  CreditRiskFeaturePipeline    ← orchestrates all of the above

Target: 0.79 AUC-ROC | Features: ~300+ after all aggregations
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.stats import chi2_contingency
import gc

# ─────────────────────────────────────────────────────────────────────────────
# 1. APPLICATION-LEVEL FEATURES
# ─────────────────────────────────────────────────────────────────────────────

class ApplicationFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Hand-crafted features from application_train / application_test.
    Generates ~40 new features on top of the original 122.
    """

    ANOMALY_VALUE = 365243  # Sentinel for unemployed in DAYS_EMPLOYED

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # ── 1a. Fix DAYS_EMPLOYED anomaly ────────────────────────────────────
        df["FLAG_DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == self.ANOMALY_VALUE).astype(np.int8)
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(self.ANOMALY_VALUE, np.nan)

        # ── 1b. Age / tenure conversions ────────────────────────────────────
        df["AGE_YEARS"]           = -df["DAYS_BIRTH"] / 365.25
        df["EMPLOYED_YEARS"]      = -df["DAYS_EMPLOYED"] / 365.25
        df["REGISTRATION_YEARS"]  = -df["DAYS_REGISTRATION"] / 365.25
        df["ID_PUBLISH_YEARS"]    = -df["DAYS_ID_PUBLISH"] / 365.25

        # ── 1c. Credit / income ratios ───────────────────────────────────────
        df["CREDIT_TO_INCOME"]      = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)
        df["ANNUITY_TO_INCOME"]     = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)
        df["CREDIT_TO_GOODS"]       = df["AMT_CREDIT"] / (df["AMT_GOODS_PRICE"] + 1)
        df["ANNUITY_TO_CREDIT"]     = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + 1)
        df["INCOME_PER_PERSON"]     = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"] + 1)
        df["INCOME_PER_CHILD"]      = df["AMT_INCOME_TOTAL"] / (df["CNT_CHILDREN"] + 1)

        # ── 1d. Days-based interactions ──────────────────────────────────────
        df["EMPLOYED_TO_AGE"]       = df["EMPLOYED_YEARS"] / (df["AGE_YEARS"] + 1)
        df["EMPLOYED_MINUS_AGE"]    = df["DAYS_EMPLOYED"] - df["DAYS_BIRTH"]

        # ── 1e. External source combinations ────────────────────────────────
        df["EXT_SOURCE_MEAN"]       = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
        df["EXT_SOURCE_STD"]        = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].std(axis=1)
        df["EXT_SOURCE_PROD"]       = (
            df["EXT_SOURCE_1"].fillna(0) *
            df["EXT_SOURCE_2"].fillna(0) *
            df["EXT_SOURCE_3"].fillna(0)
        )
        df["EXT_SOURCE_MAX"]        = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].max(axis=1)
        df["EXT_SOURCE_MIN"]        = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].min(axis=1)
        # Pairwise interactions (highly informative per Kaggle winners)
        df["EXT_1_2"]               = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"]
        df["EXT_1_3"]               = df["EXT_SOURCE_1"] * df["EXT_SOURCE_3"]
        df["EXT_2_3"]               = df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]

        # ── 1f. Document flags aggregation ───────────────────────────────────
        doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
        df["TOTAL_DOCS_SUBMITTED"]  = df[doc_cols].sum(axis=1)

        # ── 1g. Social circle defaults ───────────────────────────────────────
        df["SOCIAL_CIRCLE_DEFAULT_30_RATIO"] = (
            df["DEF_30_CNT_SOCIAL_CIRCLE"] /
            (df["OBS_30_CNT_SOCIAL_CIRCLE"] + 1)
        )
        df["SOCIAL_CIRCLE_DEFAULT_60_RATIO"] = (
            df["DEF_60_CNT_SOCIAL_CIRCLE"] /
            (df["OBS_60_CNT_SOCIAL_CIRCLE"] + 1)
        )

        # ── 1h. Enquiry flags (last X months) ────────────────────────────────
        enq_cols = [
            "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY",
            "AMT_REQ_CREDIT_BUREAU_WEEK", "AMT_REQ_CREDIT_BUREAU_MON",
            "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR",
        ]
        df["TOTAL_BUREAU_ENQUIRIES"] = df[enq_cols].sum(axis=1)

        # ── 1i. Contact flags aggregation ────────────────────────────────────
        contact_cols = [
            "FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE",
            "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL",
        ]
        df["TOTAL_CONTACT_FLAGS"] = df[contact_cols].sum(axis=1)

        # ── 1j. Goods price difference ───────────────────────────────────────
        df["GOODS_CREDIT_DIFF"] = df["AMT_GOODS_PRICE"] - df["AMT_CREDIT"]

        return df

    def get_feature_names_out(self, input_features=None):
        return None  # handled downstream


# ─────────────────────────────────────────────────────────────────────────────
# 2. BUREAU AGGREGATOR
# ─────────────────────────────────────────────────────────────────────────────

class BureauAggregator:
    """
    Aggregates bureau.csv and bureau_balance.csv per SK_ID_CURR.
    Generates ~80 features capturing external credit history.
    """

    def fit_transform(self, bureau: pd.DataFrame, bureau_balance: pd.DataFrame) -> pd.DataFrame:
        # ── 2a. Bureau Balance aggregations ──────────────────────────────────
        # STATUS: 0=on time, 1–5=overdue buckets, C=closed, X=unknown
        bureau_balance["STATUS_NUMERIC"] = bureau_balance["STATUS"].map(
            {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "C": 0, "X": np.nan}
        )
        bb_agg = bureau_balance.groupby("SK_ID_BUREAU").agg(
            bb_months_count       = ("MONTHS_BALANCE", "count"),
            bb_status_mean        = ("STATUS_NUMERIC", "mean"),
            bb_status_max         = ("STATUS_NUMERIC", "max"),
            bb_status_sum         = ("STATUS_NUMERIC", "sum"),
            bb_dpd_over_1_count   = ("STATUS_NUMERIC", lambda x: (x >= 1).sum()),
            bb_dpd_over_2_count   = ("STATUS_NUMERIC", lambda x: (x >= 2).sum()),
        ).reset_index()

        # Merge bureau balance into bureau
        bureau_full = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")

        # ── 2b. Bureau per-applicant aggregations ────────────────────────────
        cat_agg = {}
        for status in ["Active", "Closed", "Sold", "Bad debt"]:
            cat_agg[f"bureau_{status.lower().replace(' ', '_')}_count"] = (
                "CREDIT_ACTIVE", lambda x, s=status: (x == s).sum()
            )

        num_agg_dict = {
            # Credit amounts
            "AMT_CREDIT_SUM":              ["sum", "mean", "max"],
            "AMT_CREDIT_SUM_DEBT":         ["sum", "mean", "max"],
            "AMT_CREDIT_SUM_OVERDUE":      ["sum", "mean", "max"],
            "AMT_CREDIT_SUM_LIMIT":        ["sum", "mean"],
            # Days
            "DAYS_CREDIT":                 ["mean", "max", "min", "std"],
            "DAYS_CREDIT_ENDDATE":         ["mean", "max"],
            "DAYS_CREDIT_UPDATE":          ["mean"],
            "DAYS_OVERDUE":                ["mean", "max", "sum"],
            # Credit count and bureau balance
            "CNT_CREDIT_PROLONG":          ["sum", "mean"],
            "bb_months_count":             ["mean", "sum"],
            "bb_status_mean":              ["mean", "max"],
            "bb_status_max":               ["max", "mean"],
            "bb_dpd_over_1_count":         ["sum", "mean"],
            "bb_dpd_over_2_count":         ["sum", "mean"],
        }

        bureau_agg = bureau_full.groupby("SK_ID_CURR").agg(
            bureau_total_loans         = ("SK_ID_BUREAU", "count"),
            bureau_active_loans        = ("CREDIT_ACTIVE", lambda x: (x == "Active").sum()),
            bureau_closed_loans        = ("CREDIT_ACTIVE", lambda x: (x == "Closed").sum()),
            bureau_credit_sum          = ("AMT_CREDIT_SUM", "sum"),
            bureau_credit_sum_mean     = ("AMT_CREDIT_SUM", "mean"),
            bureau_credit_sum_max      = ("AMT_CREDIT_SUM", "max"),
            bureau_debt_sum            = ("AMT_CREDIT_SUM_DEBT", "sum"),
            bureau_debt_mean           = ("AMT_CREDIT_SUM_DEBT", "mean"),
            bureau_overdue_sum         = ("AMT_CREDIT_SUM_OVERDUE", "sum"),
            bureau_overdue_mean        = ("AMT_CREDIT_SUM_OVERDUE", "mean"),
            bureau_overdue_max         = ("AMT_CREDIT_SUM_OVERDUE", "max"),
            bureau_credit_limit_mean   = ("AMT_CREDIT_SUM_LIMIT", "mean"),
            bureau_days_credit_mean    = ("DAYS_CREDIT", "mean"),
            bureau_days_credit_max     = ("DAYS_CREDIT", "max"),
            bureau_days_credit_min     = ("DAYS_CREDIT", "min"),
            bureau_days_credit_std     = ("DAYS_CREDIT", "std"),
            bureau_days_overdue_mean   = ("CREDIT_DAY_OVERDUE", "mean"),
            bureau_days_overdue_max    = ("CREDIT_DAY_OVERDUE", "max"),
            bureau_days_overdue_sum    = ("CREDIT_DAY_OVERDUE", "sum"),
            bureau_days_enddate_mean   = ("DAYS_CREDIT_ENDDATE", "mean"),
            bureau_prolong_sum         = ("CNT_CREDIT_PROLONG", "sum"),
            bureau_bb_months_mean      = ("bb_months_count", "mean"),
            bureau_bb_status_mean      = ("bb_status_mean", "mean"),
            bureau_bb_status_max       = ("bb_status_max", "max"),
            bureau_bb_dpd1_sum         = ("bb_dpd_over_1_count", "sum"),
            bureau_bb_dpd2_sum         = ("bb_dpd_over_2_count", "sum"),
        ).reset_index()

        # ── 2c. Ratio features ────────────────────────────────────────────────
        bureau_agg["bureau_active_ratio"]     = (
            bureau_agg["bureau_active_loans"] / (bureau_agg["bureau_total_loans"] + 1)
        )
        bureau_agg["bureau_debt_credit_ratio"] = (
            bureau_agg["bureau_debt_sum"] / (bureau_agg["bureau_credit_sum"] + 1)
        )
        bureau_agg["bureau_overdue_credit_ratio"] = (
            bureau_agg["bureau_overdue_sum"] / (bureau_agg["bureau_credit_sum"] + 1)
        )
        bureau_agg["bureau_avg_loan_age"] = (
            bureau_agg["bureau_days_credit_mean"].abs()
        )

        print(f"  ✅ Bureau features: {bureau_agg.shape[1] - 1} features for "
              f"{bureau_agg['SK_ID_CURR'].nunique():,} customers")
        return bureau_agg


# ─────────────────────────────────────────────────────────────────────────────
# 3. PREVIOUS APPLICATION AGGREGATOR
# ─────────────────────────────────────────────────────────────────────────────

class PrevApplicationAggregator:
    """
    Aggregates previous_application.csv per SK_ID_CURR.
    Captures historical loan behavior: approval rates, goods categories, etc.
    """

    def fit_transform(self, prev_app: pd.DataFrame) -> pd.DataFrame:
        # Fix anomaly: 365243 in days columns
        days_cols = ["DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION",
                     "DAYS_LAST_DUE", "DAYS_TERMINATION"]
        for col in days_cols:
            if col in prev_app.columns:
                prev_app[col] = prev_app[col].replace(365243, np.nan)

        # ── 3a. Credit-to-application ratio ──────────────────────────────────
        prev_app["PREV_CREDIT_APPLICATION_RATIO"] = (
            prev_app["AMT_CREDIT"] / (prev_app["AMT_APPLICATION"] + 1)
        )
        prev_app["PREV_CREDIT_GOODS_RATIO"] = (
            prev_app["AMT_CREDIT"] / (prev_app["AMT_GOODS_PRICE"] + 1)
        )

        # ── 3b. Aggregate ─────────────────────────────────────────────────────
        prev_agg = prev_app.groupby("SK_ID_CURR").agg(
            prev_total_applications  = ("SK_ID_PREV", "count"),
            prev_approved_count      = ("NAME_CONTRACT_STATUS", lambda x: (x == "Approved").sum()),
            prev_refused_count       = ("NAME_CONTRACT_STATUS", lambda x: (x == "Refused").sum()),
            prev_canceled_count      = ("NAME_CONTRACT_STATUS", lambda x: (x == "Canceled").sum()),
            prev_unused_count        = ("NAME_CONTRACT_STATUS", lambda x: (x == "Unused offer").sum()),

            prev_amt_credit_mean     = ("AMT_CREDIT", "mean"),
            prev_amt_credit_max      = ("AMT_CREDIT", "max"),
            prev_amt_credit_sum      = ("AMT_CREDIT", "sum"),
            prev_amt_annuity_mean    = ("AMT_ANNUITY", "mean"),
            prev_amt_annuity_max     = ("AMT_ANNUITY", "max"),
            prev_amt_application_mean = ("AMT_APPLICATION", "mean"),
            prev_amt_goods_mean      = ("AMT_GOODS_PRICE", "mean"),

            prev_days_decision_mean  = ("DAYS_DECISION", "mean"),
            prev_days_decision_min   = ("DAYS_DECISION", "min"),
            prev_days_decision_max   = ("DAYS_DECISION", "max"),

            prev_hour_appr_max       = ("HOUR_APPR_PROCESS_START", "max"),
            prev_hour_appr_mean      = ("HOUR_APPR_PROCESS_START", "mean"),

            prev_down_payment_mean   = ("AMT_DOWN_PAYMENT", "mean"),
            prev_down_payment_sum    = ("AMT_DOWN_PAYMENT", "sum"),

            prev_rate_down_mean      = ("RATE_DOWN_PAYMENT", "mean"),
            prev_rate_interest_mean  = ("RATE_INTEREST_PRIMARY", "mean"),
            prev_rate_interest_max   = ("RATE_INTEREST_PRIMARY", "max"),

            prev_credit_app_ratio_mean = ("PREV_CREDIT_APPLICATION_RATIO", "mean"),
            prev_credit_goods_ratio_mean = ("PREV_CREDIT_GOODS_RATIO", "mean"),

            prev_sellerplace_area_mean = ("SELLERPLACE_AREA", "mean"),
            prev_sellerplace_area_max  = ("SELLERPLACE_AREA", "max"),

            prev_cnt_payment_mean    = ("CNT_PAYMENT", "mean"),
            prev_cnt_payment_sum     = ("CNT_PAYMENT", "sum"),
        ).reset_index()

        # ── 3c. Derived ratios ────────────────────────────────────────────────
        prev_agg["prev_approval_rate"] = (
            prev_agg["prev_approved_count"] / (prev_agg["prev_total_applications"] + 1)
        )
        prev_agg["prev_refusal_rate"] = (
            prev_agg["prev_refused_count"] / (prev_agg["prev_total_applications"] + 1)
        )
        prev_agg["prev_annuity_credit_ratio"] = (
            prev_agg["prev_amt_annuity_mean"] / (prev_agg["prev_amt_credit_mean"] + 1)
        )

        print(f"  ✅ Prev Application features: {prev_agg.shape[1] - 1} features for "
              f"{prev_agg['SK_ID_CURR'].nunique():,} customers")
        return prev_agg


# ─────────────────────────────────────────────────────────────────────────────
# 4. POS CASH AGGREGATOR
# ─────────────────────────────────────────────────────────────────────────────

class POSCashAggregator:
    """
    Aggregates POS_CASH_balance.csv per SK_ID_CURR.
    Captures DPD and installment completion patterns.
    """

    def fit_transform(self, pos_cash: pd.DataFrame) -> pd.DataFrame:
        pos_agg = pos_cash.groupby("SK_ID_CURR").agg(
            pos_total_records        = ("SK_ID_PREV", "count"),
            pos_months_balance_mean  = ("MONTHS_BALANCE", "mean"),
            pos_months_balance_min   = ("MONTHS_BALANCE", "min"),
            pos_cnt_instalment_mean  = ("CNT_INSTALMENT", "mean"),
            pos_cnt_instalment_future_mean = ("CNT_INSTALMENT_FUTURE", "mean"),
            pos_sk_dpd_mean          = ("SK_DPD", "mean"),
            pos_sk_dpd_max           = ("SK_DPD", "max"),
            pos_sk_dpd_sum           = ("SK_DPD", "sum"),
            pos_sk_dpd_def_mean      = ("SK_DPD_DEF", "mean"),
            pos_sk_dpd_def_max       = ("SK_DPD_DEF", "max"),
            pos_sk_dpd_def_sum       = ("SK_DPD_DEF", "sum"),
            pos_completed_count      = ("NAME_CONTRACT_STATUS", lambda x: (x == "Completed").sum()),
            pos_active_count         = ("NAME_CONTRACT_STATUS", lambda x: (x == "Active").sum()),
        ).reset_index()

        pos_agg["pos_dpd_ever_flag"] = (pos_agg["pos_sk_dpd_max"] > 0).astype(np.int8)
        pos_agg["pos_completion_rate"] = (
            pos_agg["pos_completed_count"] / (pos_agg["pos_total_records"] + 1)
        )
        pos_agg["pos_future_instalment_ratio"] = (
            pos_agg["pos_cnt_instalment_future_mean"] / (pos_agg["pos_cnt_instalment_mean"] + 1)
        )

        print(f"  ✅ POS Cash features: {pos_agg.shape[1] - 1} features for "
              f"{pos_agg['SK_ID_CURR'].nunique():,} customers")
        return pos_agg


# ─────────────────────────────────────────────────────────────────────────────
# 5. CREDIT CARD AGGREGATOR
# ─────────────────────────────────────────────────────────────────────────────

class CreditCardAggregator:
    """
    Aggregates credit_card_balance.csv per SK_ID_CURR.
    Captures revolving credit utilization and payment behavior.
    """

    def fit_transform(self, credit_card: pd.DataFrame) -> pd.DataFrame:
        # Utilization ratio
        credit_card["CC_UTILIZATION"] = (
            credit_card["AMT_BALANCE"] / (credit_card["AMT_CREDIT_LIMIT_ACTUAL"] + 1)
        ).clip(0, 5)  # Cap outliers

        # Overspending flag
        credit_card["CC_OVER_LIMIT"] = (
            credit_card["AMT_BALANCE"] > credit_card["AMT_CREDIT_LIMIT_ACTUAL"]
        ).astype(np.int8)

        # Payment ratio
        credit_card["CC_PAYMENT_RATIO"] = (
            credit_card["AMT_PAYMENT_TOTAL_CURRENT"] /
            (credit_card["AMT_INST_MIN_REGULARITY"] + 1)
        ).clip(0, 20)

        cc_agg = credit_card.groupby("SK_ID_CURR").agg(
            cc_total_records          = ("SK_ID_PREV", "count"),
            cc_months_balance_mean    = ("MONTHS_BALANCE", "mean"),
            cc_months_balance_min     = ("MONTHS_BALANCE", "min"),
            cc_amt_balance_mean       = ("AMT_BALANCE", "mean"),
            cc_amt_balance_max        = ("AMT_BALANCE", "max"),
            cc_amt_credit_limit_mean  = ("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
            cc_amt_credit_limit_max   = ("AMT_CREDIT_LIMIT_ACTUAL", "max"),
            cc_amt_drawings_total_mean = ("AMT_DRAWINGS_CURRENT", "mean"),
            cc_amt_drawings_total_sum  = ("AMT_DRAWINGS_CURRENT", "sum"),
            cc_amt_payment_mean       = ("AMT_PAYMENT_TOTAL_CURRENT", "mean"),
            cc_amt_receivable_mean    = ("AMT_RECEIVABLE_PRINCIPAL", "mean"),
            cc_cnt_drawings_mean      = ("CNT_DRAWINGS_CURRENT", "mean"),
            cc_sk_dpd_mean            = ("SK_DPD", "mean"),
            cc_sk_dpd_max             = ("SK_DPD", "max"),
            cc_sk_dpd_def_mean        = ("SK_DPD_DEF", "mean"),
            cc_utilization_mean       = ("CC_UTILIZATION", "mean"),
            cc_utilization_max        = ("CC_UTILIZATION", "max"),
            cc_over_limit_sum         = ("CC_OVER_LIMIT", "sum"),
            cc_over_limit_mean        = ("CC_OVER_LIMIT", "mean"),
            cc_payment_ratio_mean     = ("CC_PAYMENT_RATIO", "mean"),
            cc_payment_ratio_min      = ("CC_PAYMENT_RATIO", "min"),
        ).reset_index()

        cc_agg["cc_dpd_ever_flag"]   = (cc_agg["cc_sk_dpd_max"] > 0).astype(np.int8)
        cc_agg["cc_balance_limit_ratio"] = (
            cc_agg["cc_amt_balance_mean"] / (cc_agg["cc_amt_credit_limit_mean"] + 1)
        )

        print(f"  ✅ Credit Card features: {cc_agg.shape[1] - 1} features for "
              f"{cc_agg['SK_ID_CURR'].nunique():,} customers")
        return cc_agg


# ─────────────────────────────────────────────────────────────────────────────
# 6. INSTALLMENTS AGGREGATOR
# ─────────────────────────────────────────────────────────────────────────────

class InstallmentAggregator:
    """
    Aggregates installments_payments.csv per SK_ID_CURR.
    Most informative features: late payments, underpayment, payment timing.
    """

    def fit_transform(self, installments: pd.DataFrame) -> pd.DataFrame:
        # ── 6a. Derived columns ───────────────────────────────────────────────
        installments["PAYMENT_DIFF"]      = installments["AMT_PAYMENT"] - installments["AMT_INSTALMENT"]
        installments["PAYMENT_RATIO"]     = (
            installments["AMT_PAYMENT"] / (installments["AMT_INSTALMENT"] + 1e-6)
        ).clip(0, 5)
        installments["DAYS_LATE"]         = np.maximum(
            0,
            installments["DAYS_ENTRY_PAYMENT"] - installments["DAYS_INSTALMENT"]
        )
        installments["DAYS_EARLY"]        = np.maximum(
            0,
            installments["DAYS_INSTALMENT"] - installments["DAYS_ENTRY_PAYMENT"]
        )
        installments["LATE_FLAG"]         = (installments["DAYS_LATE"] > 0).astype(np.int8)
        installments["UNDERPAID_FLAG"]    = (installments["PAYMENT_DIFF"] < 0).astype(np.int8)
        installments["OVERPAID_FLAG"]     = (installments["PAYMENT_DIFF"] > 0).astype(np.int8)

        inst_agg = installments.groupby("SK_ID_CURR").agg(
            inst_total_payments       = ("SK_ID_PREV", "count"),
            inst_payment_ratio_mean   = ("PAYMENT_RATIO", "mean"),
            inst_payment_ratio_min    = ("PAYMENT_RATIO", "min"),
            inst_payment_ratio_std    = ("PAYMENT_RATIO", "std"),
            inst_payment_diff_mean    = ("PAYMENT_DIFF", "mean"),
            inst_payment_diff_min     = ("PAYMENT_DIFF", "min"),
            inst_payment_diff_sum     = ("PAYMENT_DIFF", "sum"),
            inst_days_late_mean       = ("DAYS_LATE", "mean"),
            inst_days_late_max        = ("DAYS_LATE", "max"),
            inst_days_late_sum        = ("DAYS_LATE", "sum"),
            inst_days_early_mean      = ("DAYS_EARLY", "mean"),
            inst_days_early_max       = ("DAYS_EARLY", "max"),
            inst_late_flag_mean       = ("LATE_FLAG", "mean"),
            inst_late_flag_sum        = ("LATE_FLAG", "sum"),
            inst_underpaid_mean       = ("UNDERPAID_FLAG", "mean"),
            inst_underpaid_sum        = ("UNDERPAID_FLAG", "sum"),
            inst_overpaid_mean        = ("OVERPAID_FLAG", "mean"),
            inst_num_unique_loans     = ("SK_ID_PREV", "nunique"),
            inst_amt_payment_mean     = ("AMT_PAYMENT", "mean"),
            inst_amt_payment_sum      = ("AMT_PAYMENT", "sum"),
            inst_amt_instalment_mean  = ("AMT_INSTALMENT", "mean"),
        ).reset_index()

        # Derived
        inst_agg["inst_ever_late"]     = (inst_agg["inst_days_late_max"] > 0).astype(np.int8)
        inst_agg["inst_total_paid"]    = inst_agg["inst_amt_payment_sum"]
        inst_agg["inst_late_rate"]     = (
            inst_agg["inst_late_flag_sum"] / (inst_agg["inst_total_payments"] + 1)
        )

        print(f"  ✅ Installment features: {inst_agg.shape[1] - 1} features for "
              f"{inst_agg['SK_ID_CURR'].nunique():,} customers")
        return inst_agg


# ─────────────────────────────────────────────────────────────────────────────
# 7. WEIGHT OF EVIDENCE (WoE) ENCODER
# ─────────────────────────────────────────────────────────────────────────────

class WoEEncoder(BaseEstimator, TransformerMixin):
    """
    Weight of Evidence encoding for categorical features.
    WoE = ln(Distribution of Events / Distribution of Non-Events)
    IV (Information Value) used to rank feature importance.

    Only fit on training data. Test data gets transform().
    Categories unseen in training → WoE = 0.
    """

    def __init__(self, cols=None, smoothing=0.5):
        self.cols = cols
        self.smoothing = smoothing
        self.woe_maps = {}
        self.iv_scores = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoEEncoder":
        cols = self.cols or X.select_dtypes(include=["object", "category"]).columns.tolist()

        total_events     = y.sum()
        total_nonevents  = len(y) - total_events

        for col in cols:
            temp = pd.concat([X[[col]], y.rename("TARGET")], axis=1)
            temp[col] = temp[col].astype(str).fillna("__MISSING__")

            grp = temp.groupby(col)["TARGET"].agg(["sum", "count"]).reset_index()
            grp.columns = [col, "events", "total"]
            grp["nonevents"] = grp["total"] - grp["events"]

            # Laplace smoothing to avoid log(0)
            grp["dist_events"]    = (grp["events"] + self.smoothing) / (total_events + self.smoothing * len(grp))
            grp["dist_nonevents"] = (grp["nonevents"] + self.smoothing) / (total_nonevents + self.smoothing * len(grp))

            grp["woe"] = np.log(grp["dist_events"] / grp["dist_nonevents"])
            grp["iv"]  = (grp["dist_events"] - grp["dist_nonevents"]) * grp["woe"]

            self.woe_maps[col]  = grp.set_index(col)["woe"].to_dict()
            self.iv_scores[col] = grp["iv"].sum()

        # Sort IV scores
        self.iv_scores = dict(sorted(self.iv_scores.items(), key=lambda x: x[1], reverse=True))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col, woe_map in self.woe_maps.items():
            if col in df.columns:
                df[f"{col}_WOE"] = (
                    df[col].astype(str).fillna("__MISSING__").map(woe_map).fillna(0.0)
                )
        return df

    def print_iv_table(self):
        print("\n📊 Information Value (IV) Summary:")
        print(f"  {'Feature':<35} {'IV':>8}  {'Predictive Power'}")
        print("  " + "-" * 65)
        for col, iv in self.iv_scores.items():
            if iv < 0.02:     power = "Unpredictive"
            elif iv < 0.10:   power = "Weak"
            elif iv < 0.30:   power = "Medium"
            elif iv < 0.50:   power = "Strong"
            else:              power = "Very Strong / Suspicious"
            print(f"  {col:<35} {iv:>8.4f}  {power}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. MASTER PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class CreditRiskFeaturePipeline:
    """
    Orchestrates all feature engineering steps.
    Returns a single merged DataFrame ready for model training.

    Usage:
        pipeline = CreditRiskFeaturePipeline()
        X_train, X_test, y_train = pipeline.run(
            app_train, app_test,
            bureau, bureau_balance, prev_app,
            pos_cash, credit_card, installments,
            fit_woe=True
        )
    """

    def __init__(self, woe_cols=None):
        self.app_fe       = ApplicationFeatureEngineer()
        self.bureau_agg   = BureauAggregator()
        self.prev_agg     = PrevApplicationAggregator()
        self.pos_agg      = POSCashAggregator()
        self.cc_agg       = CreditCardAggregator()
        self.inst_agg     = InstallmentAggregator()
        self.woe_encoder  = WoEEncoder(cols=woe_cols)
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy="median")
        self._fitted = False

    def _merge_all(
        self, app_df, bureau_feats, prev_feats,
        pos_feats, cc_feats, inst_feats
    ) -> pd.DataFrame:
        df = app_df.copy()
        for feats in [bureau_feats, prev_feats, pos_feats, cc_feats, inst_feats]:
            df = df.merge(feats, on="SK_ID_CURR", how="left")
        return df

    def run(
        self,
        app_train: pd.DataFrame,
        app_test: pd.DataFrame,
        bureau: pd.DataFrame,
        bureau_balance: pd.DataFrame,
        prev_app: pd.DataFrame,
        pos_cash: pd.DataFrame,
        credit_card: pd.DataFrame,
        installments: pd.DataFrame,
        fit_woe: bool = True,
    ):
        print("\n🚀 CreditRiskFeaturePipeline — START")
        print("=" * 60)

        y_train = app_train["TARGET"].copy()
        app_train = app_train.drop(columns=["TARGET"])

        # ── Step 1: Application FE ────────────────────────────────────────────
        print("\n[1/7] Application-level feature engineering...")
        app_train_fe = self.app_fe.transform(app_train)
        app_test_fe  = self.app_fe.transform(app_test)
        print(f"  Shape: {app_train_fe.shape}")

        # ── Step 2: Bureau ────────────────────────────────────────────────────
        print("\n[2/7] Bureau aggregations...")
        bureau_feats = self.bureau_agg.fit_transform(bureau, bureau_balance)
        del bureau, bureau_balance; gc.collect()

        # ── Step 3: Previous Applications ────────────────────────────────────
        print("\n[3/7] Previous application aggregations...")
        prev_feats = self.prev_agg.fit_transform(prev_app)
        del prev_app; gc.collect()

        # ── Step 4: POS Cash ──────────────────────────────────────────────────
        print("\n[4/7] POS Cash aggregations...")
        pos_feats = self.pos_agg.fit_transform(pos_cash)
        del pos_cash; gc.collect()

        # ── Step 5: Credit Card ───────────────────────────────────────────────
        print("\n[5/7] Credit card aggregations...")
        cc_feats = self.cc_agg.fit_transform(credit_card)
        del credit_card; gc.collect()

        # ── Step 6: Installments ──────────────────────────────────────────────
        print("\n[6/7] Installment aggregations...")
        inst_feats = self.inst_agg.fit_transform(installments)
        del installments; gc.collect()

        # ── Step 7: Merge everything ──────────────────────────────────────────
        print("\n[7/7] Merging all feature sets...")
        X_train = self._merge_all(app_train_fe, bureau_feats, prev_feats, pos_feats, cc_feats, inst_feats)
        X_test  = self._merge_all(app_test_fe,  bureau_feats, prev_feats, pos_feats, cc_feats, inst_feats)
        del bureau_feats, prev_feats, pos_feats, cc_feats, inst_feats; gc.collect()

        print(f"  X_train: {X_train.shape}")
        print(f"  X_test : {X_test.shape}")

        # ── Step 8: Categorical encoding ──────────────────────────────────────
        print("\n[8/8] Encoding categoricals...")
        cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        print(f"  Categorical columns: {len(cat_cols)}")

        # WoE encoding on training
        if fit_woe:
            self.woe_encoder.cols = cat_cols
            self.woe_encoder.fit(X_train, y_train)
            self.woe_encoder.print_iv_table()

        X_train = self.woe_encoder.transform(X_train)
        X_test  = self.woe_encoder.transform(X_test)

        # Drop original cat cols (WoE replacements are suffixed _WOE)
        X_train = X_train.drop(columns=cat_cols, errors="ignore")
        X_test  = X_test.drop(columns=cat_cols, errors="ignore")

        # Drop identifier columns
        drop_cols = ["SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV"]
        X_train = X_train.drop(columns=[c for c in drop_cols if c in X_train.columns])
        X_test  = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])

        print(f"\n  ✅ Final X_train shape: {X_train.shape}")
        print(f"  ✅ Final X_test  shape: {X_test.shape}")
        print(f"  ✅ Total features      : {X_train.shape[1]}")
        print("\n🎯 Feature pipeline complete!")
        self._fitted = True

        return X_train, X_test, y_train

    def save(self, path: str = "models/feature_pipeline.pkl"):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  Pipeline saved to {path}")

    @staticmethod
    def load(path: str = "models/feature_pipeline.pkl") -> "CreditRiskFeaturePipeline":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT — run as script for quick validation
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    DATA_DIR = "data/raw/"

    if not os.path.exists(f"{DATA_DIR}application_train.csv"):
        print("⚠️  Download dataset from:")
        print("   https://www.kaggle.com/c/home-credit-default-risk/data")
        print("   Place CSV files in data/raw/")
    else:
        print("Loading datasets...")
        app_train      = pd.read_csv(f"{DATA_DIR}application_train.csv")
        app_test       = pd.read_csv(f"{DATA_DIR}application_test.csv")
        bureau         = pd.read_csv(f"{DATA_DIR}bureau.csv")
        bureau_balance = pd.read_csv(f"{DATA_DIR}bureau_balance.csv")
        prev_app       = pd.read_csv(f"{DATA_DIR}previous_application.csv")
        pos_cash       = pd.read_csv(f"{DATA_DIR}POS_CASH_balance.csv")
        credit_card    = pd.read_csv(f"{DATA_DIR}credit_card_balance.csv")
        installments   = pd.read_csv(f"{DATA_DIR}installments_payments.csv")

        pipeline = CreditRiskFeaturePipeline()
        X_train, X_test, y_train = pipeline.run(
            app_train, app_test, bureau, bureau_balance,
            prev_app, pos_cash, credit_card, installments,
            fit_woe=True,
        )

        print("\nSaving datasets...")
        X_train.assign(TARGET=y_train).to_parquet("data/processed/train_features.parquet", index=False)
        X_test.to_parquet("data/processed/test_features.parquet", index=False)
        pipeline.save("models/feature_pipeline.pkl")
        print("✅ Done!")
