"""
Evaluation Utilities — Credit Risk Intelligence Platform
Agent 2 | Vaidik Sharma | github.com/Vaidik6920

Covers:
  - Full metric suite (AUC-ROC, AUC-PR, Brier, Gini, KS stat)
  - Threshold analysis (optimal cutoff by F1 / Youden / business cost)
  - Probability calibration (Platt scaling, isotonic)
  - Business metrics (approval rate, bad rate, expected loss)
  - Population stability index (PSI)
  - Model comparison table
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    log_loss, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, f1_score,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

PLOTS = Path("data/plots")
PLOTS.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# CORE METRICS
# ─────────────────────────────────────────────────────────────────────────────

def gini_coefficient(y_true, y_prob):
    """Gini = 2 * AUC - 1. Standard metric in credit risk."""
    return 2 * roc_auc_score(y_true, y_prob) - 1


def ks_statistic(y_true, y_prob):
    """Kolmogorov-Smirnov statistic — max separation between CDF of goods/bads."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return (tpr - fpr).max()


def full_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_pos = y_true.sum()
    total_neg = len(y_true) - total_pos

    return {
        "auc_roc"     : round(roc_auc_score(y_true, y_prob), 5),
        "gini"        : round(gini_coefficient(y_true, y_prob), 5),
        "ks_stat"     : round(ks_statistic(y_true, y_prob), 5),
        "auc_pr"      : round(average_precision_score(y_true, y_prob), 5),
        "brier"       : round(brier_score_loss(y_true, y_prob), 5),
        "log_loss"    : round(log_loss(y_true, y_prob), 5),
        "precision"   : round(tp / (tp + fp + 1e-9), 5),
        "recall"      : round(tp / (tp + fn + 1e-9), 5),
        "specificity" : round(tn / (tn + fp + 1e-9), 5),
        "f1"          : round(2 * tp / (2 * tp + fp + fn + 1e-9), 5),
        "mcc"         : round((tp * tn - fp * fn) /
                              (((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5 + 1e-9), 5),
        "approval_rate": round((tn + fn) / len(y_true), 5),  # fraction classified as non-default
        "bad_rate_approved": round(fn / (tn + fn + 1e-9), 5),  # defaults in approved bucket
        "threshold"   : threshold,
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }


def print_metrics(metrics: dict, title: str = "Model Metrics"):
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")
    print(f"  AUC-ROC      : {metrics['auc_roc']:.4f}")
    print(f"  Gini         : {metrics['gini']:.4f}   (= 2·AUC - 1)")
    print(f"  KS Statistic : {metrics['ks_stat']:.4f}")
    print(f"  AUC-PR       : {metrics['auc_pr']:.4f}")
    print(f"  Brier Score  : {metrics['brier']:.4f}   (lower = better)")
    print(f"  Log Loss     : {metrics['log_loss']:.4f}")
    print(f"  ---")
    print(f"  Precision    : {metrics['precision']:.4f}  @ threshold {metrics['threshold']:.2f}")
    print(f"  Recall       : {metrics['recall']:.4f}")
    print(f"  Specificity  : {metrics['specificity']:.4f}")
    print(f"  F1           : {metrics['f1']:.4f}")
    print(f"  MCC          : {metrics['mcc']:.4f}")
    print(f"  ---")
    print(f"  Approval Rate      : {metrics['approval_rate']*100:.1f}%")
    print(f"  Bad Rate (approved): {metrics['bad_rate_approved']*100:.2f}%")
    print(f"  Confusion Matrix:")
    print(f"           Pred 0   Pred 1")
    print(f"  True 0:  {metrics['TN']:>7,}  {metrics['FP']:>7,}")
    print(f"  True 1:  {metrics['FN']:>7,}  {metrics['TP']:>7,}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def threshold_analysis(y_true: np.ndarray, y_prob: np.ndarray,
                       cost_fp: float = 1.0, cost_fn: float = 5.0) -> pd.DataFrame:
    """
    Sweep thresholds from 0.01 to 0.99.
    Reports F1, Youden J, precision, recall, business cost, approval rate.

    cost_fn >> cost_fp typical in credit: missing a bad borrower is ~5x worse
    than rejecting a good one.
    """
    thresholds = np.arange(0.01, 0.99, 0.01)
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision  = tp / (tp + fp + 1e-9)
        recall     = tp / (tp + fn + 1e-9)
        specificity= tn / (tn + fp + 1e-9)
        f1         = 2 * tp / (2 * tp + fp + fn + 1e-9)
        youden     = recall + specificity - 1
        biz_cost   = cost_fp * fp + cost_fn * fn
        approval   = (tn + fn) / len(y_true)
        bad_appr   = fn / (tn + fn + 1e-9)
        rows.append({
            "threshold"   : round(t, 2),
            "precision"   : round(precision, 4),
            "recall"      : round(recall, 4),
            "specificity" : round(specificity, 4),
            "f1"          : round(f1, 4),
            "youden_j"    : round(youden, 4),
            "biz_cost"    : round(biz_cost, 1),
            "approval_rate": round(approval, 4),
            "bad_rate_approved": round(bad_appr, 4),
            "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        })

    df = pd.DataFrame(rows)
    opt_f1      = df.loc[df["f1"].idxmax()]
    opt_youden  = df.loc[df["youden_j"].idxmax()]
    opt_cost    = df.loc[df["biz_cost"].idxmin()]

    print("\n📌 Optimal Thresholds:")
    print(f"  By F1          : {opt_f1['threshold']:.2f}  "
          f"F1={opt_f1['f1']:.4f}  P={opt_f1['precision']:.4f}  R={opt_f1['recall']:.4f}")
    print(f"  By Youden J    : {opt_youden['threshold']:.2f}  "
          f"J={opt_youden['youden_j']:.4f}  Spec={opt_youden['specificity']:.4f}")
    print(f"  By Biz Cost    : {opt_cost['threshold']:.2f}  "
          f"Cost={opt_cost['biz_cost']:.0f}  "
          f"ApprovalRate={opt_cost['approval_rate']*100:.1f}%")

    return df, opt_f1["threshold"], opt_youden["threshold"], opt_cost["threshold"]


# ─────────────────────────────────────────────────────────────────────────────
# PROBABILITY CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_model(model, X_train, y_train, method="isotonic"):
    """
    Calibrate model probabilities using Platt scaling ('sigmoid') or
    isotonic regression ('isotonic').
    Returns calibrated model ready for .predict_proba().
    """
    calibrated = CalibratedClassifierCV(model, method=method, cv="prefit")
    calibrated.fit(X_train, y_train)
    return calibrated


def plot_calibration_curve(y_true, y_prob_raw, y_prob_cal=None, model_name="Model"):
    """Plot reliability diagram — predicted probabilities vs actual default rates."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Calibration curve (reliability diagram) ───────────────────────────────
    ax = axes[0]
    frac_pos_raw, mean_pred_raw = calibration_curve(y_true, y_prob_raw, n_bins=15, strategy="quantile")
    ax.plot(mean_pred_raw, frac_pos_raw, "s-", color="#2563EB", label="Uncalibrated", lw=2, ms=6)
    if y_prob_cal is not None:
        frac_pos_cal, mean_pred_cal = calibration_curve(y_true, y_prob_cal, n_bins=15, strategy="quantile")
        ax.plot(mean_pred_cal, frac_pos_cal, "o-", color="#16A34A", label="Calibrated", lw=2, ms=6)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Calibration Curve — {model_name}"); ax.legend()

    # ── Score distribution ────────────────────────────────────────────────────
    ax = axes[1]
    for tgt, color, label in zip([0, 1], ["#2563EB", "#DC2626"], ["Non-Default", "Default"]):
        ax.hist(y_prob_raw[y_true == tgt], bins=50, alpha=0.5, color=color, label=label, density=True)
    ax.set_xlabel("Predicted Default Probability")
    ax.set_ylabel("Density")
    ax.set_title(f"Score Distribution by Class — {model_name}")
    ax.legend()

    plt.tight_layout()
    path = PLOTS / f"calibration_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# FULL EVALUATION REPORT
# ─────────────────────────────────────────────────────────────────────────────

def full_evaluation_report(y_true, y_prob, model_name="Model", save_prefix=None):
    """
    Comprehensive evaluation: metrics + threshold sweep + ROC + PR + calibration.
    Returns metrics dict and optimal threshold (by Youden J).
    """
    print(f"\n{'#' * 60}")
    print(f"# EVALUATION REPORT — {model_name}")
    print(f"{'#' * 60}")

    # Metrics at default 0.5
    m = full_metrics(y_true, y_prob)
    print_metrics(m, title=f"{model_name} @ threshold=0.50")

    # Threshold sweep
    thr_df, t_f1, t_youden, t_cost = threshold_analysis(y_true, y_prob)
    thr_df.to_csv(PLOTS / f"threshold_analysis_{save_prefix or model_name}.csv", index=False)

    # Metrics at optimal threshold
    m_opt = full_metrics(y_true, y_prob, threshold=t_youden)
    print_metrics(m_opt, title=f"{model_name} @ Youden optimal threshold={t_youden:.2f}")

    # ── Dashboard figure ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35)

    # ROC
    ax_roc = fig.add_subplot(gs[0, :2])
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ax_roc.plot(fpr, tpr, lw=2, color="#2563EB", label=f"AUC={m['auc_roc']:.4f}")
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1); ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR")
    ax_roc.set_title("ROC Curve"); ax_roc.legend(loc="lower right")

    # Precision-Recall
    ax_pr = fig.add_subplot(gs[0, 2:])
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ax_pr.plot(rec, prec, lw=2, color="#DC2626", label=f"AP={m['auc_pr']:.4f}")
    ax_pr.axhline(y_true.mean(), linestyle="--", color="gray", lw=1, label=f"Baseline {y_true.mean():.3f}")
    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision"); ax_pr.set_title("Precision-Recall Curve")
    ax_pr.legend()

    # Threshold F1
    ax_f1 = fig.add_subplot(gs[1, 0])
    ax_f1.plot(thr_df["threshold"], thr_df["f1"], color="#16A34A", lw=2)
    ax_f1.axvline(t_f1, color="red", linestyle="--", lw=1, label=f"Opt={t_f1:.2f}")
    ax_f1.set_title("F1 vs Threshold"); ax_f1.set_xlabel("Threshold"); ax_f1.legend()

    # Threshold Precision/Recall
    ax_pr2 = fig.add_subplot(gs[1, 1])
    ax_pr2.plot(thr_df["threshold"], thr_df["precision"], label="Precision", color="#2563EB", lw=2)
    ax_pr2.plot(thr_df["threshold"], thr_df["recall"], label="Recall", color="#DC2626", lw=2)
    ax_pr2.axvline(t_youden, color="gray", linestyle="--", lw=1, label=f"Youden={t_youden:.2f}")
    ax_pr2.set_title("Precision / Recall vs Threshold")
    ax_pr2.set_xlabel("Threshold"); ax_pr2.legend(fontsize=8)

    # Approval rate vs bad rate
    ax_biz = fig.add_subplot(gs[1, 2])
    ax_biz.plot(thr_df["threshold"], thr_df["approval_rate"] * 100, color="#7C3AED", lw=2, label="Approval%")
    ax_biz2 = ax_biz.twinx()
    ax_biz2.plot(thr_df["threshold"], thr_df["bad_rate_approved"] * 100, color="#D97706", lw=2, label="Bad rate%")
    ax_biz.set_xlabel("Threshold"); ax_biz.set_ylabel("Approval Rate %", color="#7C3AED")
    ax_biz2.set_ylabel("Bad Rate % in Approved", color="#D97706")
    ax_biz.set_title("Business Metrics vs Threshold")

    # Score distribution
    ax_dist = fig.add_subplot(gs[1, 3])
    for tgt, color, label in zip([0, 1], ["#2563EB", "#DC2626"], ["Non-Default", "Default"]):
        ax_dist.hist(y_prob[y_true == tgt], bins=40, alpha=0.5, color=color, label=label, density=True)
    ax_dist.axvline(t_youden, color="black", linestyle="--", lw=1, label=f"Cut={t_youden:.2f}")
    ax_dist.set_title("Score Distribution by Class"); ax_dist.set_xlabel("Predicted Probability")
    ax_dist.legend(fontsize=8)

    fig.suptitle(f"Evaluation Dashboard — {model_name}", fontsize=14, fontweight="bold", y=1.01)
    path = PLOTS / f"eval_dashboard_{save_prefix or model_name.lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\n  📊 Evaluation dashboard saved: {path}")

    return m_opt, t_youden


# ─────────────────────────────────────────────────────────────────────────────
# POPULATION STABILITY INDEX (PSI)
# ─────────────────────────────────────────────────────────────────────────────

def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    PSI measures distribution shift between training and production score distributions.
    PSI < 0.10: No significant change
    PSI 0.10-0.25: Moderate change — monitor
    PSI > 0.25: Major change — retrain
    """
    def psi_bucket(e, a):
        e = np.clip(e, 1e-4, None)
        a = np.clip(a, 1e-4, None)
        return (e - a) * np.log(e / a)

    breakpoints = np.quantile(expected, np.linspace(0, 1, buckets + 1))
    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct   = np.histogram(actual,   bins=breakpoints)[0] / len(actual)
    psi          = sum(psi_bucket(e, a) for e, a in zip(expected_pct, actual_pct))

    if psi < 0.10:
        status = "✅ STABLE"
    elif psi < 0.25:
        status = "⚠️  MODERATE — monitor"
    else:
        status = "🚨 MAJOR SHIFT — retrain"

    print(f"\n  PSI = {psi:.4f}  →  {status}")
    return psi


# ─────────────────────────────────────────────────────────────────────────────
# MODEL COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

def model_comparison_table(results: dict) -> pd.DataFrame:
    """
    results = {
        "Logistic Baseline": (y_true, y_prob),
        "XGBoost"          : (y_true, y_prob),
        ...
    }
    Returns comparison DataFrame sorted by AUC-ROC.
    """
    rows = []
    for name, (y_true, y_prob) in results.items():
        m = full_metrics(y_true, y_prob)
        rows.append({
            "Model"        : name,
            "AUC-ROC"      : m["auc_roc"],
            "Gini"         : m["gini"],
            "KS Stat"      : m["ks_stat"],
            "AUC-PR"       : m["auc_pr"],
            "Brier"        : m["brier"],
            "F1 (@0.5)"    : m["f1"],
            "Precision"    : m["precision"],
            "Recall"       : m["recall"],
        })

    df = pd.DataFrame(rows).sort_values("AUC-ROC", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 90)
    print("  MODEL COMPARISON TABLE")
    print("=" * 90)
    print(df.to_string(index=False, float_format="{:.4f}".format))

    baseline_auc = df[df["Model"].str.contains("Baseline")]["AUC-ROC"].values[0]
    best_auc     = df["AUC-ROC"].iloc[0]
    print(f"\n  Improvement over baseline : +{(best_auc - baseline_auc)*100:.2f} pp")

    return df
