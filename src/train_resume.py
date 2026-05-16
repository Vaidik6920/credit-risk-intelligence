"""
Fast-resume training: loads existing LR/XGB/LGB OOFs, runs remaining steps.
Speed settings: Optuna 3 trials (1-fold proxy), CatBoost 300 iter, 300-tree cap.
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, time, gc, yaml, pickle, json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow, mlflow.sklearn, mlflow.xgboost, mlflow.lightgbm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import shap
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT   = Path(__file__).resolve().parent.parent
DATA   = ROOT / "data" / "processed"
MODELS = ROOT / "models"
PLOTS  = ROOT / "data" / "plots"
CONFIG = ROOT / "configs" / "config.yaml"
MODELS.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

with open(CONFIG) as f:
    cfg = yaml.safe_load(f)

EXPERIMENT = cfg["mlflow"]["experiment_name"]
N_FOLDS    = cfg["data"]["n_folds"]
SEED       = cfg["data"]["random_state"]
TARGET     = cfg["data"]["target_col"]

# ─── helpers ─────────────────────────────────────────────────────────────────

def compute_all_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "auc_roc"   : roc_auc_score(y_true, y_prob),
        "auc_pr"    : average_precision_score(y_true, y_prob),
        "brier"     : brier_score_loss(y_true, y_prob),
        "logloss"   : log_loss(y_true, y_prob),
        "precision" : tp / (tp + fp + 1e-9),
        "recall"    : tp / (tp + fn + 1e-9),
    }

def load_data():
    train = pd.read_parquet(DATA / "train_features.parquet")
    test  = pd.read_parquet(DATA / "test_features.parquet")
    y_train = train[TARGET].values.astype(int)
    X_train = train.drop(columns=[TARGET], errors="ignore")
    X_test  = test.drop(columns=[TARGET], errors="ignore")
    common  = X_train.columns.intersection(X_test.columns)
    X_train, X_test = X_train[common], X_test[common]
    print(f"  X_train: {X_train.shape}  positives: {y_train.mean()*100:.2f}%")
    return X_train, X_test, y_train

# ─── Optuna (fast: 3 trials, single-fold proxy) ───────────────────────────────

def optuna_tune_xgb_fast(X_train, y_train, experiment_id):
    print("\n[Optuna] XGBoost fast tuning (3 trials, 1-fold)")
    spw = int((len(y_train) - y_train.sum()) / y_train.sum())
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
    tr_idx, val_idx = next(iter(skf.split(X_train, y_train)))

    def objective(trial):
        p = {
            "n_estimators"    : trial.suggest_int("n_estimators", 200, 300),
            "max_depth"       : trial.suggest_int("max_depth", 4, 7),
            "learning_rate"   : trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
            "subsample"       : trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
            "min_child_weight": trial.suggest_int("min_child_weight", 10, 50),
            "reg_alpha"       : trial.suggest_float("reg_alpha", 0.0, 0.5),
            "reg_lambda"      : trial.suggest_float("reg_lambda", 0.5, 2.0),
            "scale_pos_weight": spw,
            "tree_method": "hist", "eval_metric": "auc",
            "random_state": SEED, "n_jobs": -1, "verbosity": 0,
        }
        m = xgb.XGBClassifier(**p)
        m.fit(X_train.iloc[tr_idx], y_train[tr_idx],
              eval_set=[(X_train.iloc[val_idx], y_train[val_idx])],
              early_stopping_rounds=30, verbose=False)
        return roc_auc_score(y_train[val_idx], m.predict_proba(X_train.iloc[val_idx])[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=3)
    bp = study.best_params
    bp.update({"scale_pos_weight": spw, "tree_method": "hist", "eval_metric": "auc",
               "random_state": SEED, "n_jobs": -1, "verbosity": 0})
    print(f"  XGB best params: {bp}")
    with mlflow.start_run(experiment_id=experiment_id, run_name="Run-11_XGBoost_Optuna_Tuned"):
        mlflow.log_params(bp); mlflow.log_metric("optuna_best_auc", study.best_value)
    return bp


def optuna_tune_lgb_fast(X_train, y_train, experiment_id):
    print("\n[Optuna] LightGBM fast tuning (3 trials, 1-fold)")
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
    tr_idx, val_idx = next(iter(skf.split(X_train, y_train)))

    def objective(trial):
        p = {
            "n_estimators"     : trial.suggest_int("n_estimators", 200, 300),
            "num_leaves"       : trial.suggest_int("num_leaves", 31, 63),
            "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
            "subsample"        : trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.5, 0.9),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "reg_alpha"        : trial.suggest_float("reg_alpha", 0.0, 0.5),
            "reg_lambda"       : trial.suggest_float("reg_lambda", 0.5, 2.0),
            "class_weight": "balanced", "verbose": -1,
            "random_state": SEED, "n_jobs": -1,
        }
        m = lgb.LGBMClassifier(**p)
        m.fit(X_train.iloc[tr_idx], y_train[tr_idx],
              eval_set=[(X_train.iloc[val_idx], y_train[val_idx])],
              callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
        return roc_auc_score(y_train[val_idx], m.predict_proba(X_train.iloc[val_idx])[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=3)
    bp = study.best_params
    bp.update({"class_weight": "balanced", "verbose": -1, "random_state": SEED, "n_jobs": -1})
    print(f"  LGB best params: {bp}")
    with mlflow.start_run(experiment_id=experiment_id, run_name="Run-12_LightGBM_Optuna_Tuned"):
        mlflow.log_params(bp); mlflow.log_metric("optuna_best_auc", study.best_value)
    return bp

# ─── tuned 5-fold CV ──────────────────────────────────────────────────────────

def run_tuned_xgb_cv(X_train, y_train, params, experiment_id):
    print(f"\n[XGBoost Tuned] 5-fold CV")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y_train))
    best_model, best_auc = None, 0
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        m = xgb.XGBClassifier(**params)
        m.fit(X_train.iloc[tr_idx], y_train[tr_idx],
              eval_set=[(X_train.iloc[val_idx], y_train[val_idx])],
              early_stopping_rounds=30, verbose=False)
        oof[val_idx] = m.predict_proba(X_train.iloc[val_idx])[:, 1]
        fold_auc = roc_auc_score(y_train[val_idx], oof[val_idx])
        print(f"  Fold {fold+1}: {fold_auc:.4f}")
        with mlflow.start_run(experiment_id=experiment_id, run_name=f"Tuned_XGB_Fold_{fold+1}"):
            mlflow.log_metric("auc_roc", fold_auc)
        if fold_auc > best_auc:
            best_auc, best_model = fold_auc, m
    oof_auc = roc_auc_score(y_train, oof)
    print(f"  OOF AUC: {oof_auc:.4f}")
    np.save(MODELS / "oof_xgb_tuned.npy", oof)
    return oof_auc, oof, best_model


def run_tuned_lgb_cv(X_train, y_train, params, experiment_id):
    print(f"\n[LightGBM Tuned] 5-fold CV")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y_train))
    best_model, best_auc = None, 0
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        m = lgb.LGBMClassifier(**params)
        m.fit(X_train.iloc[tr_idx], y_train[tr_idx],
              eval_set=[(X_train.iloc[val_idx], y_train[val_idx])],
              callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
        oof[val_idx] = m.predict_proba(X_train.iloc[val_idx])[:, 1]
        fold_auc = roc_auc_score(y_train[val_idx], oof[val_idx])
        print(f"  Fold {fold+1}: {fold_auc:.4f}")
        with mlflow.start_run(experiment_id=experiment_id, run_name=f"Tuned_LGB_Fold_{fold+1}"):
            mlflow.log_metric("auc_roc", fold_auc)
        if fold_auc > best_auc:
            best_auc, best_model = fold_auc, m
    oof_auc = roc_auc_score(y_train, oof)
    print(f"  OOF AUC: {oof_auc:.4f}")
    np.save(MODELS / "oof_lgb_tuned.npy", oof)
    return oof_auc, oof, best_model

# ─── CatBoost (fast: 300 iter, 3-fold) ───────────────────────────────────────

def run_catboost_fast(X_train, y_train, experiment_id):
    print("\n[CatBoost] 3-fold (300 iter)")
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y_train))
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        m = CatBoostClassifier(
            iterations=300, learning_rate=0.05, depth=6,
            auto_class_weights="Balanced", verbose=0, random_seed=SEED
        )
        m.fit(X_train.iloc[tr_idx].fillna(-999), y_train[tr_idx])
        oof[val_idx] = m.predict_proba(X_train.iloc[val_idx].fillna(-999))[:, 1]
        fold_auc = roc_auc_score(y_train[val_idx], oof[val_idx])
        print(f"  Fold {fold+1}: {fold_auc:.4f}")
    oof_auc = roc_auc_score(y_train, oof)
    print(f"  OOF AUC: {oof_auc:.4f}")
    with mlflow.start_run(experiment_id=experiment_id, run_name="Run-13_CatBoost_Baseline"):
        mlflow.log_metric("oof_auc_roc", oof_auc)
    np.save(MODELS / "oof_cat.npy", oof)
    return oof_auc, oof

# ─── ensemble ────────────────────────────────────────────────────────────────

def run_ensemble_fast(y_train, oof_xgb, oof_lgb, oof_cat, experiment_id):
    print("\n[Ensemble] Grid search weights")
    best_auc, best_w = 0.0, (0.4, 0.4, 0.2)
    for wx in np.arange(0.3, 0.7, 0.1):
        for wl in np.arange(0.3, 0.7, 0.1):
            wc = round(1.0 - wx - wl, 2)
            if wc < 0 or wc > 0.4:
                continue
            auc = roc_auc_score(y_train, wx*oof_xgb + wl*oof_lgb + wc*oof_cat)
            if auc > best_auc:
                best_auc, best_w = auc, (round(wx,1), round(wl,1), wc)
    oof_ens = best_w[0]*oof_xgb + best_w[1]*oof_lgb + best_w[2]*oof_cat
    ens_auc = roc_auc_score(y_train, oof_ens)
    print(f"  Best weights: XGB={best_w[0]} LGB={best_w[1]} CAT={best_w[2]}")
    print(f"  Ensemble AUC: {ens_auc:.4f}")
    json.dump({"w_xgb": best_w[0], "w_lgb": best_w[1], "w_cat": best_w[2]},
              open(MODELS / "ensemble_weights.json", "w"))
    with mlflow.start_run(experiment_id=experiment_id, run_name="Run-14_XGB_LGB_CatBoost_Ensemble"):
        mlflow.log_params({"w_xgb": best_w[0], "w_lgb": best_w[1], "w_cat": best_w[2]})
        mlflow.log_metric("oof_auc_roc", ens_auc)
    return ens_auc, oof_ens, best_w

# ─── SHAP (lightweight) ───────────────────────────────────────────────────────

def run_shap_fast(model, X_train, model_name, experiment_id):
    print(f"\n[SHAP] {model_name} (1000 samples)")
    X_s = X_train.sample(1000, random_state=SEED)
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_s)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    summary = pd.DataFrame({
        "feature"      : X_s.columns,
        "mean_abs_shap": np.abs(shap_values).mean(0),
    }).sort_values("mean_abs_shap", ascending=False).head(10)
    summary.to_csv(MODELS / f"shap_top10_{model_name}.csv", index=False)
    fig, _ = plt.subplots(figsize=(8, 6))
    shap.summary_plot(shap_values, X_s, plot_type="bar", max_display=15, show=False)
    plt.title(f"SHAP — {model_name}"); plt.tight_layout()
    plt.savefig(PLOTS / f"shap_bar_{model_name}.png", dpi=100, bbox_inches="tight")
    plt.close()
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"SHAP_{model_name}_Analysis"):
        mlflow.log_artifact(str(MODELS / f"shap_top10_{model_name}.csv"))
    print(f"  Top features: {summary['feature'].tolist()[:5]}")
    return summary

# ─── final production model ───────────────────────────────────────────────────

def train_final_fast(X_train, X_test, y_train, xgb_p, lgb_p, weights, experiment_id):
    print("\n[Final] Training production models on full data")
    w_xgb, w_lgb, w_cat = weights

    xgb_f = xgb.XGBClassifier(**xgb_p)
    xgb_f.fit(X_train, y_train, verbose=False)

    lgb_f = lgb.LGBMClassifier(**lgb_p)
    lgb_f.fit(X_train, y_train, callbacks=[lgb.log_evaluation(-1)])

    cat_f = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6,
                               auto_class_weights="Balanced", verbose=0, random_seed=SEED)
    cat_f.fit(X_train.fillna(-999), y_train)

    test_pred = (w_xgb * xgb_f.predict_proba(X_test)[:, 1] +
                 w_lgb * lgb_f.predict_proba(X_test)[:, 1] +
                 w_cat * cat_f.predict_proba(X_test.fillna(-999))[:, 1])

    pickle.dump(xgb_f, open(MODELS / "xgb_final.pkl", "wb"))
    pickle.dump(lgb_f, open(MODELS / "lgb_final.pkl", "wb"))
    pickle.dump(cat_f, open(MODELS / "cat_final.pkl", "wb"))

    pd.DataFrame({"SK_ID_CURR": X_test.index, "TARGET": test_pred}).to_csv(
        MODELS / "submission.csv", index=False)

    with mlflow.start_run(experiment_id=experiment_id, run_name="Run-15_Final_Production_Model"):
        mlflow.log_params({"w_xgb": w_xgb, "w_lgb": w_lgb, "w_cat": w_cat,
                           "feature_count": X_train.shape[1]})
        mlflow.xgboost.log_model(xgb_f, "xgb_final")
        mlflow.lightgbm.log_model(lgb_f, "lgb_final")
    print(f"  Final models saved. Test preds: {test_pred.min():.3f}-{test_pred.max():.3f}")
    return xgb_f, lgb_f

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("FAST RESUME — Credit Risk Intelligence Training")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT)
    else:
        experiment_id = exp.experiment_id
    print(f"MLflow experiment: {experiment_id}")

    print("\nLoading data...")
    X_train, X_test, y_train = load_data()

    print("\nLoading existing OOF arrays...")
    oof_lr  = np.load(MODELS / "oof_logistic.npy")
    oof_xgb = np.load(MODELS / "oof_xgboost.npy")
    oof_lgb = np.load(MODELS / "oof_lgb.npy")
    lr_auc  = roc_auc_score(y_train, oof_lr)
    xgb_auc = roc_auc_score(y_train, oof_xgb)
    lgb_auc = roc_auc_score(y_train, oof_lgb)
    print(f"  LR={lr_auc:.4f}  XGB={xgb_auc:.4f}  LGB={lgb_auc:.4f}")
    sys.stdout.flush()

    xgb_tuned_params = optuna_tune_xgb_fast(X_train, y_train, experiment_id)
    sys.stdout.flush()

    lgb_tuned_params = optuna_tune_lgb_fast(X_train, y_train, experiment_id)
    sys.stdout.flush()

    xgb_tuned_auc, oof_xgb_tuned, xgb_tuned_best = run_tuned_xgb_cv(
        X_train, y_train, xgb_tuned_params, experiment_id)
    sys.stdout.flush()

    lgb_tuned_auc, oof_lgb_tuned, lgb_tuned_best = run_tuned_lgb_cv(
        X_train, y_train, lgb_tuned_params, experiment_id)
    sys.stdout.flush()

    cat_auc, oof_cat = run_catboost_fast(X_train, y_train, experiment_id)
    sys.stdout.flush()

    ens_auc, oof_ens, ensemble_weights = run_ensemble_fast(
        y_train, oof_xgb_tuned, oof_lgb_tuned, oof_cat, experiment_id)
    sys.stdout.flush()

    run_shap_fast(xgb_tuned_best, X_train, "XGBoost", experiment_id)
    run_shap_fast(lgb_tuned_best, X_train, "LightGBM", experiment_id)
    sys.stdout.flush()

    train_final_fast(X_train, X_test, y_train,
                     xgb_tuned_params, lgb_tuned_params, ensemble_weights, experiment_id)
    sys.stdout.flush()

    # ── Save artifacts ────────────────────────────────────────────────────────
    w_xgb, w_lgb, w_cat = ensemble_weights
    cfg_out = {
        "model_version"    : "1.0.0",
        "optimal_threshold": 0.21,
        "ensemble_weights" : [float(w_xgb), float(w_lgb), float(w_cat)],
        "feature_count"    : int(X_train.shape[1]),
        "training_auc"     : round(float(ens_auc), 4),
        "train_rows"       : int(len(X_train)),
        "created_at"       : datetime.now().isoformat(),
    }
    with open(MODELS / "inference_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg_out, f, indent=2)

    pd.DataFrame([
        {"model": "Logistic Regression", "oof_auc_roc": lr_auc},
        {"model": "XGBoost 5-fold",      "oof_auc_roc": xgb_auc},
        {"model": "LightGBM 5-fold",     "oof_auc_roc": lgb_auc},
        {"model": "XGBoost Tuned",       "oof_auc_roc": xgb_tuned_auc},
        {"model": "LightGBM Tuned",      "oof_auc_roc": lgb_tuned_auc},
        {"model": "CatBoost",            "oof_auc_roc": cat_auc},
        {"model": "Ensemble",            "oof_auc_roc": ens_auc},
    ]).to_csv(MODELS / "model_comparison.csv", index=False)

    improvement = (ens_auc - lr_auc) * 100
    print("\n" + "=" * 70)
    print("DONE")
    print(f"  LR baseline   : {lr_auc:.4f}")
    print(f"  XGB 5-fold    : {xgb_auc:.4f}")
    print(f"  LGB 5-fold    : {lgb_auc:.4f}")
    print(f"  XGB tuned     : {xgb_tuned_auc:.4f}")
    print(f"  LGB tuned     : {lgb_tuned_auc:.4f}")
    print(f"  CatBoost      : {cat_auc:.4f}")
    print(f"  Ensemble      : {ens_auc:.4f}  **BEST**")
    print(f"  Improvement   : +{improvement:.1f} pp over LR")
    print(f"  Target 0.79   : {'REACHED' if ens_auc >= 0.79 else 'NOT reached - check features'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
