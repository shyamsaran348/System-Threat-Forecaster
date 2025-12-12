#!/usr/bin/env python3
"""
shap_explain_full.py
- Loads preprocessor and models from outputs/
- Computes SHAP for tuned LightGBM, base models, and stacking meta-model
- Saves plots to outputs/shap/
"""

import os
import glob
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from pprint import pprint

# ML libraries
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# plotting
import matplotlib.pyplot as plt

# SHAP
import shap

# --------- CONFIG ----------
SAVE_DIR = Path(os.getenv("SHAP_SAVE_DIR", "outputs/shap"))
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# sample size for SHAP background (for KernelExplainer or sampling)
DEFAULT_BACKGROUND_SIZE = 200
# number of rows to compute and plot SHAP (for speed); set higher for final analysis
DEFAULT_N_SAMPLES = int(os.getenv("N_SAMPLES", "2000"))

# paths (auto-discovery)
PREPROCESSOR_PATHS = list(Path("outputs/models").glob("preprocessor*.joblib")) + list(Path("outputs/models").glob("*.joblib"))
LGB_MODELS = list(Path("outputs/models").glob("lgbm*tuned*.txt")) + list(Path("outputs/models").glob("lgbm*.txt")) + list(Path("outputs/models").glob("lgbm*tuned*.pkl"))
CATBOOST_MODELS = list(Path("outputs/models").glob("*.cbm")) + list(Path("outputs/models").glob("catboost*.cbm")) + list(Path("outputs/models").glob("catboost*.cbm"))
XGB_MODELS = list(Path("outputs/models").glob("xgb*.json")) + list(Path("outputs/models").glob("xgb*.model")) + list(Path("outputs/models").glob("xgb*.bin"))
META_MODEL_PATHS = list(Path("outputs/models").glob("meta*")) + list(Path("outputs/models").glob("stack*")) + list(Path("outputs/models").glob("meta_*.pkl")) + list(Path("outputs/models").glob("stacking_*.pkl"))

# fallback: try outputs/ root
if not PREPROCESSOR_PATHS:
    PREPROCESSOR_PATHS = list(Path("outputs").glob("preprocessor*.joblib"))
# --------- helper functions ----------
def find_first(paths):
    return paths[0] if paths else None

def load_preprocessor(path):
    print("Loading preprocessor:", path)
    return joblib.load(path)

def load_lgb_model(path):
    print("Loading LightGBM model:", path)
    # lgb.Booster or sklearn wrapper
    # Try joblib first
    if str(path).endswith(".pkl") or str(path).endswith(".joblib"):
        model = joblib.load(path)
        return model
    try:
        booster = lgb.Booster(model_file=str(path))
        return booster
    except Exception as e:
        print("Failed to load as Booster:", e)
        raise

def load_catboost(path):
    print("Loading CatBoost model:", path)
    m = CatBoostClassifier()
    m.load_model(str(path))
    return m

def load_xgb(path):
    print("Loading XGBoost model:", path)
    # try xgboost.Booster
    try:
        booster = xgb.Booster()
        booster.load_model(str(path))
        return booster
    except Exception as e:
        print("Failed to load XGBoost booster:", e)
        raise

def load_meta(path):
    print("Loading meta model:", path)
    m = joblib.load(path)
    return m

# ---------- Discover models ----------
preproc_path = find_first(PREPROCESSOR_PATHS)
lgb_path = find_first(LGB_MODELS)
cat_path = find_first(CATBOOST_MODELS)
xgb_path = find_first(XGB_MODELS)
meta_path = find_first(META_MODEL_PATHS)

print("Discovered files:")
print("preprocessor:", preproc_path)
print("lgb:", lgb_path)
print("cat:", cat_path)
print("xgb:", xgb_path)
print("meta:", meta_path)

# If missing prints
if not preproc_path:
    raise FileNotFoundError("Preprocessor not found in outputs/models/*.joblib. Please place preprocessor.joblib at outputs/models/ or pass path manually.")

# ---------- Load preprocessor ----------
preprocessor = load_preprocessor(preproc_path)

# ---------- Load a sample dataset ----------
# Attempt to find a train csv under data/ or outputs/
possible_csv = list(Path("data").glob("*.csv")) + list(Path("outputs").glob("*.csv")) + list(Path("data").glob("train*.csv"))
train_csv = possible_csv[0] if possible_csv else None
if not train_csv:
    raise FileNotFoundError("No CSV train file found in data/ or outputs/. Please place train.csv or give a dataset.")
print("Using dataset:", train_csv)
df = pd.read_csv(train_csv)
if "target" in df.columns:
    X_raw = df.drop(columns=["target"])
    y = df["target"]
else:
    X_raw = df.copy()
    y = None

# Apply preprocessor to get model-ready X
print("Transforming features with preprocessor...")
X = preprocessor.transform(X_raw)  # assume this returns numpy array or pandas
# if preprocessor returns numpy, try to get feature names if available
try:
    feature_names = preprocessor.get_feature_names_out(X_raw.columns)
except Exception:
    # fallback: numeric feature names
    if hasattr(X, "shape"):
        feature_names = [f"f{i}" for i in range(X.shape[1])]
    else:
        feature_names = X_raw.columns.tolist()

# For SHAP we will sample rows for speed if many rows
n_samples = min(DEFAULT_N_SAMPLES, X.shape[0])
sample_idx = np.random.choice(np.arange(X.shape[0]), size=n_samples, replace=False)
X_sample = X[sample_idx] if isinstance(X, np.ndarray) else X.iloc[sample_idx]
X_raw_sample = X_raw.iloc[sample_idx]

# ---------- 1) SHAP for Tuned LightGBM ----------
if lgb_path:
    try:
        lgb_model = load_lgb_model(lgb_path)
        # If sklearn wrapper, handle predict_proba; if lgb.Booster use .predict
        print("Creating TreeExplainer for LightGBM...")
        explainer = shap.TreeExplainer(lgb_model)
        shap_values = explainer.shap_values(X_sample)
        # shap_values for binary classifier: list [neg, pos] or array
        if isinstance(shap_values, list):
            # choose class 1 (positive)
            shap_val_pos = shap_values[1]
        else:
            shap_val_pos = shap_values
        # Save shap values
        np.save(SAVE_DIR / "shap_lgb_values.npy", shap_val_pos)
        joblib.dump(explainer, SAVE_DIR / "shap_lgb_explainer.joblib")
        print("Saved LGB SHAP values and explainer to", SAVE_DIR)
        # Summary plot
        plt.figure(figsize=(8,6))
        shap.summary_plot(shap_val_pos, X_raw_sample, show=False)
        plt.tight_layout()
        plt.savefig(SAVE_DIR / "shap_lgb_summary.png", dpi=150)
        plt.close()
        print("Saved shap_lgb_summary.png")
    except Exception as e:
        print("Failed to compute SHAP for LightGBM:", e)

else:
    print("No LightGBM model found automatically. Place it at outputs/models/ or modify the script.")

# ---------- 2) SHAP for base models (CatBoost, XGB) ----------
base_shap = {}  # dictionary to store per-model shap arrays keyed by model name
if cat_path:
    try:
        cat_model = load_catboost(cat_path)
        print("CatBoost: using shap.TreeExplainer")
        cat_expl = shap.TreeExplainer(cat_model)
        cat_shap = cat_expl.shap_values(X_sample)
        # cat_shap may be list for multiclass; pick pos
        if isinstance(cat_shap, list):
            cat_shap = cat_shap[1]
        base_shap["catboost"] = {"explainer": cat_expl, "shap_values": cat_shap}
        np.save(SAVE_DIR / "shap_cat_values.npy", cat_shap)
        plt.figure(figsize=(8,6))
        shap.summary_plot(cat_shap, X_raw_sample, show=False)
        plt.tight_layout(); plt.savefig(SAVE_DIR / "shap_cat_summary.png"); plt.close()
    except Exception as e:
        print("CatBoost SHAP failed:", e)

if xgb_path:
    try:
        xgb_model = load_xgb(xgb_path)
        xgb_expl = shap.TreeExplainer(xgb_model)
        xgb_shap = xgb_expl.shap_values(X_sample)
        if isinstance(xgb_shap, list):
            xgb_shap = xgb_shap[1]
        base_shap["xgb"] = {"explainer": xgb_expl, "shap_values": xgb_shap}
        np.save(SAVE_DIR / "shap_xgb_values.npy", xgb_shap)
        plt.figure(figsize=(8,6))
        shap.summary_plot(xgb_shap, X_raw_sample, show=False)
        plt.tight_layout(); plt.savefig(SAVE_DIR / "shap_xgb_summary.png"); plt.close()
    except Exception as e:
        print("XGBoost SHAP failed:", e)

# ---------- 3) SHAP for Stacking meta-model ----------
if meta_path:
    try:
        meta_model = load_meta(meta_path)
        # We need meta-feature matrix: detect stacking OOF files or reconstruct from base model predictions
        # Try to find a CSV with "oof" or "stack" in filename
        oof_candidates = list(Path("outputs").glob("*oof*.csv")) + list(Path("outputs").glob("*stack*.csv")) + list(Path("outputs").glob("outputs/*oof*.csv"))
        oof_path = oof_candidates[0] if oof_candidates else None
        if oof_path:
            df_oof = pd.read_csv(oof_path)
            # Expect columns like lgb_pred, cat_pred, xgb_pred etc. Find numeric cols
            meta_features = df_oof.select_dtypes("number").drop(columns=["target"], errors="ignore")
            print("Using OOF CSV for meta-features:", oof_path)
            # sample meta rows corresponding to sample_idx if indices align; else sample first rows
            meta_sample = meta_features.iloc[sample_idx] if sample_idx.size <= len(meta_features) else meta_features.iloc[:len(sample_idx)]
        else:
            # fallback: generate meta features by predicting X_sample with base models
            preds = {}
            if lgb_path:
                if isinstance(lgb_model, lgb.Booster):
                    p = lgb_model.predict(X_sample)
                else:
                    p = lgb_model.predict_proba(X_sample)[:,1]
                preds["lgb"] = p
            if cat_path:
                preds["cat"] = cat_model.predict_proba(X_sample)[:,1]
            if xgb_path:
                preds["xgb"] = xgb_model.predict(xgb.DMatrix(X_sample))
            meta_sample = pd.DataFrame(preds)

        # Now explain the meta-model
        # If meta is linear/logistic -> LinearExplainer
        print("Meta model class:", meta_model.__class__)
        if hasattr(shap, "LinearExplainer") and ("Logistic" in meta_model.__class__.__name__ or hasattr(meta_model, "coef_")):
            me = shap.LinearExplainer(meta_model, meta_sample, feature_dependence="independent")
            meta_shap = me.shap_values(meta_sample)
            np.save(SAVE_DIR / "shap_meta_values.npy", meta_shap)
            joblib.dump(me, SAVE_DIR / "shap_meta_explainer.joblib")
            plt.figure(figsize=(6,4))
            shap.summary_plot(meta_shap, meta_sample, show=False)
            plt.tight_layout(); plt.savefig(SAVE_DIR / "shap_meta_summary.png"); plt.close()
            print("Saved meta-model SHAP summary")
        else:
            # fallback KernelExplainer (slower)
            print("Using KernelExplainer for meta-model (slower).")
            me = shap.KernelExplainer(meta_model.predict_proba, shap.kmeans(meta_sample, min(50, len(meta_sample))))
            meta_shap = me.shap_values(meta_sample)
            if isinstance(meta_shap, list):
                meta_shap = meta_shap[1]
            np.save(SAVE_DIR / "shap_meta_values.npy", meta_shap)
            plt.figure(figsize=(6,4))
            shap.summary_plot(meta_shap, meta_sample, show=False)
            plt.tight_layout(); plt.savefig(SAVE_DIR / "shap_meta_summary.png"); plt.close()
    except Exception as e:
        print("Meta-model SHAP failed:", e)
else:
    print("Meta model not found automatically. Place meta model under outputs/models/ with 'meta' or 'stack' in filename.")

# ---------- 4) Approximate 'stacked' attribution -> map meta back to original features ----------
# Only possible if: meta is linear and base SHAPs exist per base model.
try:
    if meta_path and base_shap:
        # load meta model coefficients
        if hasattr(meta_model, "coef_"):
            coefs = meta_model.coef_.flatten()
            meta_feature_names = meta_sample.columns.tolist()
            coef_map = dict(zip(meta_feature_names, coefs))
            print("Meta coeffs:", coef_map)
            # For each base model we have base_shap[model]["shap_values"] aligned with X_raw_sample columns
            # Multiply each base shap by meta coef for that base's meta-feature, then sum across base models
            # Ensure we align feature order
            # Get base feature names -> use X_raw_sample.columns
            final_shap = np.zeros_like(list(base_shap.values())[0]["shap_values"])
            for meta_fname in meta_feature_names:
                base_key = None
                # try to map meta feature name to base model key
                if "lgb" in meta_fname.lower():
                    base_key = "lgb"
                elif "cat" in meta_fname.lower():
                    base_key = "catboost"
                elif "xgb" in meta_fname.lower():
                    base_key = "xgb"
                else:
                    # try direct key
                    if meta_fname in base_shap:
                        base_key = meta_fname
                if base_key and base_key in base_shap:
                    b_shap = base_shap[base_key]["shap_values"]
                    # Some shap arrays may be (n_samples, n_features)
                    # Resize if needed
                    coef = coef_map[meta_fname]
                    # Weighted add
                    # ensure shapes align
                    if b_shap.shape == final_shap.shape:
                        final_shap += coef * b_shap
                    else:
                        # attempt to broadcast (if categories mismatch)
                        print(f"Shape mismatch for {base_key}: base_shap {b_shap.shape}, final {final_shap.shape}")
                else:
                    print("No base SHAP found for meta feature:", meta_fname)
            # save approximate final stacked SHAP
            np.save(SAVE_DIR / "shap_stacked_approx.npy", final_shap)
            plt.figure(figsize=(8,6))
            shap.summary_plot(final_shap, X_raw_sample, show=False)
            plt.tight_layout(); plt.savefig(SAVE_DIR / "shap_stacked_approx_summary.png"); plt.close()
            print("Saved approximate stacked SHAP summary.")
        else:
            print("Meta model has no coef_; cannot compute linear combination. Consider KernelExplainer mapping instead.")
except Exception as e:
    print("Failed to compute approximate stacked attribution:", e)

print("Done. Outputs saved in:", SAVE_DIR)
