#!/usr/bin/env python3
"""
Optuna-based weight search for final ensemble.
"""

import optuna
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

# Paths
oof_paths = {
    "stack": "outputs/stacking_v2/stack_oof.csv",
    "cat":   "outputs/catboost_full/oof_cat.csv",
    "lgb":   "outputs/tune_lgbm/quick50/oof_preds.csv",
    "xgb":   "outputs/xgb_full/oof_xgb.csv"
}

sub_paths = {
    "stack": "outputs/stacking_v2/submission_stacked.csv",
    "cat":   "outputs/catboost_full/submission_cat.csv",
    "lgb":   "outputs/final/submission_lgbm_tuned.csv",
    "xgb":   "outputs/xgb_full/submission_xgb.csv"
}

print("[INFO] Loading OOF predictions...")

# Load OOF arrays
oof_dict = {name: pd.read_csv(path)["oof_pred"].values for name, path in oof_paths.items()}
y = pd.read_csv("data/raw/train.csv")["target"].values

# Load submission arrays
sub_dict = {name: pd.read_csv(path)["target"].values for name, path in sub_paths.items()}
df_test = pd.read_csv("data/raw/test.csv")

# Order of models
model_names = ["stack", "cat", "lgb", "xgb"]

def objective(trial):
    weights = np.array([
        trial.suggest_float(f"w_{m}", 0.0, 1.0) for m in model_names
    ])
    weights /= weights.sum()  # normalize

    blended = np.zeros_like(y, dtype=float)
    for w, m in zip(weights, model_names):
        blended += w * oof_dict[m]

    auc = roc_auc_score(y, blended)
    return auc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

print("[INFO] Best AUC:", study.best_value)
best_weights = study.best_params
print("[INFO] Best weights:", best_weights)

# ---- Produce final test blend ----
W = np.array([best_weights[f"w_{m}"] for m in model_names])
W /= W.sum()

final_pred = np.zeros(len(df_test))
for w, m in zip(W, model_names):
    final_pred += w * sub_dict[m]

out_dir = Path("outputs/final_ensemble_optuna")
out_dir.mkdir(parents=True, exist_ok=True)

pd.DataFrame({"id": df_test.index, "target": final_pred}).to_csv(
    out_dir / "submission_final_optuna.csv",
    index=False
)

print("[INFO] Saved optimized ensemble prediction to:", out_dir / "submission_final_optuna.csv")
