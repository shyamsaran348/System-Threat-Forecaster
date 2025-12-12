#!/usr/bin/env python3
"""
Optuna tuning for LightGBM (OOF CV).

Usage example:
python -m src.ml.tune_lgbm_optuna \
  --train_path data/raw/train.csv \
  --out_dir outputs/tune_lgbm \
  --n_trials 100 \
  --folds 5 \
  --seed 42 \
  --study_name lgbm_tune_run \
  --save_oof
"""
import argparse
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_path", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_trials", type=int, default=100)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--study_name", default="lgbm_optuna")
    p.add_argument("--save_oof", action="store_true",
                   help="If set, save OOF predictions and trained fold models with best params")
    p.add_argument("--preprocessor", default="outputs/models/preprocessor.joblib",
                   help="Path to saved preprocessor (joblib). If not found, raw features used.")
    return p.parse_args()


def load_data(train_path):
    df = pd.read_csv(train_path)
    if "target" not in df.columns:
        raise ValueError("train CSV must contain 'target' column")
    X = df.drop(columns=["target"])
    y = df["target"].values
    return df, X, y


def preprocess_with_saved(preprocessor_path, X):
    if Path(preprocessor_path).exists():
        pre = joblib.load(preprocessor_path)
        X_processed = pre.transform(X)
        # If transform returns sparse matrix
        if hasattr(X_processed, "toarray"):
            X_processed = X_processed.toarray()
        return X_processed, pre
    else:
        print(f"[WARN] preprocessor not found at {preprocessor_path}. Using raw features.")
        return X.values, None


def objective(trial, X, y, folds, seed):
    params = {
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 16, 1024),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "n_estimators": 10000,
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": -1,
    }

    oof_preds = np.zeros(len(y))
    aucs = []
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model = LGBMClassifier(**params)

        # Use callback-based early stopping for cross-version compatibility
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=200)]
        )

        val_pred = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_pred
        auc = roc_auc_score(y_val, val_pred)
        aucs.append(auc)

        # report intermediate result for pruning
        trial.report(np.mean(aucs), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    mean_auc = float(np.mean(aucs))
    return mean_auc


def run_study(X, y, out_dir, n_trials, folds, seed, study_name, preprocessor_path, save_oof):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    models_dir.mkdir(exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner
    )

    study.optimize(lambda t: objective(t, X, y, folds, seed), n_trials=n_trials, show_progress_bar=True)

    # Save study
    with open(out_dir / "study.pkl", "wb") as f:
        pickle.dump(study, f)

    # Save best params
    best_params = study.best_params.copy()
    best_params.update({
        "n_estimators": 10000,
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": -1
    })
    with open(out_dir / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    print(f"[INFO] best trial value={study.best_value}")
    print(f"[INFO] best params saved to {out_dir/'best_params.json'}")

    # Optionally: train OOF predictions & save fold models using best params
    if save_oof:
        oof_preds = np.zeros(len(y))
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            clf = LGBMClassifier(**best_params)

            clf.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(stopping_rounds=200)]
            )

            val_pred = clf.predict_proba(X_val)[:, 1]
            oof_preds[val_idx] = val_pred
            # Save fold model
            joblib.dump(clf, models_dir / f"lgbm_fold{fold+1}.joblib")

        # Save OOF dataframe
        oof_df = pd.DataFrame({"oof_pred": oof_preds})
        oof_df.to_csv(out_dir / "oof_preds.csv", index=False)
        print(f"[INFO] saved OOF preds and fold models to {models_dir}")

    return study


def main():
    args = parse_args()
    df, X_raw, y = load_data(args.train_path)
    X_proc, pre = preprocess_with_saved(args.preprocessor, X_raw)

    if not isinstance(X_proc, np.ndarray):
        X_proc = np.asarray(X_proc)

    if y.dtype.kind in "O":
        le = LabelEncoder()
        y = le.fit_transform(y)

    run_study(X_proc, y, args.out_dir, args.n_trials, args.folds, args.seed,
              args.study_name, args.preprocessor, args.save_oof)


if __name__ == "__main__":
    main()
