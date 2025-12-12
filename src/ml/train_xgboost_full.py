#!/usr/bin/env python3
"""
Train XGBoost with K-fold OOF on full dataset using xgb.train()
(compatible with all XGBoost versions)

Outputs:
 - <out_dir>/oof_xgb.csv
 - <out_dir>/xgb_fold{n}.json
 - <out_dir>/submission_xgb.csv
"""

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings("ignore")

# --- Fix import path for preprocessor.joblib ---
import sys
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_preprocessor(p):
    p = Path(p)
    if p.exists():
        print("[INFO] Loading preprocessor:", p)
        return joblib.load(p)
    print("[WARN] No preprocessor found — using numeric-only features")
    return None


def prepare_matrix(pre, df):
    if pre:
        X = pre.transform(df)
        if hasattr(X, "toarray"):
            X = X.toarray()
        return X
    else:
        return df.select_dtypes(include=[np.number]).values


def train_xgb(train_path, test_path, out_dir, folds=5, seed=42,
              preprocessor_path="outputs/models/preprocessor.joblib"):

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading data...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    y = df_train["target"].values

    pre = load_preprocessor(preprocessor_path)
    X = prepare_matrix(pre, df_train)
    X_test = prepare_matrix(pre, df_test)

    n = len(df_train)
    oof = np.zeros(n)
    test_preds = np.zeros(len(df_test))

    params = {
        "eta": 0.02,
        "max_depth": 7,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "seed": seed,
        "verbosity": 0
    }

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    for fold, (tr, va) in enumerate(skf.split(X, y)):
        print(f"[INFO] XGB Fold {fold+1}/{folds}")

        dtrain = xgb.DMatrix(X[tr], label=y[tr])
        dvalid = xgb.DMatrix(X[va], label=y[va])
        dtest = xgb.DMatrix(X_test)

        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=20000,
            evals=[(dvalid, "valid")],
            early_stopping_rounds=200,
            verbose_eval=False
        )

        val_pred = booster.predict(dvalid)
        oof[va] = val_pred
        fold_auc = roc_auc_score(y[va], val_pred)
        print(f"[FOLD {fold+1}] AUC = {fold_auc:.6f}")

        # accumulate test predictions
        test_preds += booster.predict(dtest) / folds

        booster.save_model(str(out_dir / f"xgb_fold{fold+1}.json"))

    # save OOF + submission
    pd.DataFrame({"oof_pred": oof}).to_csv(out_dir / "oof_xgb.csv", index=False)
    pd.DataFrame({"id": df_test.index, "target": test_preds}).to_csv(
        out_dir / "submission_xgb.csv", index=False
    )

    final_auc = roc_auc_score(y, oof)
    print("[INFO] XGB FINAL OOF AUC:", final_auc)
    print("[INFO] Saved outputs to:", out_dir)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_path", default="data/raw/train.csv")
    p.add_argument("--test_path", default="data/raw/test.csv")
    p.add_argument("--out_dir", default="outputs/xgb_full")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--preprocessor", default="outputs/models/preprocessor.joblib")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_xgb(args.train_path, args.test_path, args.out_dir, args.folds, args.seed, args.preprocessor)
