#!/usr/bin/env python3
"""
Train CatBoost with 5-fold OOF on full dataset.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def train_catboost(train_path, test_path, out_dir, folds=5, seed=42):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading data...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    y = df_train["target"].values
    X = df_train.drop(columns=["target"])
    X_test = df_test.copy()

    # Identify categorical columns
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    print("[INFO] Using CatBoost with categorical features:", cat_cols)

    # ⭐ FIX: Convert categorical columns to string and fill NaNs
    X[cat_cols] = X[cat_cols].astype(str).fillna("NA")
    X_test[cat_cols] = X_test[cat_cols].astype(str).fillna("NA")

    n = len(df_train)
    oof = np.zeros(n)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    test_preds = np.zeros(len(df_test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"[INFO] Training fold {fold+1}/{folds}...")

        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        train_pool = Pool(X_tr, y_tr, cat_features=cat_cols)
        val_pool = Pool(X_val, y_val, cat_features=cat_cols)

        model = CatBoostClassifier(
            depth=8,
            learning_rate=0.03,
            loss_function="Logloss",
            eval_metric="AUC",
            random_state=seed,
            verbose=False,
            iterations=5000,
            od_type="Iter",
            od_wait=200
        )

        model.fit(train_pool, eval_set=val_pool)

        val_pred = model.predict_proba(X_val)[:, 1]
        oof[val_idx] = val_pred
        fold_auc = roc_auc_score(y_val, val_pred)
        print(f"[FOLD {fold+1}] AUC = {fold_auc:.6f}")

        model.save_model(out_dir / f"cat_fold{fold+1}.cbm")

        test_pool = Pool(X_test, cat_features=cat_cols)
        test_preds += model.predict_proba(test_pool)[:, 1] / folds

    pd.DataFrame({"oof_pred": oof}).to_csv(out_dir / "oof_cat.csv", index=False)
    pd.DataFrame({"id": df_test.index, "target": test_preds}).to_csv(out_dir / "submission_cat.csv", index=False)

    print("[INFO] CatBoost OOF AUC:", roc_auc_score(y, oof))
    print("[INFO] Saved to:", out_dir)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_path", default="data/raw/train.csv")
    p.add_argument("--test_path", default="data/raw/test.csv")
    p.add_argument("--out_dir", default="outputs/catboost_full")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_catboost(args.train_path, args.test_path, args.out_dir, args.folds, args.seed)
