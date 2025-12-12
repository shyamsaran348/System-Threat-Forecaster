# src/ml/train.py
"""
Robust train script for System Thread Forecaster (LightGBM + TabularPreprocessor)

Main fixes vs earlier version:
 - Ensures project root is on sys.path so imports like `from src.ml.preprocess` work.
 - Provides clearer import errors and helpful hints.
 - Otherwise behavior is the same: build preprocessor, train OOF LightGBM, save artifacts.
"""

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Tuple

# --- Ensure project root on sys.path so `from src.ml...` works ---
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]  # project_root/src/ml/train.py -> project_root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# now safe imports
try:
    from src.ml.preprocess import build_preprocessor_from_df, TabularPreprocessor
except Exception as e:
    raise ImportError(
        "Failed to import preprocess module. Make sure file src/ml/preprocess.py exists "
        "and defines build_preprocessor_from_df and TabularPreprocessor.\n"
        f"Original error: {e}"
    )

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import joblib

def ensure_dirs(out_dir: Path):
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (out_dir / "submission").mkdir(parents=True, exist_ok=True)

def load_data(train_path: Path, test_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

def get_feature_names_safe(tp: TabularPreprocessor, X_trans: np.ndarray):
    try:
        names = tp.get_feature_names_out()
        if len(names) == X_trans.shape[1]:
            return names
    except Exception:
        pass
    # fallback generic names
    return [f"f{i}" for i in range(X_trans.shape[1])]

def train_oof_lgbm(X: np.ndarray, y: np.ndarray, params: dict, n_splits: int, seed: int,
                   out_dir: Path, feature_names: list):
    """
    Out-of-fold LightGBM training using callbacks for early stopping (compatible across LGBM versions).
    Returns: oof_preds, cv_auc, feature_importance_df, list_of_model_paths
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(y))
    feature_importance_df = pd.DataFrame()
    fold = 0
    saved_model_paths = []

    for train_idx, valid_idx in skf.split(X, y):
        fold += 1
        print(f"\n--- Fold {fold}/{n_splits} ---")
        X_train, X_val = X[train_idx], X[valid_idx]
        y_train, y_val = y[train_idx], y[valid_idx]

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        # Prepare safe params and callback-compatible settings
        num_boost_round = int(params.get("num_boost_round", 1000))
        es_rounds = int(params.get("early_stopping_rounds", 100))

        lgb_params_safe = params.copy()
        lgb_params_safe.pop("num_boost_round", None)
        lgb_params_safe.pop("early_stopping_rounds", None)

        callbacks = []
        if es_rounds and es_rounds > 0:
            # early stopping callback
            try:
                callbacks.append(lgb.early_stopping(stopping_rounds=es_rounds))
            except Exception:
                # for very old versions fallback to callback wrapper
                from lightgbm.callback import early_stopping as _es
                callbacks.append(_es(stopping_rounds=es_rounds))
        # logging callback
        try:
            callbacks.append(lgb.log_evaluation(period=100))
        except Exception:
            from lightgbm.callback import log_evaluation as _log
            callbacks.append(_log(period=100))

        # Train with callbacks (works across versions)
        booster = lgb.train(
            lgb_params_safe,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dtrain, dval],
            valid_names=["train", "valid"],
            callbacks=callbacks
        )

        # Save model per fold
        model_path = out_dir / "models" / f"lgbm_fold{fold}.txt"
        booster.save_model(str(model_path))
        saved_model_paths.append(model_path)
        print(f"Saved fold model to {model_path}")

        # OOF preds
        val_preds = booster.predict(X_val, num_iteration=booster.best_iteration)
        oof_preds[valid_idx] = val_preds

        # Feature importance (gain)
        try:
            imp = booster.feature_importance(importance_type="gain")
            imp_df = pd.DataFrame({
                "feature": feature_names,
                "importance": imp,
                "fold": fold
            })
            feature_importance_df = pd.concat([feature_importance_df, imp_df], axis=0)
        except Exception as e:
            print("Warning: could not extract feature importances for this fold:", e)

        # fold AUC
        fold_auc = roc_auc_score(y_val, val_preds)
        print(f"Fold {fold} AUC: {fold_auc:.6f}")

    # CV metric
    cv_auc = roc_auc_score(y, oof_preds)
    print(f"\n==== Overall OOF AUC: {cv_auc:.6f} ====")

    # Average importances
    if not feature_importance_df.empty:
        fi = feature_importance_df.groupby("feature")["importance"].mean().reset_index().sort_values("importance", ascending=False)
    else:
        fi = pd.DataFrame(columns=["feature", "importance"])

    return oof_preds, cv_auc, fi, saved_model_paths


def main(args):
    out_dir = Path(args.out_dir)
    ensure_dirs(out_dir)

    print("Loading data...")
    train, test = load_data(Path(args.train_path), Path(args.test_path))

    if args.debug:
        train = train.sample(n=min(20000, len(train)), random_state=args.seed).reset_index(drop=True)
        print(f"DEBUG mode: sampled {len(train)} rows for training")

    if "target" not in train.columns:
        raise KeyError("Training CSV must contain a 'target' column.")

    y = train["target"].astype(int).values

    print("Building & fitting preprocessor...")
    tp, feature_lists = build_preprocessor_from_df(
        train,
        n_hash_features=args.n_hash_features,
        low_card_thresh=args.low_card_thresh,
        drop_id_cols=args.drop_id_cols.split(",") if args.drop_id_cols else None,
        date_cols=None,
        target_col="target"
    )

    preproc_path = out_dir / "models" / "preprocessor.joblib"
    tp.save(str(preproc_path))
    print("Saved preprocessor to", preproc_path)

    print("Transforming training data...")
    X = tp.transform(train)
    print("X shape:", X.shape)

    feature_names = get_feature_names_safe(tp, X)

    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "seed": args.seed,
        "learning_rate": args.lr,
        "num_leaves": args.num_leaves,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": 1,
        "lambda_l1": args.lambda_l1,
        "lambda_l2": args.lambda_l2,
        "num_boost_round": args.n_rounds,
        "early_stopping_rounds": args.early_stopping
    }

    print("Starting OOF training with LightGBM...")
    oof_preds, cv_auc, fi_df, model_paths = train_oof_lgbm(X, y, lgb_params, args.n_splits, args.seed, out_dir, feature_names)

    # Save OOF preds
    oof_df = pd.DataFrame({"index": np.arange(len(y)), "target": y, "oof_pred": oof_preds})
    oof_path = out_dir / "metrics" / "oof_predictions.csv"
    oof_df.to_csv(oof_path, index=False)
    print("Saved OOF predictions to", oof_path)

    # Save metrics
    metrics_path = out_dir / "metrics" / "cv_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"oof_auc: {cv_auc}\n")
    print("Saved CV metrics to", metrics_path)

    # Save feature importance
    fi_path = out_dir / "metrics" / "feature_importances.csv"
    fi_df.to_csv(fi_path, index=False)
    print("Saved feature importances to", fi_path)

    # Prepare test and predict with ensemble of fold models
    print("Preparing test set and generating submission...")
    X_test = tp.transform(test)
    print("X_test shape:", X_test.shape)

    preds = np.zeros(X_test.shape[0], dtype=float)
    for model_path in model_paths:
        booster = lgb.Booster(model_file=str(model_path))
        preds += booster.predict(X_test, num_iteration=booster.best_iteration) / len(model_paths)

    submission = pd.DataFrame({
        "MachineID": test["MachineID"] if "MachineID" in test.columns else test.index,
        "target": preds
    })
    submission_path = out_dir / "submission" / "submission_lgbm.csv"
    submission.to_csv(submission_path, index=False)
    print("Saved submission to", submission_path)

    # Summary
    summary = {
        "cv_auc": float(cv_auc),
        "n_rows_train": int(len(train)),
        "n_rows_test": int(len(test)),
        "feature_counts": {k: len(v) for k, v in feature_lists.items()}
    }
    with open(out_dir / "metrics" / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved summary to", out_dir / "metrics" / "summary.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_rounds", type=int, default=2000)
    parser.add_argument("--early_stopping", type=int, default=100)
    parser.add_argument("--n_hash_features", type=int, default=128)
    parser.add_argument("--low_card_thresh", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--num_leaves", type=int, default=31)
    parser.add_argument("--feature_fraction", type=float, default=0.8)
    parser.add_argument("--bagging_fraction", type=float, default=0.9)
    parser.add_argument("--lambda_l1", type=float, default=0.0)
    parser.add_argument("--lambda_l2", type=float, default=0.0)
    parser.add_argument("--drop_id_cols", type=str, default=None,
                        help="comma-separated ID cols to drop in addition to default (MachineID). Example: 'MachineID,OtherID'")
    parser.add_argument("--debug", action="store_true", help="use small sample for quick debug")
    args = parser.parse_args()
    main(args)
