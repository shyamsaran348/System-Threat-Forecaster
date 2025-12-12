#!/usr/bin/env python3
import argparse
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def find_oof_files(outputs_dir: Path):
    candidates = list(outputs_dir.rglob("*oof*.csv"))
    # dedupe
    return list({p.resolve(): p for p in candidates}.keys())

def extract_pred_col(df):
    for c in df.columns:
        if c.lower().count("oof") or c.lower().count("pred"):
            return c
    # fallback
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None

def is_valid_submission(df_sub, test_len):
    """A valid test submission must contain exactly test_len rows + numeric prediction values"""
    if len(df_sub) != test_len:
        return False
    # remove id columns and see if remaining numeric predictions exist
    numeric_cols = [c for c in df_sub.columns if pd.api.types.is_numeric_dtype(df_sub[c]) and c != "id"]
    return len(numeric_cols) >= 1

def run_stack(train_path, test_path, outputs_dir, folds, seed):
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading train/test")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    y = df_train["target"].values
    n = len(df_train)
    test_len = len(df_test)

    # -------------------------
    # 1. Locate OOF files
    # -------------------------
    oof_candidates = find_oof_files(Path("outputs"))
    print(f"[INFO] Found {len(oof_candidates)} OOF candidates.")

    models = []
    for path in oof_candidates:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[SKIP] Failed to read {path}: {e}")
            continue

        if len(df) != n:
            print(f"[SKIP] {path} has {len(df)} rows (expected {n})")
            continue

        col = extract_pred_col(df)
        if not col:
            print(f"[SKIP] {path} has no prediction column")
            continue

        models.append({
            "name": path.stem,
            "oof_path": path,
            "oof_col": col,
            "sub_path": None       # assign later
        })
        print(f"[USE] {path} -> column '{col}'")

    if len(models) < 2:
        raise RuntimeError("Need at least 2 valid OOF files to build stacking.")

    # -------------------------
    # 2. Find valid submissions
    # -------------------------
    print("[INFO] Searching for valid submissions")
    all_csvs = list(Path("outputs").rglob("*.csv"))

    for m in models:
        best_match = None
        for csv_path in all_csvs:
            try:
                df_sub = pd.read_csv(csv_path)
            except:
                continue

            if is_valid_submission(df_sub, test_len):
                # use numeric pred col
                numeric_cols = [c for c in df_sub.columns if pd.api.types.is_numeric_dtype(df_sub[c]) and c != "id"]
                if len(numeric_cols) == 0:
                    continue

                # accept only submissions whose parent folder matches model family
                if m["name"].split("_")[0].lower() in csv_path.parent.name.lower():
                    best_match = csv_path
                    break

        m["sub_path"] = best_match
        print(f"[SUB] For {m['name']} -> {best_match}")

    # -------------------------
    # Build OOF matrix
    # -------------------------
    X_oof = np.zeros((n, len(models)))
    test_pred_matrix = []

    for idx, m in enumerate(models):
        df_oof = pd.read_csv(m["oof_path"])
        X_oof[:, idx] = df_oof[m["oof_col"]].values

        # test preds if valid
        if m["sub_path"]:
            df_sub = pd.read_csv(m["sub_path"])
            pred_col = [c for c in df_sub.columns if pd.api.types.is_numeric_dtype(df_sub[c]) and c != "id"][0]
            test_pred_matrix.append(df_sub[pred_col].values)
        else:
            test_pred_matrix.append(np.zeros(test_len) * np.nan)

    test_pred_matrix = np.array(test_pred_matrix)  # shape: (models, test_len)
    
    # -------------------------
    # 3. Meta-learner CV
    # -------------------------
    print("[INFO] Training stacking meta-learner")
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof_stack = np.zeros(n)

    meta = LogisticRegression(max_iter=2000)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_oof, y)):
        meta.fit(X_oof[tr_idx], y[tr_idx])
        preds = meta.predict_proba(X_oof[val_idx])[:, 1]
        oof_stack[val_idx] = preds

    auc = roc_auc_score(y, oof_stack)
    print(f"[RESULT] STACKED OOF AUC = {auc:.6f}")

    # Train final meta
    meta_final = LogisticRegression(max_iter=2000)
    meta_final.fit(X_oof, y)

    # -------------------------
    # 4. Final test prediction
    # -------------------------
    test_matrix = np.nan_to_num(test_pred_matrix).T
    final_test = meta_final.predict_proba(test_matrix)[:, 1]

    sub = pd.DataFrame({"id": df_test.index, "target": final_test})
    sub.to_csv(outputs_dir / "submission_stacked.csv", index=False)

    pd.DataFrame({"oof_pred": oof_stack}).to_csv(outputs_dir / "stack_oof.csv", index=False)
    joblib.dump(meta_final, outputs_dir / "meta_model.joblib")

    print(f"[INFO] Saved stacked submission and meta model to: {outputs_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data/raw/train.csv")
    parser.add_argument("--test_path", default="data/raw/test.csv")
    parser.add_argument("--out_dir", default="outputs/stacking")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(">>> STACKING SCRIPT STARTED <<<")

    run_stack(
        Path(args.train_path),
        Path(args.test_path),
        Path(args.out_dir),
        args.folds,
        args.seed
    )
