#!/usr/bin/env python3
# shap_explain_full_args.py — fixed: aligns SHAP arrays and feature matrix

import argparse, joblib, numpy as np, pandas as pd, shap, matplotlib.pyplot as plt
from pathlib import Path
import lightgbm as lgb

def get_feature_names_from_preprocessor(preproc, input_cols):
    """
    Try several methods to get feature names after preprocessing.
    Returns list of names of length matching transformed features.
    """
    # 1) Try get_feature_names_out (sklearn >=1.0)
    try:
        names = preproc.get_feature_names_out(input_cols)
        return list(names)
    except Exception:
        pass

    # 2) If ColumnTransformer with named transformers
    try:
        # This works for sklearn ColumnTransformer
        if hasattr(preproc, "transformers_"):
            names = []
            for name, trans, cols in preproc.transformers_:
                # skip 'drop' or 'passthrough' special cases
                if trans == "drop":
                    continue
                if trans == "passthrough":
                    if isinstance(cols, (list, tuple)):
                        names.extend(list(cols))
                    else:
                        names.append(cols)
                else:
                    # try to get feature names from transformer
                    try:
                        if hasattr(trans, "get_feature_names_out"):
                            out_names = trans.get_feature_names_out(cols)
                            names.extend(list(out_names))
                        elif hasattr(trans, "get_feature_names"):
                            out_names = trans.get_feature_names(cols)
                            names.extend(list(out_names))
                        else:
                            # fallback: create prefixed names
                            if isinstance(cols, (list, tuple)):
                                for c in cols:
                                    names.append(f"{name}__{c}")
                            else:
                                names.append(f"{name}__{cols}")
                    except Exception:
                        if isinstance(cols, (list, tuple)):
                            for c in cols:
                                names.append(f"{name}__{c}")
                        else:
                            names.append(f"{name}__{cols}")
            return names
    except Exception:
        pass

    # 3) fallback: create generic names based on transformed shape later
    return None

def main():
    p = argparse.ArgumentParser(description="SHAP explain (LightGBM)")
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--preproc", type=str, required=True)
    p.add_argument("--lgb_model", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="outputs/shap")
    p.add_argument("--n_samples", type=int, default=2000)
    args = p.parse_args()

    SAVE_DIR = Path(args.save_dir); SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset:", args.train_csv)
    df = pd.read_csv(args.train_csv)
    if "target" in df.columns:
        X_raw = df.drop(columns=["target"])
    else:
        X_raw = df.copy()

    print("Loading preprocessor:", args.preproc)
    preproc = joblib.load(args.preproc)

    print("Transforming features...")
    X_trans = preproc.transform(X_raw)  # numpy array or pandas DataFrame

    # get or synthesize feature names for transformed features
    feature_names = get_feature_names_from_preprocessor(preproc, X_raw.columns)
    if feature_names is None:
        # fallback: create generic names based on X_trans shape
        n_feats = X_trans.shape[1]
        feature_names = [f"f{i}" for i in range(n_feats)]
        print(f"Warning: could not infer feature names from preprocessor; using generic names f0..f{n_feats-1}")

    # ensure X_trans_sample is numpy array
    n = min(args.n_samples, X_trans.shape[0])
    idx = np.random.choice(X_trans.shape[0], size=n, replace=False)
    if isinstance(X_trans, (pd.DataFrame)):
        X_samp = X_trans.iloc[idx].to_numpy()
    else:
        X_samp = X_trans[idx]

    # Create DataFrame that matches SHAP's expected shape & column names
    X_samp_df = pd.DataFrame(X_samp, columns=feature_names)

    print("Loading LightGBM model:", args.lgb_model)
    lgb_model = lgb.Booster(model_file=args.lgb_model)

    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(X_samp)
    shap_pos = shap_values[1] if isinstance(shap_values, list) else shap_values

    # ensure shapes align
    if shap_pos.shape[1] != X_samp_df.shape[1]:
        raise ValueError(f"SHAP values columns ({shap_pos.shape[1]}) != transformed features ({X_samp_df.shape[1]}).")

    # Save arrays and explainer
    np.save(SAVE_DIR / "shap_lgb_values.npy", shap_pos)
    joblib.dump(explainer, SAVE_DIR / "shap_lgb_explainer.joblib")

    print("Saving summary plot...")
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_pos, X_samp_df, show=False)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "shap_lgb_summary.png", dpi=150)
    plt.close()

    print("Saved outputs in:", SAVE_DIR)

if __name__ == "__main__":
    main()
