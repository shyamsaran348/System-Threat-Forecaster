#!/usr/bin/env python3
"""
Train final LightGBM on full training data using best_params.json produced by Optuna.
Saves final model and submission CSV to outputs/final/
"""

# Make repo root importable so joblib.load() can unpickle objects that refer to `src.*`
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # project_root/src/ml/this_file -> two parents up
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import joblib
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder

# ----- Paths (adjust if needed) -----
BEST = Path("outputs/tune_lgbm/quick50/best_params.json")
PRE = Path("outputs/models/preprocessor.joblib")
TRAIN = Path("data/raw/train.csv")
TEST = Path("data/raw/test.csv")
OUT_DIR = Path("outputs/final")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----- Load data -----
print("[INFO] Loading training data...")
df = pd.read_csv(TRAIN)
if "target" not in df.columns:
    raise ValueError("train CSV must contain 'target' column.")
X = df.drop(columns=["target"])
y = df["target"].values

# ----- Load preprocessor (if exists) and transform -----
pre = None
if PRE.exists():
    print(f"[INFO] Loading preprocessor from {PRE} ...")
    pre = joblib.load(PRE)
    X_proc = pre.transform(X)
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()
else:
    print(f"[WARN] Preprocessor not found at {PRE}. Using raw features.")
    X_proc = X.values

# ----- Load best params -----
if not BEST.exists():
    raise FileNotFoundError(f"Best params not found at {BEST}. Run tuning first.")
with open(BEST, "r") as f:
    best = json.load(f)

# Defensive conversions and safe defaults
for k, v in list(best.items()):
    # convert float integers to int where appropriate
    if isinstance(v, float) and float(v).is_integer():
        best[k] = int(v)
# ensure safe keys
best.setdefault("random_state", 42)
best.setdefault("n_jobs", -1)
best.setdefault("verbosity", -1)
# If 'n_estimators' is huge, you may optionally cap it (uncomment line below)
# best["n_estimators"] = min(best.get("n_estimators", 10000), 10000)

print("[INFO] Best params loaded:")
print(json.dumps(best, indent=2))

# ----- Train on full dataset -----
print("[INFO] Training final model on full data...")
clf = LGBMClassifier(**best)
clf.fit(X_proc, y)
print("[INFO] Training complete.")

# ----- Save final model and preprocessor -----
joblib.dump(clf, OUT_DIR / "final_lgbm_tuned.joblib")
if pre is not None:
    joblib.dump(pre, OUT_DIR / "preprocessor.joblib")
print(f"[INFO] Saved final model and preprocessor to {OUT_DIR}")

# ----- Prepare test data and produce submission -----
print("[INFO] Loading test data and producing submission...")
test_df = pd.read_csv(TEST)
# infer id column if it exists
if "id" in test_df.columns:
    test_ids = test_df["id"]
    X_test_raw = test_df.drop(columns=["id"])
else:
    test_ids = np.arange(len(test_df))
    X_test_raw = test_df

if pre is not None:
    X_test = pre.transform(X_test_raw)
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()
else:
    X_test = X_test_raw.values

preds = clf.predict_proba(X_test)[:, 1]
sub = pd.DataFrame({"id": test_ids, "target": preds})
sub.to_csv(OUT_DIR / "submission_lgbm_tuned.csv", index=False)
print(f"[INFO] Submission saved to {OUT_DIR / '/submission_lgbm_tuned.csv'}")
