#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
stack_path = Path("outputs/stacking_v2/submission_stacked.csv")
cat_path   = Path("outputs/catboost_full/submission_cat.csv")
lgb_path   = Path("outputs/final/submission_lgbm_tuned.csv")
xgb_path   = Path("outputs/xgb_full/submission_xgb.csv")

out_path   = Path("outputs/final_ensemble")
out_path.mkdir(parents=True, exist_ok=True)

print("[INFO] Loading submissions...")
s_stack = pd.read_csv(stack_path)["target"].values
s_cat   = pd.read_csv(cat_path)["target"].values
s_lgb   = pd.read_csv(lgb_path)["target"].values
s_xgb   = pd.read_csv(xgb_path)["target"].values

# Ensure same ordering
df_test = pd.read_csv("data/raw/test.csv")
ids = df_test.index.values

# Weights (initial guess — good baseline)
w_stack = 0.50
w_cat   = 0.30
w_lgb   = 0.15
w_xgb   = 0.05

print("[INFO] Blending predictions...")
final_pred = (
    w_stack * s_stack +
    w_cat   * s_cat   +
    w_lgb   * s_lgb   +
    w_xgb   * s_xgb
)

# Save final submission
sub = pd.DataFrame({"id": ids, "target": final_pred})
sub.to_csv(out_path / "submission_final_blend.csv", index=False)

print("[INFO] Saved final submission to:", out_path / "submission_final_blend.csv")
print("[DONE]")
