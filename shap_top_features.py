# shap_top_features.py
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

SAVE_DIR = Path("outputs/shap")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# load shap and feature names
shap_vals = np.load(SAVE_DIR / "shap_lgb_values.npy")  # shape: (n_samples, n_features)
# feature names saved? Try to load from CSV used to produce shap summary
# We'll attempt to read the DataFrame used: if present, there may be a file with 'feature_names' or we use generic f0..fN-1
# Try to infer names from the saved summary plotting DataFrame (not saved). Fallback to generic names.
n_feats = shap_vals.shape[1]
try:
    # try to reload the dataframe used earlier if present (not mandatory)
    import joblib
    expl = joblib.load(SAVE_DIR / "shap_lgb_explainer.joblib")
    # expl.expected_value etc. but not feature names; fallback
    feature_names = [f"f{i}" for i in range(n_feats)]
except Exception:
    feature_names = [f"f{i}" for i in range(n_feats)]

# compute mean absolute SHAP per feature
mean_abs = np.mean(np.abs(shap_vals), axis=0)
df_feat = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
df_feat = df_feat.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
df_feat.to_csv(SAVE_DIR / "shap_feature_importance_mean_abs.csv", index=False)

# plot top 30
topk = min(30, len(df_feat))
plt.figure(figsize=(8, max(4, topk*0.25)))
plt.barh(df_feat["feature"].iloc[:topk][::-1], df_feat["mean_abs_shap"].iloc[:topk][::-1])
plt.xlabel("Mean |SHAP value|")
plt.title(f"Top {topk} features by mean |SHAP| (LightGBM)")
plt.tight_layout()
plt.savefig(SAVE_DIR / f"shap_top{topk}_bar.png", dpi=150)
plt.close()

print("Saved:", SAVE_DIR / "shap_feature_importance_mean_abs.csv")
print("Saved:", SAVE_DIR / f"shap_top{topk}_bar.png")
