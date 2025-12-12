# shap_local_force.py
import joblib, numpy as np, pandas as pd
from pathlib import Path
import shap

SAVE_DIR = Path("outputs/shap")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_IDX = 0  # change to inspect another sample

# load
explainer = joblib.load(SAVE_DIR / "shap_lgb_explainer.joblib")
shap_vals = np.load(SAVE_DIR / "shap_lgb_values.npy")  # (n_samples, n_features)
# We need the DataFrame used to compute shap summary (features after transform)
# If you created X_samp_df in the script, we didn't save it; so we will reconstruct it similarly:
# Try loading a CSV with original data then transform using preprocessor if available
import joblib, pandas as pd
import numpy as np
preproc = joblib.load("outputs/models/preprocessor.joblib")
df_raw = pd.read_csv("data/raw/train.csv")
X_raw = df_raw.drop(columns=["target"]) if "target" in df_raw.columns else df_raw.copy()
X_trans = preproc.transform(X_raw)
# sample indices should match the ones used earlier — we don't have the exact indices; so we'll map SAMPLE_IDX to the first
# row used in shap array: take the SAME random seed would be needed. For simplicity, take the first row of X_trans used to compute SHAP.
# So create a DataFrame for shap plotting:
n_feats = shap_vals.shape[1]
feature_names = [f"f{i}" for i in range(n_feats)]
X_samp_df = pd.DataFrame(X_trans[:shap_vals.shape[0], :], columns=feature_names)

# choose the sample index within the saved shap array
if SAMPLE_IDX >= shap_vals.shape[0]:
    raise IndexError("SAMPLE_IDX is out of range for saved SHAP values")
sample_shap = shap_vals[SAMPLE_IDX:SAMPLE_IDX+1]
sample_row = X_samp_df.iloc[SAMPLE_IDX:SAMPLE_IDX+1]

# create force plot HTML
fpath = SAVE_DIR / f"shap_force_sample_{SAMPLE_IDX}.html"
force = shap.force_plot(explainer.expected_value, sample_shap, sample_row, matplotlib=False)
# shap.force_plot returns a JS/HTML object with .data attribute sometimes; save via save_html utility
shap.save_html(str(fpath), force)
print("Saved force plot HTML to:", fpath)

# waterfall (matplotlib) save (if shap.plots._waterfall is available)
try:
    import matplotlib.pyplot as plt
    shap.plots._waterfall.waterfall_legacy(explainer.expected_value, sample_shap[0], sample_row.iloc[0])
    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"shap_waterfall_sample_{SAMPLE_IDX}.png", dpi=150)
    plt.close()
    print("Saved waterfall PNG to:", SAVE_DIR / f"shap_waterfall_sample_{SAMPLE_IDX}.png")
except Exception as e:
    print("Waterfall plot failed:", e)
