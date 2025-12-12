# src/ml/ensemble_blend.py
"""
Blend saved CatBoost and LightGBM fold models using OOF weights.
Produces:
 - outputs/ensemble/oof_blend.csv
 - outputs/ensemble/submission_blend.csv
 - prints blended OOF AUC
Assumes:
 - LightGBM fold models at outputs/models/lgbm_fold{1..k}.txt
 - CatBoost fold models at outputs/catboost/catboost_fold{1..k}.cbm  (or outputs/catboost)
 - Preprocessor saved at outputs/models/preprocessor.joblib (or outputs/preprocessor_cb.joblib)
Adjust paths if needed.
"""

# src/ml/ensemble_blend.py
import sys
from pathlib import Path
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from catboost import CatBoostClassifier
# ... rest of file


# ---- CONFIG ----
OUT_DIR = Path("outputs")
LGB_MODELS_DIR = OUT_DIR / "models"          # lgbm_fold*.txt
CAT_MODELS_DIR = OUT_DIR / "catboost"        # catboost_fold*.cbm (if you used outputs/catboost)
PREPROC_PATHS = [
    OUT_DIR / "models" / "preprocessor.joblib",
    OUT_DIR / "preprocessor_cb.joblib",      # fallback
    OUT_DIR / "models" / "preprocessor_cb.joblib"
]
TRAIN_PATH = Path("data/raw/train.csv")
TEST_PATH = Path("data/raw/test.csv")
ENSEMBLE_OUT = OUT_DIR / "ensemble"
ENSEMBLE_OUT.mkdir(parents=True, exist_ok=True)

# ---- helper load preprocessor ----
preproc = None
for p in PREPROC_PATHS:
    if p.exists():
        preproc = joblib.load(p)
        print("Loaded preprocessor:", p)
        break
if preproc is None:
    raise FileNotFoundError("No preprocessor.joblib found in expected locations.")

# ---- load data ----
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
y = train["target"].astype(int).values

# ---- transform ----
X = preproc.transform(train)
X_test = preproc.transform(test)

# ---- discover saved model files ----
lgb_paths = sorted([p for p in (LGB_MODELS_DIR).glob("lgbm_fold*.txt")])
cat_paths = sorted([p for p in (CAT_MODELS_DIR).glob("catboost_fold*.cbm")]) \
            or sorted([p for p in (OUT_DIR / "catboost").glob("catboost_fold*.cbm")])

if len(lgb_paths) == 0 and len(cat_paths) == 0:
    raise FileNotFoundError("No model files found. Check paths LGB_MODELS_DIR and CAT_MODELS_DIR.")

print("Found LGB models:", lgb_paths)
print("Found CatBoost models:", cat_paths)

# ---- helper to get OOF preds per-method ----
def oof_preds_from_lightgbm(X, y, model_paths):
    from sklearn.model_selection import StratifiedKFold
    n = X.shape[0]
    oof = np.zeros(n)
    k = len(model_paths)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    for (fold_idx, (tr, va)), mpath in zip(enumerate(skf.split(X, y)), model_paths):
        booster = lgb.Booster(model_file=str(mpath))
        oof[va] = booster.predict(X[va], num_iteration=booster.best_iteration)
    return oof

def oof_preds_from_catboost(X, y, model_paths):
    from sklearn.model_selection import StratifiedKFold
    n = X.shape[0]
    oof = np.zeros(n)
    k = len(model_paths)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    for (fold_idx, (tr, va)), mpath in zip(enumerate(skf.split(X, y)), model_paths):
        model = CatBoostClassifier()
        model.load_model(str(mpath))
        oof[va] = model.predict_proba(X[va])[:,1]
    return oof

# ---- create OOF for each model family (if models exist) ----
oof_dict = {}
if len(lgb_paths) > 0:
    print("Computing LGB OOF...")
    oof_lgb = oof_preds_from_lightgbm(X, y, lgb_paths)
    oof_dict['lgb'] = oof_lgb
    print("LGB OOF AUC:", roc_auc_score(y, oof_lgb))

if len(cat_paths) > 0:
    print("Computing CatBoost OOF...")
    oof_cat = oof_preds_from_catboost(X, y, cat_paths)
    oof_dict['cat'] = oof_cat
    print("CatBoost OOF AUC:", roc_auc_score(y, oof_cat))

# ---- determine weights automatically from OOF AUCs ----
aucs = {k: float(roc_auc_score(y, v)) for k, v in oof_dict.items()}
print("Per-model OOF AUCs:", aucs)
# weight = auc / sum(aucs)
total = sum(aucs.values())
weights = {k: (aucs[k] / total) for k in aucs}
print("Auto blend weights:", weights)

# ---- compute blended OOF and final AUC ----
blend_oof = np.zeros_like(next(iter(oof_dict.values())))
for k, arr in oof_dict.items():
    blend_oof += arr * weights[k]
blend_auc = roc_auc_score(y, blend_oof)
print("Blended OOF AUC:", blend_auc)

# save oof
pd.DataFrame({"target": y, "oof_blend": blend_oof}).to_csv(ENSEMBLE_OUT / "oof_blend.csv", index=False)
print("Saved OOF blend to", ENSEMBLE_OUT / "oof_blend.csv")

# ---- produce blended test preds (average across folds per model, then weighted) ----
test_preds = np.zeros(X_test.shape[0])
model_counts = 0

if len(lgb_paths) > 0:
    # average LGB predictions across fold models
    preds_lgb = np.zeros(X_test.shape[0])
    for p in lgb_paths:
        booster = lgb.Booster(model_file=str(p))
        preds_lgb += booster.predict(X_test, num_iteration=booster.best_iteration)
    preds_lgb /= len(lgb_paths)
    test_preds += preds_lgb * weights.get('lgb', 0.0)
    model_counts += 1

if len(cat_paths) > 0:
    preds_cat = np.zeros(X_test.shape[0])
    for p in cat_paths:
        m = CatBoostClassifier()
        m.load_model(str(p))
        preds_cat += m.predict_proba(X_test)[:,1]
    preds_cat /= len(cat_paths)
    test_preds += preds_cat * weights.get('cat', 0.0)
    model_counts += 1

# save submission
submission = pd.DataFrame({
    "MachineID": test["MachineID"] if "MachineID" in test.columns else test.index,
    "target": test_preds
})
submission_path = ENSEMBLE_OUT / "submission_blend.csv"
submission.to_csv(submission_path, index=False)
print("Saved blended submission to", submission_path)
