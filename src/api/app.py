# src/api/app.py
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from pathlib import Path
import joblib, pandas as pd, numpy as np, lightgbm as lgb, os, json

# Add src to path to allow unpickling TabularPreprocessor
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent))

SHAP_DIR = ROOT.parent / "outputs" / "shap"
MODEL_DIR = ROOT.parent / "outputs" / "models"

# Adjust these filenames if your final model / preproc are named differently
PREPROC_FILE = MODEL_DIR / "preprocessor.joblib"
MODEL_FILE   = MODEL_DIR / "final_model.joblib"
FEATURES_FILE = MODEL_DIR / "input_features.json"

app = Flask(__name__)
CORS(app)

# load preprocessor & model
if not PREPROC_FILE.exists():
    raise FileNotFoundError(f"Preprocessor not found: {PREPROC_FILE}")
preproc = joblib.load(PREPROC_FILE)

# Load LightGBM booster or joblib model
if not MODEL_FILE.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_FILE}")
model = joblib.load(MODEL_FILE)

# Load expected features
if not FEATURES_FILE.exists():
    raise FileNotFoundError(f"Features file not found: {FEATURES_FILE}")
with open(FEATURES_FILE, "r") as f:
    EXPECTED_FEATURES = json.load(f)

def to_dataframe(features):
    """
    Accepts:
     - dict of single sample {feat: val, ...}
     - list of dicts [{...}, {...}]
    Returns pandas.DataFrame aligned with training features (drop unknown columns, fill missing with NaN)
    """
    if isinstance(features, dict):
        df_input = pd.DataFrame([features])
    elif isinstance(features, list):
        df_input = pd.DataFrame(features)
    else:
        raise ValueError("features must be dict or list of dicts")
    
    # Reindex to match training features exactly
    # This ensures all expected columns exist (filled with NaN if missing)
    # and extra columns are dropped
    df_aligned = df_input.reindex(columns=EXPECTED_FEATURES)
    
    return df_aligned

def transform_and_predict(df_raw):
    # Drop target if present (shouldn't be, but just in case)
    if "target" in df_raw.columns:
        df_raw = df_raw.drop(columns=["target"])
        
    X = preproc.transform(df_raw)
    
    # predict_proba style via LightGBM
    if isinstance(model, lgb.Booster):
        probs = model.predict(X)  # returns float prob for positive class
    else:
        # Check if it has predict_proba
        if hasattr(model, "predict_proba"):
             probs = model.predict_proba(X)[:, 1]
        else:
             probs = model.predict(X)
             
    return probs

def risk_category(p):
    if p >= 0.75: return "High"
    if p >= 0.4: return "Medium"
    return "Low"

@app.route("/ping")
def ping():
    return jsonify({"status": "ok", "model_loaded": True})

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Accepts JSON payload:
    { "instances": [ {feat1: val, feat2: val, ...}, {...} ] }
    or single:
    { "instance": { ... } }
    Returns:
    { predictions: [{probability:0.7, risk:"High"}, ...], n: int }
    """
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "invalid JSON"}), 400
        
    if payload is None:
        return jsonify({"error": "invalid JSON"}), 400

    if "instance" in payload:
        instances = [payload["instance"]]
    elif "instances" in payload:
        instances = payload["instances"]
    else:
        # allow top-level dict to be a single instance
        if isinstance(payload, dict):
            instances = [payload]
        else:
            return jsonify({"error": "provide 'instance' or 'instances' in JSON"}), 400

    try:
        df = to_dataframe(instances)
        probs = transform_and_predict(df)
    except Exception as e:
        return jsonify({"error": f"preprocessing/prediction error: {e}"}), 500

    out = []
    for p in np.asarray(probs).tolist():
        out.append({"probability": float(p), "risk": risk_category(p)})
    return jsonify({"n": len(out), "predictions": out})

@app.route("/api/shap/global", methods=["GET"])
def shap_global():
    p = SHAP_DIR / "shap_lgb_summary.png"
    if not p.exists():
        return jsonify({"error": "global SHAP image not found"}), 404
    return send_file(str(p), mimetype="image/png")

@app.route("/api/shap/force/<int:idx>", methods=["GET"])
def shap_force(idx):
    html = SHAP_DIR / f"shap_force_sample_{idx}.html"
    if not html.exists():
        return jsonify({"error": "force plot not found for index"}), 404
    return send_file(str(html), mimetype="text/html")

@app.route("/api/shap/top_features", methods=["GET"])
def shap_top_features():
    csvp = SHAP_DIR / "shap_feature_importance_mean_abs.csv"
    if not csvp.exists():
        return jsonify({"error":"shap CSV missing"}), 404
    n = int(request.args.get("n", 20))
    df = pd.read_csv(csvp)
    return df.head(n).to_json(orient="records")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001)) # Changed default port to 5001 to avoid conflicts
    app.run(host="0.0.0.0", port=port)
