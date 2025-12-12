import joblib
import lightgbm as lgb
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path to allow unpickling TabularPreprocessor
import sys
sys.path.append(str(Path.cwd()))

OUTPUTS_DIR = Path("outputs/models")
SHAP_DIR = Path("outputs/shap")
PREPROC_PATH = OUTPUTS_DIR / "preprocessor.joblib"
MODEL_PATH = OUTPUTS_DIR / "lgbm_fold5.txt"
SHAP_CSV = SHAP_DIR / "shap_feature_importance_mean_abs.csv"

def main():
    print(f"Loading preprocessor from {PREPROC_PATH}...")
    preproc = joblib.load(PREPROC_PATH)
    print("Preprocessor loaded.")
    
    # Extract input features
    input_features = []
    if hasattr(preproc, "num_cols"):
        input_features.extend(preproc.num_cols)
        input_features.extend(preproc.low_card_cols)
        input_features.extend(preproc.high_card_cols)
        # Date cols are handled separately in transform but are part of input
        if hasattr(preproc, "date_cols_present_"):
             input_features.extend(preproc.date_cols_present_)
        
        print(f"Found {len(input_features)} input features.")
        
        features_path = OUTPUTS_DIR / "input_features.json"
        with open(features_path, "w") as f:
            json.dump(input_features, f, indent=2)
        print(f"Saved input features to {features_path}")
    
    # Map output features
    try:
        out_names = preproc.get_feature_names_out()
        print(f"Preprocessor produces {len(out_names)} output features.")
        
        # Load SHAP importance
        if SHAP_CSV.exists():
            shap_df = pd.read_csv(SHAP_CSV)
            # shap_df has 'feature' (f0, f1...) and 'mean_abs_shap'
            
            # Create mapping: f0 -> out_names[0]
            mapping = {f"f{i}": name for i, name in enumerate(out_names)}
            
            # Add mapped name to dataframe
            shap_df["feature_name"] = shap_df["feature"].map(mapping)
            
            # Save mapped importance
            mapped_csv = SHAP_DIR / "shap_feature_importance_mapped.csv"
            shap_df.to_csv(mapped_csv, index=False)
            print(f"Saved mapped SHAP importance to {mapped_csv}")
            
            # Print top 10
            print("Top 10 Features:")
            print(shap_df.head(10)[["feature_name", "mean_abs_shap"]])
            
            # Save top 20 features for UI
            top_features = shap_df.head(20)["feature_name"].tolist()
            top_features_path = OUTPUTS_DIR / "top_features.json"
            with open(top_features_path, "w") as f:
                json.dump(top_features, f, indent=2)
            print(f"Saved top 20 features to {top_features_path}")
            
    except Exception as e:
        print(f"Error mapping features: {e}")

    # Save final model as joblib
    print(f"Loading model from {MODEL_PATH}...")
    model = lgb.Booster(model_file=str(MODEL_PATH))
    final_model_path = OUTPUTS_DIR / "final_model.joblib"
    joblib.dump(model, final_model_path)
    print(f"Saved final model to {final_model_path}")

if __name__ == "__main__":
    main()
