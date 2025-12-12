# app_shap_streamlit.py
import streamlit as st
from PIL import Image
from pathlib import Path
import joblib, numpy as np, pandas as pd
import shap

SAVE_DIR = Path("outputs/shap")

st.set_page_config(layout="wide", page_title="SHAP Explainability")
st.title("SHAP Explainability — System Thread Forecaster")

st.header("Global explanations")
cols = st.columns(2)
with cols[0]:
    p = SAVE_DIR / "shap_lgb_summary.png"
    if p.exists(): st.image(Image.open(p), caption="LGB SHAP summary", use_column_width=True)
    else: st.write("No global summary image found:", p)
with cols[1]:
    p2 = SAVE_DIR / "shap_top30_bar.png"
    if p2.exists(): st.image(Image.open(p2), caption="Top features (mean |SHAP|)", use_column_width=True)
    else: st.write("No top-features bar found:", p2)

st.header("Local explanation (force plot)")
idx = st.number_input("Sample index (within saved SHAP array)", min_value=0, max_value=100000, value=0, step=1)
force_html = SAVE_DIR / f"shap_force_sample_{idx}.html"
if force_html.exists():
    st.components.v1.html(force_html.read_text(), height=450)
else:
    st.write("Force plot HTML not found for that index. Generate it with shap_local_force.py and set SAMPLE_IDX accordingly.")
