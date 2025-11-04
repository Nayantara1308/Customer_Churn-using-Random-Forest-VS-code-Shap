
import streamlit as st
import pandas as pd
import os

OUT_DIR = 'outputs/shap_safe'
imp = pd.read_csv(os.path.join(OUT_DIR, 'shap_feature_importance.csv'))

st.title("🔍 SHAP Feature Importance Viewer")
st.bar_chart(imp.set_index('feature').head(20))
st.dataframe(imp.head(50))
