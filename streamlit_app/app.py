import streamlit as st
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Alzheimer’s Detection", layout="wide")

st.title(" Deep Learning & Bayesian Modeling for Early Alzheimer’s Detection")

st.subheader("Project Overview")
st.write("""
This app presents our DS 4420 project results:
1️⃣ CNN classification of MRI images  
2️⃣ Bayesian logistic regression on ROI features  
""")

st.image("streamlit_app/assets/mri_example.png", caption="Sample MRI input")

df = pd.read_json("results/cnn_metrics.json")
st.dataframe(df)

if st.button("Show ROC Curve"):
    st.image("results/roc_curve.png", caption="CNN ROC Curve")
