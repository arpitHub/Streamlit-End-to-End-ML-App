import streamlit as st

st.set_page_config(
    page_title="End-to-End ML App",
    page_icon="🤖",
    layout="wide",
)

st.title("End-to-End Machine Learning App 🤖")

st.markdown(
    """
Welcome!

This app lets you:

1. **Choose or upload a dataset**
2. **Explore it visually and statistically**
3. **Build and compare machine learning models (Light Mode)**
4. **Inspect results and download predictions**

Use the navigation menu on the left to get started.

### Pages:
- **Dataset Explorer** → Pick a dataset (built‑in, pydataset, or CSV)
- **EDA Dashboard** → Summary stats, correlation, profiling
- **Model Builder** → Lightweight sklearn model comparison
- **Model Results** → Predictions + CSV download
"""
)
