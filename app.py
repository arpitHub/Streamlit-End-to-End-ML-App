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
3. **Build and compare machine learning models — or forecast time series data**
4. **Inspect results, feature importance, and download predictions**

Use the navigation menu on the left to get started.

### Pages:
- **Dataset Explorer** → Pick a dataset (built‑in, pydataset with 750+ datasets, or your own CSV)
- **EDA Dashboard** → Overview metrics, distributions, relationships, correlation, and full profiling
- **Model Builder** → Cross-validated scikit-learn model comparison with optional hyperparameter
  tuning, or automatic time series forecasting for time-indexed datasets
- **Model Results** → Metrics, confusion matrix / residual plots, feature importance, and CSV download
"""
)
