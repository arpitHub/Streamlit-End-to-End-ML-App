import streamlit as st
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

from utils.state import get_state
from utils.eda_tools import get_basic_stats, get_corr_heatmap

st.title("🔍 EDA Dashboard")

state = get_state()
df = state.get("df")

if df is None:
    st.warning("Please select a dataset first.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Quick Stats", "Correlation", "Full Profiling"])

with tab1:
    st.dataframe(get_basic_stats(df))

with tab2:
    fig = get_corr_heatmap(df)
    if fig:
        st.plotly_chart(fig)
    else:
        st.info("Not enough numeric columns.")

with tab3:
    if st.button("Generate Profiling Report"):
        profile = ProfileReport(df, explorative=True)
        st_profile_report(profile)
