import streamlit as st
from utils.state import get_state
from utils.sklearn_compare import run_sklearn_compare

st.title("🤖 Model Builder (Light Mode Only)")

state = get_state()
df = state.get("df")

if df is None:
    st.warning("Please select a dataset first.")
    st.stop()

target = st.selectbox("Select target column", df.columns)
state["target_column"] = target

if st.button("Run Model Comparison"):
    with st.spinner("Training models..."):
        results, best_model, X_test, y_test = run_sklearn_compare(df, target)
        state["sklearn_results"] = results
        state["best_model"] = best_model
        state["X_test"] = X_test
        state["y_test"] = y_test

if state.get("sklearn_results") is not None:
    st.subheader("Model Leaderboard")
    st.dataframe(state["sklearn_results"])
    st.success("Best model saved. Go to Model Results page.")
