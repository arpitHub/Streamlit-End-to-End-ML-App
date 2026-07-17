import streamlit as st
from utils.state import get_state
from utils.sklearn_compare import run_sklearn_compare
from utils.timeseries_tools import (
    detect_time_column,
    default_value_column,
    is_time_series_dataset,
    run_time_series_forecast,
)

st.title("🤖 Model Builder (Light Mode Only)")

state = get_state()
df = state.get("df")

if df is None:
    st.warning("Please select a dataset first.")
    st.stop()

ts_available = is_time_series_dataset(df)

mode = "Standard ML"
if ts_available:
    mode = st.radio(
        "This dataset has a time column — choose how to model it",
        ["Time Series Forecast", "Standard ML (classification/regression)"],
        horizontal=True,
    )
    mode = "Time Series" if mode.startswith("Time Series") else "Standard ML"

if mode == "Time Series":
    time_col = detect_time_column(df)
    value_options = [c for c in df.select_dtypes(include="number").columns if c != time_col]

    if not value_options:
        st.warning("No numeric column available to forecast.")
        st.stop()

    default_col = default_value_column(df, time_col)
    value_col = st.selectbox(
        "Value column to forecast", value_options,
        index=value_options.index(default_col) if default_col in value_options else 0,
    )
    horizon = st.slider("Forecast horizon (periods into the future)", 1, 60, 10)

    if st.button("Run Time Series Forecast"):
        with st.spinner("Fitting forecasting models..."):
            try:
                result = run_time_series_forecast(df, time_col, value_col, forecast_periods=horizon)
                state["ts_result"] = result
                state["is_time_series"] = True
                state["sklearn_results"] = None
                state["best_model"] = None
            except ValueError as e:
                st.error(str(e))

    if state.get("ts_result") is not None:
        result = state["ts_result"]
        st.subheader("Forecasting Method Leaderboard")
        st.caption("Ranked by RMSE on a held-out chronological test window (lower is better).")
        st.dataframe(result["leaderboard"])
        st.success(f"Best method: **{result['best_model']}**. Go to Model Results page for the forecast chart and CSV.")

else:
    target = st.selectbox("Select target column", df.columns)
    state["target_column"] = target

    with st.expander("Advanced options"):
        cv_folds = st.slider("Cross-validation folds", 3, 10, 5)
        tune = st.checkbox(
            "Hyperparameter tuning (grid search — slower)",
            value=False,
            help="Searches a small parameter grid per model instead of using defaults.",
        )

    if st.button("Run Model Comparison"):
        with st.spinner("Training models..."):
            results, best_model, X_test, y_test, problem_type = run_sklearn_compare(
                df, target, cv_folds=cv_folds, tune=tune
            )
            state["sklearn_results"] = results
            state["best_model"] = best_model
            state["X_test"] = X_test
            state["y_test"] = y_test
            state["problem_type"] = problem_type
            state["perm_importance"] = None
            state["is_time_series"] = False
            state["ts_result"] = None

    if state.get("sklearn_results") is not None:
        st.subheader("Model Leaderboard")
        st.caption(
            "Models are ranked by mean cross-validation score on the training split "
            "(accuracy for classification, R² for regression). "
            "Test Score is measured on a 20% holdout the models never saw."
        )
        st.dataframe(state["sklearn_results"], hide_index=True)
        st.success("Best model saved. Go to Model Results page.")
