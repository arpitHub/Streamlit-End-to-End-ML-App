import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.state import get_state
from utils.model_eval import (
    get_actual_vs_predicted_fig,
    get_classification_metrics,
    get_classification_report_df,
    get_confusion_matrix_fig,
    get_importance_fig,
    get_permutation_importance_df,
    get_regression_metrics,
    get_residuals_fig,
)

st.title("📈 Model Results")

state = get_state()

if state.get("is_time_series") and state.get("ts_result") is not None:
    result = state["ts_result"]
    time_col = result["time_col"]
    value_col = result["value_col"]
    history = result["history"]
    test_preds = result["test_predictions"]
    forecast = result["forecast"]

    st.subheader(f"Forecast — {value_col}")
    st.caption(f"Best method: **{result['best_model']}**")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history[time_col], y=history["Actual"], mode="lines", name="History", line=dict(color="steelblue"),
    ))
    fig.add_trace(go.Scatter(
        x=test_preds[time_col], y=test_preds["Predicted"], mode="lines", name="Backtest (held-out)",
        line=dict(color="orange", dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=forecast[time_col], y=forecast[value_col], mode="lines+markers", name="Forecast",
        line=dict(color="firebrick", dash="dash"),
    ))
    fig.update_layout(xaxis_title=time_col, yaxis_title=value_col, legend=dict(orientation="h"))
    st.plotly_chart(fig)

    st.subheader("Backtest: Actual vs Predicted")
    st.dataframe(test_preds)

    st.subheader("Future Forecast")
    st.dataframe(forecast)

    csv = forecast.to_csv(index=False).encode("utf-8")
    st.download_button("Download Forecast CSV", csv, "forecast.csv", "text/csv")

else:
    model = state.get("best_model")
    X_test = state.get("X_test")
    y_test = state.get("y_test")
    problem_type = state.get("problem_type")

    if model is None:
        st.warning("Please run model comparison first.")
        st.stop()

    preds = model.predict(X_test)

    if problem_type == "classification":
        metrics = get_classification_metrics(y_test, preds)
        st.metric("Test Accuracy", f"{metrics['accuracy']:.3f}")

        st.subheader("Confusion Matrix")
        st.plotly_chart(get_confusion_matrix_fig(y_test, preds))

        st.subheader("Per-class Precision / Recall / F1")
        st.dataframe(get_classification_report_df(y_test, preds), hide_index=True)

    else:
        metrics = get_regression_metrics(y_test, preds)
        c1, c2, c3 = st.columns(3)
        c1.metric("R²", f"{metrics['r2']:.3f}")
        c2.metric("RMSE", f"{metrics['rmse']:.3f}")
        c3.metric("MAE", f"{metrics['mae']:.3f}")

        st.subheader("Actual vs Predicted")
        st.plotly_chart(get_actual_vs_predicted_fig(y_test, preds))

        st.subheader("Residual Distribution")
        st.plotly_chart(get_residuals_fig(y_test, preds))

    st.subheader("Feature Importance")
    st.caption(
        "Permutation importance on the test set: how much the model's score drops "
        "when each column's values are shuffled."
    )
    if st.button("Compute Feature Importance"):
        with st.spinner("Computing permutation importance..."):
            state["perm_importance"] = get_permutation_importance_df(model, X_test, y_test)
    if state.get("perm_importance") is not None:
        st.plotly_chart(get_importance_fig(state["perm_importance"]))

    st.subheader("Download Predictions")
    df_pred = pd.DataFrame({"Actual": y_test, "Predicted": preds})
    st.dataframe(df_pred.head(10), hide_index=True)
    csv = df_pred.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "predictions.csv", "text/csv")
