import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.state import get_state

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

    if model is None:
        st.warning("Please run model comparison first.")
        st.stop()

    preds = model.predict(X_test)

    st.subheader("Predictions")
    st.write(preds[:10])

    st.subheader("Download Predictions")
    df_pred = pd.DataFrame({"Actual": y_test, "Predicted": preds})
    csv = df_pred.to_csv(index=False).encode("utf-8")

    st.download_button("Download CSV", csv, "predictions.csv", "text/csv")
