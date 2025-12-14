import streamlit as st
import pandas as pd

from utils.state import get_state

st.title("📈 Model Results")

state = get_state()

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
