import streamlit as st
from utils.data_loader import (
    list_builtin_datasets,
    load_builtin_dataset,
    list_pydatasets,
    load_pydataset,
    load_uploaded_file,
)
from utils.state import get_state
from utils.timeseries_tools import is_time_series_dataset

st.title("📊 Dataset Explorer")

state = get_state()

source = st.sidebar.radio(
    "Choose data source",
    ["Built-in datasets", "pydataset library", "Upload your own"],
)

df = None

if source == "Built-in datasets":
    name = st.sidebar.selectbox("Select dataset", list_builtin_datasets())
    df = load_builtin_dataset(name)
    state["dataset_name"] = name

elif source == "pydataset library":
    datasets = list_pydatasets()
    if datasets:  # only show dropdown if list is not empty
        name = st.sidebar.selectbox("Select a pydataset", datasets)
        if name:
            df = load_pydataset(name)
            state["dataset_name"] = name
    else:
        st.sidebar.warning("No datasets found in pydataset.")


else:
    file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = load_uploaded_file(file)
        state["dataset_name"] = file.name

if df is not None:
    if state.get("dataset_name") != state.get("_last_loaded_dataset_name"):
        state["target_column"] = None
        state["sklearn_results"] = None
        state["best_model"] = None
        state["X_test"] = None
        state["y_test"] = None
        state["predictions"] = None
        state["is_time_series"] = False
        state["ts_result"] = None
        state["_last_loaded_dataset_name"] = state.get("dataset_name")

    state["df"] = df

    st.subheader("Preview")
    st.dataframe(df.head())

    st.subheader("Shape")
    st.write(df.shape)

    st.subheader("Column Types")
    st.write(df.dtypes.astype(str))

    if is_time_series_dataset(df):
        st.info(
            "This looks like a time series dataset (it has a 'time' column). "
            "Head to **Model Builder** to run a time series forecast on it."
        )
else:
    st.info("Please select or upload a dataset.")
