import streamlit as st
from utils.data_loader import (
    list_builtin_datasets,
    load_builtin_dataset,
    list_pydatasets,
    load_pydataset,
    load_uploaded_file,
)
from utils.state import get_state

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
    state["df"] = df

    st.subheader("Preview")
    st.dataframe(df.head())

    st.subheader("Shape")
    st.write(df.shape)

    st.subheader("Column Types")
    st.write(df.dtypes)
else:
    st.info("Please select or upload a dataset.")
