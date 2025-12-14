import pandas as pd
import streamlit as st
from sklearn import datasets
from pydataset import data as pydata

def list_builtin_datasets():
    return ["Iris", "Wine", "Breast Cancer"]

@st.cache_data
def load_builtin_dataset(name: str) -> pd.DataFrame:
    if name == "Iris":
        return datasets.load_iris(as_frame=True).frame
    elif name == "Wine":
        return datasets.load_wine(as_frame=True).frame
    elif name == "Breast Cancer":
        return datasets.load_breast_cancer(as_frame=True).frame

@st.cache_data
def list_pydatasets():
    """
    Returns a sorted list of all pydataset dataset names.
    Uses the stable 'datasets' index instead of pydata().
    """
    try:
        df = pydata("datasets")  # official way to list datasets
        return sorted(df["dataset"].tolist())
    except Exception:
        return []

@st.cache_data
def load_pydataset(name: str) -> pd.DataFrame:
    return pydata(name)

@st.cache_data
def load_uploaded_file(uploaded_file):
    return pd.read_csv(uploaded_file)
