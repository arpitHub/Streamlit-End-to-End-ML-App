import streamlit as st

def get_state():
    if "app_state" not in st.session_state:
        st.session_state.app_state = {
            "dataset_name": None,
            "df": None,
            "target_column": None,
            "sklearn_results": None,
            "best_model": None,
            "X_test": None,
            "y_test": None,
            "predictions": None,
            "_last_loaded_dataset_name": None,
        }
    return st.session_state.app_state
