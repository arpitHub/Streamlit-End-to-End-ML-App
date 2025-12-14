import pandas as pd
import plotly.express as px

def get_basic_stats(df: pd.DataFrame):
    return df.describe(include="all").transpose()

def get_corr_heatmap(df: pd.DataFrame):
    num_df = df.select_dtypes(include=["number"])
    if num_df.shape[1] < 2:
        return None
    corr = num_df.corr()
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        aspect="auto",
    )
    return fig
