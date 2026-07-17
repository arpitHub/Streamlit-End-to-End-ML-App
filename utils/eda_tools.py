import numpy as np
import pandas as pd
import plotly.express as px

def get_basic_stats(df: pd.DataFrame):
    return df.describe(include="all").transpose()

def get_overview_metrics(df: pd.DataFrame) -> dict:
    n_cells = df.shape[0] * df.shape[1]
    n_missing = int(df.isna().sum().sum())
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "numeric_columns": df.select_dtypes(include="number").shape[1],
        "categorical_columns": df.select_dtypes(exclude="number").shape[1],
        "missing_cells": n_missing,
        "missing_pct": (100.0 * n_missing / n_cells) if n_cells else 0.0,
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_kb": df.memory_usage(deep=True).sum() / 1024,
    }

def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "Column": missing.index.astype(str),
        "Missing": missing.values,
        "Missing %": (100.0 * missing.values / len(df)).round(2),
    })

def get_numeric_columns(df: pd.DataFrame):
    return df.select_dtypes(include="number").columns.tolist()

def get_categorical_columns(df: pd.DataFrame, max_unique: int = 30):
    """Columns with few enough unique values to be usable for grouping/coloring."""
    return [c for c in df.columns if df[c].nunique() <= max_unique]

def get_histogram(df: pd.DataFrame, column: str, color: str = None, bins: int = 30):
    fig = px.histogram(
        df, x=column, color=color, nbins=bins, marginal="box", opacity=0.75,
    )
    fig.update_layout(bargap=0.05)
    return fig

def get_value_counts_bar(df: pd.DataFrame, column: str, top_n: int = 20):
    counts = df[column].astype(str).value_counts().head(top_n).reset_index()
    counts.columns = [column, "Count"]
    fig = px.bar(counts, x=column, y="Count")
    fig.update_layout(xaxis_title=column, yaxis_title="Count")
    return fig

def get_box_by_category(df: pd.DataFrame, num_col: str, cat_col: str):
    return px.box(df, x=cat_col, y=num_col, points="outliers")

def get_scatter(df: pd.DataFrame, x: str, y: str, color: str = None):
    return px.scatter(df, x=x, y=y, color=color, opacity=0.7, trendline=None)

def get_line_over_time(df: pd.DataFrame, time_col: str, value_col: str):
    data = df[[time_col, value_col]].dropna().sort_values(time_col)
    return px.line(data, x=time_col, y=value_col)

def get_corr_heatmap(df: pd.DataFrame, method: str = "pearson"):
    num_df = df.select_dtypes(include=["number"])
    if num_df.shape[1] < 2:
        return None
    corr = num_df.corr(method=method)
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    return fig

def get_top_correlations(df: pd.DataFrame, method: str = "pearson", top_n: int = 10) -> pd.DataFrame:
    num_df = df.select_dtypes(include=["number"])
    if num_df.shape[1] < 2:
        return pd.DataFrame()
    corr = num_df.corr(method=method)
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    stacked = upper.stack().reset_index()
    stacked.columns = ["Column A", "Column B", "Correlation"]
    stacked["|r|"] = stacked["Correlation"].abs()
    stacked = stacked.sort_values("|r|", ascending=False).drop(columns="|r|").head(top_n)
    stacked["Correlation"] = stacked["Correlation"].round(3)
    return stacked.reset_index(drop=True)
