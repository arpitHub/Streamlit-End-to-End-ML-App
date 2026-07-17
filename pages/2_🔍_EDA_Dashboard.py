import streamlit as st
import streamlit.components.v1 as components
from ydata_profiling import ProfileReport

from utils.state import get_state
from utils.eda_tools import (
    get_basic_stats,
    get_box_by_category,
    get_categorical_columns,
    get_corr_heatmap,
    get_histogram,
    get_line_over_time,
    get_missing_summary,
    get_numeric_columns,
    get_overview_metrics,
    get_scatter,
    get_top_correlations,
    get_value_counts_bar,
)
from utils.timeseries_tools import detect_time_column, is_time_series_dataset

st.title("🔍 EDA Dashboard")

state = get_state()
df = state.get("df")

if df is None:
    st.warning("Please select a dataset first.")
    st.stop()

st.caption(f"Dataset: **{state.get('dataset_name') or 'unnamed'}**")

tab_overview, tab_dist, tab_rel, tab_corr, tab_profile = st.tabs(
    ["Overview", "Distributions", "Relationships", "Correlation", "Full Profiling"]
)

numeric_cols = get_numeric_columns(df)
cat_cols = get_categorical_columns(df)

with tab_overview:
    m = get_overview_metrics(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{m['rows']:,}")
    c2.metric("Columns", m["columns"], help="Total number of columns")
    c3.metric("Missing cells", f"{m['missing_cells']:,} ({m['missing_pct']:.1f}%)")
    c4.metric("Duplicate rows", f"{m['duplicate_rows']:,}")
    st.caption(
        f"{m['numeric_columns']} numeric · {m['categorical_columns']} non-numeric columns · "
        f"{m['memory_kb']:.1f} KB in memory"
    )

    st.subheader("Summary Statistics")
    st.dataframe(get_basic_stats(df))

    missing = get_missing_summary(df)
    st.subheader("Missing Values")
    if missing.empty:
        st.success("No missing values in this dataset.")
    else:
        st.dataframe(missing, hide_index=True)

with tab_dist:
    if not len(df.columns):
        st.info("No columns to plot.")
    else:
        col = st.selectbox("Column", df.columns, key="dist_col")
        if col in numeric_cols:
            color_options = ["(none)"] + [c for c in cat_cols if c != col]
            color = st.selectbox("Group by (optional)", color_options, key="dist_color")
            color = None if color == "(none)" else color
            st.plotly_chart(get_histogram(df, col, color=color))
            if color:
                st.plotly_chart(get_box_by_category(df, col, color))
        else:
            st.plotly_chart(get_value_counts_bar(df, col))
            st.caption("Showing the 20 most frequent values.")

with tab_rel:
    if is_time_series_dataset(df):
        time_col = detect_time_column(df)
        ts_values = [c for c in numeric_cols if c != time_col]
        if ts_values:
            st.subheader("Value over time")
            value_col = st.selectbox("Value column", ts_values, key="rel_ts_value")
            st.plotly_chart(get_line_over_time(df, time_col, value_col))
            st.divider()

    if len(numeric_cols) < 2:
        st.info("Need at least two numeric columns for a scatter plot.")
    else:
        c1, c2, c3 = st.columns(3)
        x = c1.selectbox("X axis", numeric_cols, index=0, key="rel_x")
        y = c2.selectbox("Y axis", numeric_cols, index=1, key="rel_y")
        color_options = ["(none)"] + [c for c in cat_cols if c not in (x, y)]
        color = c3.selectbox("Color by (optional)", color_options, key="rel_color")
        color = None if color == "(none)" else color
        st.plotly_chart(get_scatter(df, x, y, color=color))

with tab_corr:
    if len(numeric_cols) < 2:
        st.info("Not enough numeric columns.")
    else:
        method = st.radio(
            "Correlation method", ["pearson", "spearman"], horizontal=True, key="corr_method"
        )
        fig = get_corr_heatmap(df, method=method)
        st.plotly_chart(fig)

        top = get_top_correlations(df, method=method)
        if not top.empty:
            st.subheader("Strongest Pairs")
            st.dataframe(top, hide_index=True)

with tab_profile:
    st.caption(
        "Generates a complete ydata-profiling report. Can take a while on large datasets."
    )
    PROFILE_ROW_LIMIT = 20_000
    profile_df = df
    if len(df) > PROFILE_ROW_LIMIT:
        st.warning(
            f"This dataset has {len(df):,} rows. Profiling the full thing can be slow "
            f"and memory-heavy, so it'll run on a random sample of {PROFILE_ROW_LIMIT:,} rows."
        )
        use_full = st.checkbox("Use the full dataset instead (may be slow)", value=False)
        if not use_full:
            profile_df = df.sample(PROFILE_ROW_LIMIT, random_state=42)

    if st.button("Generate Profiling Report"):
        with st.spinner("Generating profiling report..."):
            profile = ProfileReport(profile_df, explorative=True)
            components.html(profile.to_html(), height=1000, scrolling=True)
