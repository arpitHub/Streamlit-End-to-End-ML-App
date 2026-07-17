# Streamlit End-to-End ML App 🤖

An interactive Streamlit application that walks through a complete machine
learning workflow — pick a dataset, explore it, train and compare models, and
inspect the results — without writing any code.

## Features

### 📊 Dataset Explorer
- **Built-in datasets** — Iris, Wine, Breast Cancer (from scikit-learn)
- **pydataset library** — 750+ classic R datasets, searchable by name
- **CSV upload** — bring your own data
- Preview, shape, and column type summary; time series datasets are
  detected automatically (they carry a `time` column) and flagged.

### 🔍 EDA Dashboard
- **Overview** — rows/columns/missing/duplicate metrics, summary statistics,
  missing-values breakdown
- **Distributions** — histograms with box marginals for numeric columns
  (optionally grouped by a categorical column), value-count bars for
  categorical columns
- **Relationships** — scatter plots with optional categorical coloring;
  time series datasets also get a value-over-time line chart
- **Correlation** — Pearson/Spearman heatmap plus a "strongest pairs" table
- **Full Profiling** — complete [ydata-profiling](https://github.com/ydataai/ydata-profiling)
  report on demand

### 🤖 Model Builder
- **Standard ML** — compares 7–8 scikit-learn models (logistic/linear
  regression, random forest, gradient boosting, SVM, KNN, naive Bayes,
  decision tree, ridge/lasso) with:
  - automatic problem-type detection (classification vs regression)
  - preprocessing pipeline: median/mode imputation, one-hot encoding,
    standard scaling
  - k-fold cross-validation leaderboard (configurable folds) with a
    20% holdout test score
  - optional grid-search hyperparameter tuning
- **Time Series Forecast** — offered automatically for time-indexed
  datasets; compares naive, moving average, linear trend, simple
  exponential smoothing, and Holt's linear trend on a chronological
  holdout, then projects future periods with the best method.

### 📈 Model Results
- **Classification** — test accuracy, confusion matrix, per-class
  precision/recall/F1
- **Regression** — R², RMSE, MAE, actual-vs-predicted plot, residual
  distribution
- **Feature importance** — permutation importance on the test set
- **Time series** — history/backtest/forecast chart
- Downloadable predictions/forecast CSV

## Getting Started

```bash
git clone https://github.com/arpitHub/Streamlit-End-to-End-ML-App.git
cd Streamlit-End-to-End-ML-App
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 and use the sidebar to navigate.

### Dev Container / Codespaces

The repo includes a [`.devcontainer`](.devcontainer/devcontainer.json) that
installs dependencies and launches the app automatically on port 8501.

## Project Structure

```
├── app.py                      # Landing page
├── pages/
│   ├── 1_📊_Dataset_Explorer.py
│   ├── 2_🔍_EDA_Dashboard.py
│   ├── 3_🤖_Model_Builder.py
│   └── 4_📈_Model_Results.py
├── utils/
│   ├── data_loader.py          # Dataset loading (built-in, pydataset, CSV)
│   ├── eda_tools.py            # EDA stats and Plotly figures
│   ├── sklearn_compare.py      # Model comparison with CV + tuning
│   ├── timeseries_tools.py     # Time series detection + forecasting
│   ├── model_eval.py           # Evaluation metrics, plots, importance
│   └── state.py                # Shared session state
├── requirements.txt
└── Procfile / setup.sh         # Heroku-style deployment
```

## Tech Stack

- [Streamlit](https://streamlit.io/) — UI
- [scikit-learn](https://scikit-learn.org/) — models, preprocessing, metrics
- [Plotly](https://plotly.com/python/) — interactive charts
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data handling
- [ydata-profiling](https://github.com/ydataai/ydata-profiling) — automated EDA reports
- [pydataset](https://github.com/iamaziz/PyDataset) — sample datasets
