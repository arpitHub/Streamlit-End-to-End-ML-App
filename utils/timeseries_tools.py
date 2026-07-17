import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


def detect_time_column(df: pd.DataFrame):
    """pydataset converts R ts objects to a DataFrame with a literal 'time' column."""
    for col in df.columns:
        if str(col).strip().lower() == "time":
            return col
    return None


def is_time_series_dataset(df: pd.DataFrame) -> bool:
    """True only when the dataset has an explicit 'time' column (how pydataset
    represents R ts/mts objects) — a plain row-number index doesn't count,
    since that would flag nearly every tabular dataset as a time series."""
    if df is None or df.shape[0] < 8:
        return False
    time_col = detect_time_column(df)
    if time_col is None:
        return False
    s = df[time_col]
    return pd.api.types.is_numeric_dtype(s) and s.is_monotonic_increasing


def default_value_column(df: pd.DataFrame, time_col):
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != time_col]
    return numeric_cols[0] if numeric_cols else None


def _naive_forecast(train: np.ndarray, horizon: int) -> np.ndarray:
    return np.full(horizon, train[-1])


def _moving_average_forecast(train: np.ndarray, horizon: int, window: int = None) -> np.ndarray:
    window = window or max(1, min(5, len(train) // 4 or 1))
    return np.full(horizon, np.mean(train[-window:]))


def _linear_trend_forecast(train_t: np.ndarray, train_y: np.ndarray, future_t: np.ndarray) -> np.ndarray:
    model = LinearRegression()
    model.fit(train_t.reshape(-1, 1), train_y)
    return model.predict(future_t.reshape(-1, 1))


def _best_ses_alpha(train: np.ndarray) -> float:
    best_alpha, best_sse = 0.3, np.inf
    for alpha in np.arange(0.05, 1.0, 0.05):
        level = train[0]
        sse = 0.0
        for y in train[1:]:
            sse += (y - level) ** 2
            level = alpha * y + (1 - alpha) * level
        if sse < best_sse:
            best_sse, best_alpha = sse, alpha
    return best_alpha


def _ses_forecast(train: np.ndarray, horizon: int, alpha: float = None):
    if alpha is None:
        alpha = _best_ses_alpha(train)
    level = train[0]
    for y in train[1:]:
        level = alpha * y + (1 - alpha) * level
    return np.full(horizon, level), alpha


def _holt_forecast(train: np.ndarray, horizon: int, alpha: float = 0.3, beta: float = 0.1) -> np.ndarray:
    level = train[0]
    trend = train[1] - train[0] if len(train) > 1 else 0.0
    for y in train[1:]:
        prev_level = level
        level = alpha * y + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
    return np.array([level + (h + 1) * trend for h in range(horizon)])


def run_time_series_forecast(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    test_ratio: float = 0.2,
    forecast_periods: int = 10,
):
    """Compares a handful of simple forecasting methods on a chronological holdout,
    then refits the best one on the full series to project future periods."""
    data = df[[time_col, value_col]].dropna().sort_values(time_col)
    t = data[time_col].to_numpy(dtype=float)
    y = data[value_col].to_numpy(dtype=float)
    n = len(y)
    if n < 8:
        raise ValueError("Not enough data points for a time series forecast (need at least 8).")

    test_size = max(1, int(round(n * test_ratio)))
    train_t, test_t = t[: n - test_size], t[n - test_size:]
    train_y, test_y = y[: n - test_size], y[n - test_size:]
    horizon = len(test_y)

    ses_test_preds, ses_alpha = _ses_forecast(train_y, horizon)
    candidates = {
        "Naive": _naive_forecast(train_y, horizon),
        "Moving Average": _moving_average_forecast(train_y, horizon),
        "Linear Trend": _linear_trend_forecast(train_t, train_y, test_t),
        f"Simple Exp. Smoothing (α={ses_alpha:.2f})": ses_test_preds,
        "Holt's Linear Trend": _holt_forecast(train_y, horizon),
    }

    results = []
    for name, preds in candidates.items():
        rmse = float(np.sqrt(mean_squared_error(test_y, preds)))
        mae = float(mean_absolute_error(test_y, preds))
        results.append({"Model": name, "RMSE": rmse, "MAE": mae})
    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    best_name = results_df.iloc[0]["Model"]

    step = float(np.median(np.diff(t))) if n > 1 else 1.0
    future_t = np.array([t[-1] + step * (i + 1) for i in range(forecast_periods)])

    base_name = best_name.split(" (α=")[0]
    if base_name == "Naive":
        future_preds = _naive_forecast(y, forecast_periods)
    elif base_name == "Moving Average":
        future_preds = _moving_average_forecast(y, forecast_periods)
    elif base_name == "Linear Trend":
        future_preds = _linear_trend_forecast(t, y, future_t)
    elif base_name == "Simple Exp. Smoothing":
        future_preds, _ = _ses_forecast(y, forecast_periods)
    else:
        future_preds = _holt_forecast(y, forecast_periods)

    return {
        "leaderboard": results_df,
        "best_model": best_name,
        "time_col": time_col,
        "value_col": value_col,
        "history": data.rename(columns={value_col: "Actual"})[[time_col, "Actual"]],
        "test_predictions": pd.DataFrame(
            {time_col: test_t, "Actual": test_y, "Predicted": candidates[best_name]}
        ),
        "forecast": pd.DataFrame({time_col: future_t, value_col: future_preds}),
    }
