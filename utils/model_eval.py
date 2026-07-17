import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

def get_classification_metrics(y_true, y_pred) -> dict:
    return {"accuracy": accuracy_score(y_true, y_pred)}

def get_classification_report_df(y_true, y_pred) -> pd.DataFrame:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    df = pd.DataFrame(report).transpose()
    df.index.name = "Class"
    return df.round(3).reset_index()

def get_confusion_matrix_fig(y_true, y_pred):
    labels = sorted(pd.unique(pd.concat([pd.Series(y_true), pd.Series(y_pred)])))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = px.imshow(
        cm,
        x=[str(l) for l in labels],
        y=[str(l) for l in labels],
        text_auto=True,
        color_continuous_scale="Blues",
        aspect="auto",
    )
    fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual", coloraxis_showscale=False)
    return fig

def get_regression_metrics(y_true, y_pred) -> dict:
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": mean_absolute_error(y_true, y_pred),
    }

def get_actual_vs_predicted_fig(y_true, y_pred):
    df = pd.DataFrame({"Actual": np.asarray(y_true), "Predicted": np.asarray(y_pred)})
    fig = px.scatter(df, x="Actual", y="Predicted", opacity=0.7)
    lo = min(df["Actual"].min(), df["Predicted"].min())
    hi = max(df["Actual"].max(), df["Predicted"].max())
    fig.add_shape(
        type="line", x0=lo, y0=lo, x1=hi, y1=hi,
        line=dict(color="firebrick", dash="dash"),
    )
    return fig

def get_residuals_fig(y_true, y_pred):
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig = px.histogram(x=residuals, nbins=30)
    fig.update_layout(xaxis_title="Residual (Actual − Predicted)", yaxis_title="Count", bargap=0.05)
    return fig

def get_permutation_importance_df(model, X_test, y_test, n_repeats: int = 5, top_n: int = 20) -> pd.DataFrame:
    """Model-agnostic importance: how much the model's score drops when each
    input column is shuffled. Works directly on the fitted pipeline."""
    result = permutation_importance(
        model, X_test, y_test, n_repeats=n_repeats, random_state=42, n_jobs=-1
    )
    df = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": result.importances_mean,
        "Std": result.importances_std,
    }).sort_values("Importance", ascending=False).head(top_n)
    return df.reset_index(drop=True)

def get_importance_fig(importance_df: pd.DataFrame):
    fig = px.bar(
        importance_df.iloc[::-1],
        x="Importance",
        y="Feature",
        orientation="h",
        error_x="Std",
    )
    fig.update_layout(yaxis_title="", xaxis_title="Mean score drop when shuffled")
    return fig
