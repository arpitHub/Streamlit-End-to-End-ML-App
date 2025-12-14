import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from pycaret.classification import setup as cls_setup, compare_models as cls_compare, pull as cls_pull, finalize_model as cls_finalize, predict_model as cls_predict
from pycaret.regression import setup as reg_setup, compare_models as reg_compare, pull as reg_pull, finalize_model as reg_finalize, predict_model as reg_predict

def infer_problem_type(df: pd.DataFrame, target: str) -> str:
    y = df[target]
    if y.dtype == "object" or y.nunique() < 20:
        return "classification"
    return "regression"

def basic_train_test_split(df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def run_pycaret_automl(df: pd.DataFrame, target: str, problem_type: str):
    if problem_type == "classification":
        cls_setup(data=df, target=target, session_id=42, silent=True, verbose=False)
        best = cls_compare()
        leaderboard = cls_pull().copy()
        final_model = cls_finalize(best)
        return final_model, leaderboard, "classification"
    else:
        reg_setup(data=df, target=target, session_id=42, silent=True, verbose=False)
        best = reg_compare()
        leaderboard = reg_pull().copy()
        final_model = reg_finalize(best)
        return final_model, leaderboard, "regression"

def evaluate_basic_model(problem_type: str, y_true, y_pred):
    if problem_type == "classification":
        return {"accuracy": accuracy_score(y_true, y_pred)}
    else:
        return {"r2": r2_score(y_true, y_pred)}
