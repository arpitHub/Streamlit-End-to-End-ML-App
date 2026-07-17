import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

CLASSIFICATION_GRIDS = {
    "Logistic Regression": {"C": [0.1, 1, 10]},
    "Random Forest": {"n_estimators": [100, 300], "max_depth": [None, 10]},
    "Gradient Boosting": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
    "SVC": {"C": [0.1, 1, 10]},
    "KNN": {"n_neighbors": [3, 5, 7]},
    "Decision Tree": {"max_depth": [None, 5, 10]},
}

REGRESSION_GRIDS = {
    "Ridge": {"alpha": [0.1, 1, 10]},
    "Lasso": {"alpha": [0.01, 0.1, 1]},
    "Random Forest Regressor": {"n_estimators": [100, 300], "max_depth": [None, 10]},
    "Gradient Boosting Regressor": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
    "SVR": {"C": [0.1, 1, 10]},
    "KNN Regressor": {"n_neighbors": [3, 5, 7]},
    "Decision Tree Regressor": {"max_depth": [None, 5, 10]},
}

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    return ColumnTransformer(transformers)

def _make_cv(y_train, is_classification: bool, cv_folds: int):
    """Clamp folds so small datasets and rare classes don't break CV."""
    if is_classification:
        min_class = int(y_train.value_counts().min())
        if min_class >= 2:
            return StratifiedKFold(
                n_splits=max(2, min(cv_folds, min_class)), shuffle=True, random_state=42
            )
    return KFold(
        n_splits=max(2, min(cv_folds, len(y_train) // 2)), shuffle=True, random_state=42
    )

def run_sklearn_compare(df: pd.DataFrame, target: str, cv_folds: int = 5, tune: bool = False):
    df = df.dropna(subset=[target])

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    is_classification = (y.dtype == "object") or (y.nunique() < 20)
    problem_type = "classification" if is_classification else "regression"

    if is_classification:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=500),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "SVC": SVC(),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "Decision Tree": DecisionTreeClassifier(),
        }
        grids = CLASSIFICATION_GRIDS
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "Random Forest Regressor": RandomForestRegressor(),
            "Gradient Boosting Regressor": GradientBoostingRegressor(),
            "SVR": SVR(),
            "KNN Regressor": KNeighborsRegressor(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
        }
        grids = REGRESSION_GRIDS

    cv = _make_cv(y_train, is_classification, cv_folds)

    results = []
    best_model = None
    best_score = -float("inf")

    for name, model in models.items():
        try:
            pipeline = Pipeline([
                ("preprocessor", build_preprocessor(X_train)),
                ("model", model),
            ])

            best_params = ""
            if tune and name in grids:
                grid = {f"model__{k}": v for k, v in grids[name].items()}
                search = GridSearchCV(pipeline, grid, cv=cv, n_jobs=-1)
                search.fit(X_train, y_train)
                fitted = search.best_estimator_
                cv_mean = float(search.best_score_)
                cv_std = float(search.cv_results_["std_test_score"][search.best_index_])
                best_params = ", ".join(
                    f"{k.replace('model__', '')}={v}" for k, v in search.best_params_.items()
                )
            else:
                scores = cross_val_score(pipeline, X_train, y_train, cv=cv, n_jobs=-1)
                cv_mean = float(np.mean(scores))
                cv_std = float(np.std(scores))
                fitted = pipeline.fit(X_train, y_train)

            test_preds = fitted.predict(X_test)
            test_score = (
                accuracy_score(y_test, test_preds)
                if is_classification
                else r2_score(y_test, test_preds)
            )

            results.append({
                "Model": name,
                "CV Mean": round(cv_mean, 4),
                "CV Std": round(cv_std, 4),
                "Test Score": round(float(test_score), 4),
                "Best Params": best_params,
            })

            if cv_mean > best_score:
                best_score = cv_mean
                best_model = fitted

        except Exception as e:
            results.append({
                "Model": name,
                "CV Mean": None,
                "CV Std": None,
                "Test Score": None,
                "Best Params": "",
                "Error": str(e),
            })

    leaderboard = pd.DataFrame(results).sort_values(
        "CV Mean", ascending=False, na_position="last"
    )
    if "Error" in leaderboard.columns and leaderboard["Error"].isna().all():
        leaderboard = leaderboard.drop(columns="Error")
    if not tune:
        leaderboard = leaderboard.drop(columns="Best Params")

    return leaderboard, best_model, X_test, y_test, problem_type
