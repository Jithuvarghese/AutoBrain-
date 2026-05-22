import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from models.storage_utils import append_step, get_project_path, read_state, write_state


router = APIRouter()


class TrainPayload(BaseModel):
    algorithm: str
    hyperparameters: dict = {}


CLASSIFICATION_ALGORITHMS = {
    "logistic_regression": {
        "id": "logistic_regression",
        "name": "Logistic Regression",
        "description": "Fast linear baseline for classification.",
        "tags": ["Linear", "Interpretable"],
        "hyperparameter_schema": {"C": {"type": "number", "default": 1.0}},
        "default_hyperparameters": {"C": 1.0},
        "factory": LogisticRegression,
    },
    "decision_tree": {
        "id": "decision_tree",
        "name": "Decision Tree",
        "description": "Tree-based model with human-readable splits.",
        "tags": ["Tree", "Interpretable"],
        "hyperparameter_schema": {"max_depth": {"type": "number", "default": None}},
        "default_hyperparameters": {"max_depth": None},
        "factory": DecisionTreeClassifier,
    },
    "random_forest": {
        "id": "random_forest",
        "name": "Random Forest",
        "description": "Strong ensemble of decision trees.",
        "tags": ["Ensemble", "High Accuracy"],
        "hyperparameter_schema": {"n_estimators": {"type": "number", "default": 100}, "max_depth": {"type": "number", "default": None}},
        "default_hyperparameters": {"n_estimators": 100, "max_depth": None},
        "factory": RandomForestClassifier,
    },
    "gradient_boosting": {
        "id": "gradient_boosting",
        "name": "Gradient Boosting",
        "description": "Boosted trees with strong predictive power.",
        "tags": ["Ensemble", "Accuracy"],
        "hyperparameter_schema": {"n_estimators": {"type": "number", "default": 100}, "learning_rate": {"type": "number", "default": 0.1}},
        "default_hyperparameters": {"n_estimators": 100, "learning_rate": 0.1},
        "factory": GradientBoostingClassifier,
    },
    "svm": {
        "id": "svm",
        "name": "Support Vector Machine",
        "description": "Margin-based classifier for complex boundaries.",
        "tags": ["Kernel", "Robust"],
        "hyperparameter_schema": {"C": {"type": "number", "default": 1.0}, "kernel": {"type": "select", "default": "rbf", "options": ["linear", "rbf", "poly", "sigmoid"]}},
        "default_hyperparameters": {"C": 1.0, "kernel": "rbf"},
        "factory": SVC,
    },
    "knn": {
        "id": "knn",
        "name": "K-Nearest Neighbors",
        "description": "Instance-based classifier using local neighborhoods.",
        "tags": ["Distance", "Simple"],
        "hyperparameter_schema": {"n_neighbors": {"type": "number", "default": 5}},
        "default_hyperparameters": {"n_neighbors": 5},
        "factory": KNeighborsClassifier,
    },
    "naive_bayes": {
        "id": "naive_bayes",
        "name": "Naive Bayes",
        "description": "Fast probabilistic classifier.",
        "tags": ["Probabilistic", "Fast"],
        "hyperparameter_schema": {},
        "default_hyperparameters": {},
        "factory": GaussianNB,
    },
}


REGRESSION_ALGORITHMS = {
    "linear_regression": {
        "id": "linear_regression",
        "name": "Linear Regression",
        "description": "Baseline linear regression model.",
        "tags": ["Linear", "Interpretable"],
        "hyperparameter_schema": {},
        "default_hyperparameters": {},
        "factory": LinearRegression,
    },
    "decision_tree_regressor": {
        "id": "decision_tree_regressor",
        "name": "Decision Tree Regressor",
        "description": "Tree model for nonlinear regression.",
        "tags": ["Tree", "Interpretable"],
        "hyperparameter_schema": {"max_depth": {"type": "number", "default": None}},
        "default_hyperparameters": {"max_depth": None},
        "factory": DecisionTreeRegressor,
    },
    "random_forest_regressor": {
        "id": "random_forest_regressor",
        "name": "Random Forest Regressor",
        "description": "Ensemble regression with strong generalization.",
        "tags": ["Ensemble", "High Accuracy"],
        "hyperparameter_schema": {"n_estimators": {"type": "number", "default": 100}},
        "default_hyperparameters": {"n_estimators": 100},
        "factory": RandomForestRegressor,
    },
    "gradient_boosting_regressor": {
        "id": "gradient_boosting_regressor",
        "name": "Gradient Boosting Regressor",
        "description": "Boosted tree ensemble for regression.",
        "tags": ["Ensemble", "Accuracy"],
        "hyperparameter_schema": {"n_estimators": {"type": "number", "default": 100}, "learning_rate": {"type": "number", "default": 0.1}},
        "default_hyperparameters": {"n_estimators": 100, "learning_rate": 0.1},
        "factory": GradientBoostingRegressor,
    },
    "knn_regressor": {
        "id": "knn_regressor",
        "name": "K-Nearest Neighbors Regressor",
        "description": "Distance-based regressor.",
        "tags": ["Distance", "Simple"],
        "hyperparameter_schema": {"n_neighbors": {"type": "number", "default": 5}},
        "default_hyperparameters": {"n_neighbors": 5},
        "factory": KNeighborsRegressor,
    },
}


def _load_split(project_id: str):
    project_path = get_project_path(project_id)
    x_train_path = project_path / "X_train.csv"
    y_train_path = project_path / "y_train.csv"
    if not x_train_path.exists() or not y_train_path.exists():
        raise FileNotFoundError("Training data not found. Configure sampling first.")
    x_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path).iloc[:, 0]
    return x_train, y_train, project_path


def _coerce_hyperparameters(schema: dict, incoming: dict) -> dict:
    hyperparameters = {}
    for name, config in schema.items():
        value = incoming.get(name, config.get("default"))
        if value is None:
            hyperparameters[name] = None
            continue
        if config.get("type") == "number":
            try:
                number_value = float(value)
                hyperparameters[name] = int(number_value) if number_value.is_integer() else number_value
            except Exception:
                hyperparameters[name] = value
        elif config.get("type") == "boolean":
            hyperparameters[name] = bool(value) if not isinstance(value, str) else value.lower() in {"true", "1", "yes", "on"}
        else:
            hyperparameters[name] = value
    for name, value in incoming.items():
        if name not in hyperparameters:
            hyperparameters[name] = value
    return hyperparameters


def _clean_training_features(features: pd.DataFrame):
    cleaned = features.copy()
    metadata = {}
    for column_name in cleaned.columns:
        series = cleaned[column_name]
        if pd.api.types.is_numeric_dtype(series):
            cleaned[column_name] = pd.to_numeric(series, errors="coerce").fillna(0)
            metadata[column_name] = {"kind": "numeric"}
        elif pd.api.types.is_bool_dtype(series):
            cleaned[column_name] = series.astype(int)
            metadata[column_name] = {"kind": "boolean"}
        else:
            numeric_series = pd.to_numeric(series, errors="coerce")
            if numeric_series.notna().all():
                cleaned[column_name] = numeric_series.fillna(0)
                metadata[column_name] = {"kind": "numeric"}
            else:
                categorical = series.astype("category")
                cleaned[column_name] = categorical.cat.codes.replace(-1, 0)
                metadata[column_name] = {"kind": "category", "categories": [str(item) for item in categorical.cat.categories.tolist()]}
    return cleaned, metadata


@router.get("/{project_id}/algorithms")
def get_algorithms(project_id: str):
    try:
        state = read_state(project_id)
        problem_type = state.get("sampling", {}).get("problem_type")
        if problem_type == "regression":
            return list(REGRESSION_ALGORITHMS.values())
        return list(CLASSIFICATION_ALGORITHMS.values())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load algorithms: {exc}") from exc


@router.post("/{project_id}/train")
def train_model(project_id: str, payload: TrainPayload):
    try:
        state = read_state(project_id)
        problem_type = state.get("sampling", {}).get("problem_type")
        algorithms = CLASSIFICATION_ALGORITHMS if problem_type == "classification" else REGRESSION_ALGORITHMS
        if payload.algorithm not in algorithms:
            raise HTTPException(status_code=400, detail="Algorithm is not available for this project type")

        x_train, y_train, project_path = _load_split(project_id)
        cleaned_x_train, metadata = _clean_training_features(x_train)
        algorithm_spec = algorithms[payload.algorithm]
        hyperparameters = _coerce_hyperparameters(algorithm_spec["hyperparameter_schema"], payload.hyperparameters)
        model = algorithm_spec["factory"](**{k: v for k, v in hyperparameters.items() if v is not None})
        model_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("model", model)])

        start_time = time.perf_counter()
        model_pipeline.fit(cleaned_x_train, y_train)
        training_time = time.perf_counter() - start_time

        joblib.dump(model_pipeline, project_path / "model.pkl")
        joblib.dump(list(cleaned_x_train.columns), project_path / "feature_names.pkl")
        joblib.dump(metadata, project_path / "feature_metadata.pkl")

        state["training"] = {
            "algorithm": payload.algorithm,
            "algorithm_name": algorithm_spec["name"],
            "hyperparameters": hyperparameters,
            "training_time_seconds": round(training_time, 4),
            "model_saved": True,
        }
        append_step(state, 5)
        write_state(project_id, state)
        return state["training"]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}") from exc
