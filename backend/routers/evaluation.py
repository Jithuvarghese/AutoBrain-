import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from models.storage_utils import append_step, get_project_path, read_state, write_state


router = APIRouter()


def _apply_feature_metadata(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    df_copy = df.copy()
    for col, info in (metadata or {}).items():
        if col not in df_copy.columns:
            continue
        if info.get("kind") == "category":
            categories = info.get("categories", [])
            df_copy[col] = pd.Categorical(df_copy[col].astype(str), categories=categories).codes
            df_copy[col] = df_copy[col].replace(-1, 0)
        elif info.get("kind") == "boolean":
            df_copy[col] = df_copy[col].astype(int)
        else:
            df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce").fillna(0)
    return df_copy


def _feature_importance_map(model, feature_names):
    estimator = model.named_steps.get("model", model) if hasattr(model, "named_steps") else model
    importances = {}
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = estimator.coef_
        values = np.mean(np.abs(values), axis=0) if np.ndim(values) > 1 else np.abs(values)
    else:
        return {}
    ordered = sorted(zip(feature_names, values), key=lambda item: float(item[1]), reverse=True)[:15]
    for feature_name, score in ordered:
        importances[str(feature_name)] = float(score)
    return importances


@router.post("/{project_id}/evaluate")
def evaluate_model(project_id: str):
    try:
        state = read_state(project_id)
        project_path = get_project_path(project_id)
        model_path = project_path / "model.pkl"
        x_test_path = project_path / "X_test.csv"
        y_test_path = project_path / "y_test.csv"
        metadata_path = project_path / "feature_metadata.pkl"
        if not model_path.exists() or not x_test_path.exists() or not y_test_path.exists():
            raise FileNotFoundError("Model or test data not found")

        model = joblib.load(model_path)
        x_test = pd.read_csv(x_test_path)
        y_test = pd.read_csv(y_test_path).iloc[:, 0]
        metadata = joblib.load(metadata_path) if metadata_path.exists() else {}
        x_test = _apply_feature_metadata(x_test, metadata)

        y_pred = model.predict(x_test)
        problem_type = state.get("sampling", {}).get("problem_type")
        feature_names_path = project_path / "feature_names.pkl"
        feature_names = joblib.load(feature_names_path) if feature_names_path.exists() else list(x_test.columns)

        evaluation_state = {
            "problem_type": problem_type,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "roc_auc": None,
            "mae": None,
            "mse": None,
            "rmse": None,
            "r2_score": None,
            "confusion_matrix": [],
            "confusion_matrix_labels": [],
            "feature_importances": _feature_importance_map(model, feature_names),
            "classification_report": {},
            "actual_vs_predicted": [],
        }

        if problem_type == "classification":
            labels = list(pd.Index(y_test.astype(str)).unique())
            evaluation_state.update(
                {
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
                    "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
                    "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
                    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                    "confusion_matrix_labels": [str(label) for label in labels],
                    "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
                }
            )
            unique_values = pd.Index(y_test.astype(str)).unique()
            if len(unique_values) == 2 and hasattr(model, "predict_proba"):
                try:
                    probabilities = model.predict_proba(x_test)[:, 1]
                    evaluation_state["roc_auc"] = float(roc_auc_score(pd.factorize(y_test)[0], probabilities))
                except Exception:
                    evaluation_state["roc_auc"] = None
        else:
            mse = float(mean_squared_error(y_test, y_pred))
            evaluation_state.update(
                {
                    "mae": float(mean_absolute_error(y_test, y_pred)),
                    "mse": mse,
                    "rmse": float(np.sqrt(mse)),
                    "r2_score": float(r2_score(y_test, y_pred)),
                    "actual_vs_predicted": [
                        {"actual": float(actual) if pd.notna(actual) else None, "predicted": float(predicted) if pd.notna(predicted) else None}
                        for actual, predicted in list(zip(y_test.tolist(), y_pred.tolist()))[:100]
                    ],
                }
            )

        state["evaluation"] = evaluation_state
        append_step(state, 6)
        write_state(project_id, state)
        return evaluation_state
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}") from exc


@router.get("/{project_id}/results")
def evaluation_results(project_id: str):
    try:
        state = read_state(project_id)
        return state.get("evaluation", {})
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load evaluation results: {exc}") from exc
