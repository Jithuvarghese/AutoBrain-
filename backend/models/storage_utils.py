import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STORAGE_ROOT = Path(__file__).parent.parent / "storage" / "projects"


def ensure_storage_root() -> Path:
    STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    return STORAGE_ROOT


def get_project_path(project_id: str) -> Path:
    ensure_storage_root()
    return STORAGE_ROOT / project_id


def get_state_path(project_id: str) -> Path:
    return get_project_path(project_id) / "state.json"


def get_best_data_path(project_id: str) -> Path:
    project_path = get_project_path(project_id)
    for filename in ("engineered_data.csv", "processed_data.csv", "raw_data.csv"):
        candidate = project_path / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No dataset found for this project")


def get_data_path(project_id: str, filename: str) -> Path:
    return get_project_path(project_id) / filename


def load_best_dataframe(project_id: str) -> tuple[pd.DataFrame, str]:
    data_path = get_best_data_path(project_id)
    return pd.read_csv(data_path), data_path.name


def read_state(project_id: str) -> dict:
    state_path = get_state_path(project_id)
    if not state_path.exists():
        raise FileNotFoundError("Project state not found")
    with state_path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def _to_python(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_python(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_python(item) for item in value]
    if isinstance(value, tuple):
        return [_to_python(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def write_state(project_id: str, state: dict) -> None:
    state_path = get_state_path(project_id)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("w", encoding="utf-8") as file_handle:
        json.dump(_to_python(state), file_handle, indent=2, ensure_ascii=True)


def list_all_projects() -> list[dict]:
    ensure_storage_root()
    projects = []
    for state_path in STORAGE_ROOT.glob("*/state.json"):
        try:
            with state_path.open("r", encoding="utf-8") as file_handle:
                state = json.load(file_handle)
            projects.append(
                {
                    "project_id": state.get("project_id"),
                    "project_name": state.get("project_name"),
                    "created_at": state.get("created_at"),
                    "current_step": state.get("current_step", 1),
                    "steps_completed": state.get("steps_completed", []),
                }
            )
        except Exception:
            continue
    projects.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return projects


def create_initial_state(project_id: str, project_name: str) -> dict:
    return {
        "project_id": project_id,
        "project_name": project_name,
        "created_at": datetime.utcnow().isoformat(),
        "current_step": 1,
        "steps_completed": [],
        "upload": {
            "filename": None,
            "rows": 0,
            "columns": 0,
            "column_names": [],
            "dtypes": {},
            "null_counts": {},
            "duplicate_count": 0,
            "column_info": [],
            "preview": [],
        },
        "preprocessing": {
            "mode": None,
            "actions_applied": [],
            "dropped_columns": [],
            "filled_nulls": {},
            "removed_duplicates": False,
            "rows_before": 0,
            "columns_before": 0,
            "rows_after": 0,
            "columns_after": 0,
        },
        "feature_engineering": {
            "actions": [],
            "new_columns": [],
            "dropped_columns": [],
            "encoded_columns": [],
            "scaled_columns": [],
        },
        "sampling": {
            "target_column": None,
            "problem_type": None,
            "test_size": 0.2,
            "random_state": 42,
            "stratify": False,
            "train_rows": 0,
            "test_rows": 0,
            "feature_count": 0,
            "feature_names": [],
            "class_distribution": {},
            "target_distribution": {},
        },
        "training": {
            "algorithm": None,
            "algorithm_name": None,
            "hyperparameters": {},
            "training_time_seconds": 0,
            "model_saved": False,
        },
        "evaluation": {
            "problem_type": None,
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
            "feature_importances": {},
            "classification_report": {},
            "actual_vs_predicted": [],
        },
    }


def ensure_project_path(project_id: str) -> Path:
    project_path = get_project_path(project_id)
    project_path.mkdir(parents=True, exist_ok=True)
    return project_path


def append_step(state: dict, step_number: int) -> None:
    steps = state.setdefault("steps_completed", [])
    if step_number not in steps:
        steps.append(step_number)
        steps.sort()
    state["current_step"] = min(7, max(state.get("current_step", 1), step_number + 1))


def copy_dataframe_to_project(df: pd.DataFrame, project_id: str, filename: str) -> Path:
    project_path = ensure_project_path(project_id)
    output_path = project_path / filename
    df.to_csv(output_path, index=False)
    return output_path


def delete_project(project_id: str) -> None:
    project_path = get_project_path(project_id)
    if project_path.exists():
        shutil.rmtree(project_path)