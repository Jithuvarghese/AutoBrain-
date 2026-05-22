import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sklearn.model_selection import train_test_split

from models.storage_utils import append_step, get_project_path, load_best_dataframe, read_state, write_state


router = APIRouter()


class SamplingPayload(BaseModel):
    target_column: str
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = False


def _problem_type_for_target(series: pd.Series) -> str:
    if series.dtype == "object" or series.nunique(dropna=True) <= 10:
        return "classification"
    return "regression"


@router.post("/{project_id}/configure")
def configure_sampling(project_id: str, payload: SamplingPayload):
    try:
        dataframe, _ = load_best_dataframe(project_id)
        if payload.target_column not in dataframe.columns:
            raise HTTPException(status_code=400, detail="Target column not found in dataset")

        target = dataframe[payload.target_column]
        features = dataframe.drop(columns=[payload.target_column])
        problem_type = _problem_type_for_target(target)

        class_distribution = {}
        target_distribution = {}
        if problem_type == "classification":
            class_distribution = target.astype(str).value_counts().to_dict()
        else:
            target_distribution = {
                "min": float(pd.to_numeric(target, errors="coerce").min()),
                "max": float(pd.to_numeric(target, errors="coerce").max()),
                "mean": float(pd.to_numeric(target, errors="coerce").mean()),
                "median": float(pd.to_numeric(target, errors="coerce").median()),
                "std": float(pd.to_numeric(target, errors="coerce").std()),
            }

        stratify_values = target if (problem_type == "classification" and payload.stratify) else None
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=payload.test_size,
            random_state=payload.random_state,
            stratify=stratify_values,
        )

        project_path = get_project_path(project_id)
        x_train.to_csv(project_path / "X_train.csv", index=False)
        x_test.to_csv(project_path / "X_test.csv", index=False)
        y_train.to_csv(project_path / "y_train.csv", index=False, header=True)
        y_test.to_csv(project_path / "y_test.csv", index=False, header=True)

        state = read_state(project_id)
        state["sampling"] = {
            "target_column": payload.target_column,
            "problem_type": problem_type,
            "test_size": payload.test_size,
            "random_state": payload.random_state,
            "stratify": payload.stratify,
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "feature_count": int(features.shape[1]),
            "feature_names": [str(column) for column in features.columns.tolist()],
            "class_distribution": class_distribution,
            "target_distribution": target_distribution,
        }
        append_step(state, 4)
        write_state(project_id, state)
        return state["sampling"]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Sampling configuration failed: {exc}") from exc


@router.get("/{project_id}/info")
def sampling_info(project_id: str):
    try:
        state = read_state(project_id)
        return state.get("sampling", {})
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load sampling info: {exc}") from exc
