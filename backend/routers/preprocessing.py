from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from models.storage_utils import append_step, get_project_path, read_state, write_state


router = APIRouter()


class PreprocessingAction(BaseModel):
    type: str
    column: str | None = None
    value: str | None = None
    dtype: str | None = None


class PreprocessingPayload(BaseModel):
    actions: list[PreprocessingAction] = Field(default_factory=list)


def _load_raw_dataframe(project_id: str) -> pd.DataFrame:
    raw_path = get_project_path(project_id) / "raw_data.csv"
    if not raw_path.exists():
        raise FileNotFoundError("Raw dataset not found")
    return pd.read_csv(raw_path)


def _save_processed(project_id: str, dataframe: pd.DataFrame) -> None:
    output_path = get_project_path(project_id) / "processed_data.csv"
    dataframe.to_csv(output_path, index=False)


def _build_suggestions(dataframe: pd.DataFrame) -> list[dict]:
    suggestions = []
    row_count = len(dataframe)
    for column_name in dataframe.columns:
        series = dataframe[column_name]
        null_count = int(series.isna().sum())
        null_pct = (null_count / row_count * 100) if row_count else 0
        if null_count > 0 and null_pct > 50:
            suggestions.append(
                {
                    "type": "drop_column",
                    "column": column_name,
                    "issue": f"{column_name} has {round(null_pct, 2)}% missing values",
                    "reason": "Too many missing values can weaken model quality.",
                    "recommended_action": "Drop the column",
                    "severity": "high",
                }
            )
        elif 0 < null_count <= row_count * 0.5:
            suggestions.append(
                {
                    "type": "fill_null",
                    "column": column_name,
                    "issue": f"{column_name} contains {null_count} missing values",
                    "reason": "Missing values should be handled before modeling.",
                    "recommended_action": "Fill missing values",
                    "severity": "medium",
                }
            )
        if series.nunique(dropna=True) <= 1:
            suggestions.append(
                {
                    "type": "drop_column",
                    "column": column_name,
                    "issue": f"{column_name} is constant or near-constant",
                    "reason": "Constant columns add no predictive value.",
                    "recommended_action": "Drop the column",
                    "severity": "high",
                }
            )
    duplicate_count = int(dataframe.duplicated().sum())
    if duplicate_count > 0:
        suggestions.append(
            {
                "type": "remove_duplicates",
                "column": None,
                "issue": f"Dataset has {duplicate_count} duplicate rows",
                "reason": "Duplicates can bias training and evaluation.",
                "recommended_action": "Remove duplicate rows",
                "severity": "medium",
            }
        )
    return suggestions


@router.get("/{project_id}/suggestions")
def get_suggestions(project_id: str):
    try:
        dataframe = _load_raw_dataframe(project_id)
        return {"suggestions": _build_suggestions(dataframe)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to build suggestions: {exc}") from exc


@router.post("/{project_id}/auto")
def auto_clean(project_id: str):
    try:
        dataframe = _load_raw_dataframe(project_id)
        original_rows, original_columns = dataframe.shape
        state = read_state(project_id)
        actions_applied: list[str] = []
        dropped_columns: list[str] = []
        filled_nulls: dict[str, str] = {}

        duplicate_count = int(dataframe.duplicated().sum())
        for column_name in list(dataframe.columns):
            series = dataframe[column_name]
            null_pct = series.isna().mean() * 100 if len(series) else 0
            if null_pct > 50 or series.nunique(dropna=True) <= 1:
                dataframe = dataframe.drop(columns=[column_name])
                dropped_columns.append(column_name)
                actions_applied.append(f"Dropped column '{column_name}'")

        for column_name in dataframe.columns:
            series = dataframe[column_name]
            if series.isna().any():
                if pd.api.types.is_numeric_dtype(series):
                    fill_value = series.median()
                    filled_nulls[column_name] = "median"
                else:
                    mode_values = series.mode(dropna=True)
                    fill_value = mode_values.iloc[0] if not mode_values.empty else ""
                    filled_nulls[column_name] = "mode"
                dataframe[column_name] = series.fillna(fill_value)
                actions_applied.append(f"Filled nulls in '{column_name}' using {filled_nulls[column_name]}")

        if duplicate_count > 0:
            dataframe = dataframe.drop_duplicates()
            actions_applied.append(f"Removed {duplicate_count} duplicate rows")

        _save_processed(project_id, dataframe)
        state["preprocessing"] = {
            "mode": "auto",
            "actions_applied": actions_applied,
            "dropped_columns": dropped_columns,
            "filled_nulls": filled_nulls,
            "removed_duplicates": duplicate_count > 0,
            "rows_before": int(original_rows),
            "columns_before": int(original_columns),
            "rows_after": int(dataframe.shape[0]),
            "columns_after": int(dataframe.shape[1]),
        }
        append_step(state, 2)
        write_state(project_id, state)
        return state["preprocessing"]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Auto preprocessing failed: {exc}") from exc


@router.post("/{project_id}/manual")
def manual_clean(project_id: str, payload: PreprocessingPayload):
    try:
        dataframe = _load_raw_dataframe(project_id)
        original_rows, original_columns = dataframe.shape
        actions_applied: list[str] = []
        dropped_columns: list[str] = []
        filled_nulls: dict[str, str] = {}
        removed_duplicates = False

        for action in payload.actions:
            if action.type == "drop_column" and action.column in dataframe.columns:
                dataframe = dataframe.drop(columns=[action.column])
                dropped_columns.append(action.column)
                actions_applied.append(f"Dropped column '{action.column}'")
            elif action.type == "fill_null" and action.column in dataframe.columns:
                series = dataframe[action.column]
                if action.value == "median" and pd.api.types.is_numeric_dtype(series):
                    fill_value = series.median()
                elif action.value == "mean" and pd.api.types.is_numeric_dtype(series):
                    fill_value = series.mean()
                else:
                    mode_values = series.mode(dropna=True)
                    fill_value = mode_values.iloc[0] if not mode_values.empty else ""
                dataframe[action.column] = series.fillna(fill_value)
                filled_nulls[action.column] = action.value or "mode"
                actions_applied.append(f"Filled nulls in '{action.column}' using {filled_nulls[action.column]}")
            elif action.type == "remove_duplicates":
                before = len(dataframe)
                dataframe = dataframe.drop_duplicates()
                removed = before - len(dataframe)
                removed_duplicates = removed > 0 or removed_duplicates
                actions_applied.append(f"Removed {removed} duplicate rows")
            elif action.type == "drop_rows_with_null" and action.column in dataframe.columns:
                before = len(dataframe)
                dataframe = dataframe.dropna(subset=[action.column])
                removed = before - len(dataframe)
                actions_applied.append(f"Dropped {removed} rows with null values in '{action.column}'")
            elif action.type == "change_dtype" and action.column in dataframe.columns and action.dtype:
                if action.dtype == "int":
                    dataframe[action.column] = pd.to_numeric(dataframe[action.column], errors="coerce").astype("Int64")
                elif action.dtype == "float":
                    dataframe[action.column] = pd.to_numeric(dataframe[action.column], errors="coerce")
                else:
                    dataframe[action.column] = dataframe[action.column].astype(str)
                actions_applied.append(f"Changed dtype of '{action.column}' to {action.dtype}")

        _save_processed(project_id, dataframe)
        state = read_state(project_id)
        state["preprocessing"] = {
            "mode": "manual",
            "actions_applied": actions_applied,
            "dropped_columns": dropped_columns,
            "filled_nulls": filled_nulls,
            "removed_duplicates": removed_duplicates,
            "rows_before": int(original_rows),
            "columns_before": int(original_columns),
            "rows_after": int(dataframe.shape[0]),
            "columns_after": int(dataframe.shape[1]),
        }
        append_step(state, 2)
        write_state(project_id, state)
        return state["preprocessing"]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Manual preprocessing failed: {exc}") from exc


@router.get("/{project_id}/stats")
def preprocessing_stats(project_id: str):
    try:
        state = read_state(project_id)
        return state.get("preprocessing", {})
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read preprocessing stats: {exc}") from exc