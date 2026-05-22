import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from models.storage_utils import append_step, get_project_path, load_best_dataframe, read_state, write_state


router = APIRouter()


class FeatureAction(BaseModel):
    type: str
    column: str | None = None
    column2: str | None = None
    params: dict = Field(default_factory=dict)
    code: str | None = None


class FeaturePayload(BaseModel):
    actions: list[FeatureAction] = Field(default_factory=list)


def _safe_series_preview(series: pd.Series) -> list:
    return [None if pd.isna(value) else (value.item() if hasattr(value, "item") else value) for value in series.dropna().head(3).tolist()]


def _build_column_summary(dataframe: pd.DataFrame) -> list[dict]:
    summary = []
    for column_name in dataframe.columns:
        series = dataframe[column_name]
        dtype_name = str(series.dtype)
        if pd.api.types.is_bool_dtype(series):
            column_type = "boolean"
        elif pd.api.types.is_numeric_dtype(series):
            column_type = "numeric"
        else:
            column_type = "categorical"
        summary.append(
            {
                "name": column_name,
                "dtype": dtype_name,
                "type": column_type,
                "unique_values": int(series.nunique(dropna=True)),
                "null_count": int(series.isna().sum()),
                "sample_values": _safe_series_preview(series),
            }
        )
    return summary


@router.get("/{project_id}/columns")
def get_columns(project_id: str):
    try:
        dataframe, source_name = load_best_dataframe(project_id)
        return {"source": source_name, "columns": _build_column_summary(dataframe)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load columns: {exc}") from exc


@router.post("/{project_id}/apply")
def apply_feature_actions(project_id: str, payload: FeaturePayload):
    try:
        dataframe, _ = load_best_dataframe(project_id)
        state = read_state(project_id)
        actions_applied: list[str] = []
        new_columns: list[str] = []
        dropped_columns: list[str] = []
        encoded_columns: list[str] = []
        scaled_columns: list[str] = []

        for action in payload.actions:
            if action.type == "one_hot_encode" and action.column in dataframe.columns:
                encoded = pd.get_dummies(dataframe[action.column], prefix=action.column)
                new_columns.extend(encoded.columns.tolist())
                dataframe = pd.concat([dataframe.drop(columns=[action.column]), encoded], axis=1)
                actions_applied.append(f"One-hot encoded '{action.column}'")
                encoded_columns.append(action.column)
            elif action.type == "label_encode" and action.column in dataframe.columns:
                encoder = LabelEncoder()
                dataframe[f"{action.column}_encoded"] = encoder.fit_transform(dataframe[action.column].astype(str))
                dataframe = dataframe.drop(columns=[action.column])
                new_columns.append(f"{action.column}_encoded")
                actions_applied.append(f"Label encoded '{action.column}'")
                encoded_columns.append(action.column)
            elif action.type == "min_max_scale" and action.column in dataframe.columns:
                scaler = MinMaxScaler()
                dataframe[[action.column]] = scaler.fit_transform(dataframe[[action.column]])
                actions_applied.append(f"Applied min-max scaling to '{action.column}'")
                scaled_columns.append(action.column)
            elif action.type == "standard_scale" and action.column in dataframe.columns:
                scaler = StandardScaler()
                dataframe[[action.column]] = scaler.fit_transform(dataframe[[action.column]])
                actions_applied.append(f"Applied standard scaling to '{action.column}'")
                scaled_columns.append(action.column)
            elif action.type == "log_transform" and action.column in dataframe.columns:
                new_column_name = f"{action.column}_log"
                dataframe[new_column_name] = np.log1p(pd.to_numeric(dataframe[action.column], errors="coerce").fillna(0).clip(lower=0))
                new_columns.append(new_column_name)
                actions_applied.append(f"Created log transform '{new_column_name}'")
            elif action.type == "drop_column" and action.column in dataframe.columns:
                dataframe = dataframe.drop(columns=[action.column])
                dropped_columns.append(action.column)
                actions_applied.append(f"Dropped column '{action.column}'")
            elif action.type == "create_interaction" and action.column in dataframe.columns and action.column2 in dataframe.columns:
                new_column_name = f"{action.column}_x_{action.column2}"
                dataframe[new_column_name] = pd.to_numeric(dataframe[action.column], errors="coerce").fillna(0) * pd.to_numeric(dataframe[action.column2], errors="coerce").fillna(0)
                new_columns.append(new_column_name)
                actions_applied.append(f"Created interaction '{new_column_name}'")
            elif action.type == "bin_column" and action.column in dataframe.columns:
                bins = int(action.params.get("bins", 4))
                labels = action.params.get("labels")
                dataframe[action.column] = pd.cut(dataframe[action.column], bins=bins, labels=labels)
                actions_applied.append(f"Binned '{action.column}' into {bins} groups")
            elif action.type == "custom_code":
                local_scope = {"df": dataframe, "pd": pd, "np": np}
                try:
                    exec(action.code or "", {"__builtins__": {}}, local_scope)
                    dataframe = local_scope["df"]
                    actions_applied.append("Executed custom code transformation")
                except Exception as exc:
                    raise HTTPException(status_code=400, detail=f"Custom code failed: {exc}") from exc

        output_path = get_project_path(project_id) / "engineered_data.csv"
        dataframe.to_csv(output_path, index=False)

        state["feature_engineering"] = {
            "actions": actions_applied,
            "new_columns": new_columns,
            "dropped_columns": dropped_columns,
            "encoded_columns": encoded_columns,
            "scaled_columns": scaled_columns,
        }
        append_step(state, 3)
        write_state(project_id, state)
        return {"actions": actions_applied, "columns": _build_column_summary(dataframe)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Feature engineering failed: {exc}") from exc


@router.post("/{project_id}/reset")
def reset_features(project_id: str):
    try:
        project_path = get_project_path(project_id)
        engineered_path = project_path / "engineered_data.csv"
        if engineered_path.exists():
            engineered_path.unlink()
        state = read_state(project_id)
        state["feature_engineering"] = {
            "actions": [],
            "new_columns": [],
            "dropped_columns": [],
            "encoded_columns": [],
            "scaled_columns": [],
        }
        write_state(project_id, state)
        return {"success": True}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to reset features: {exc}") from exc
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from models.storage_utils import append_step, get_project_path, load_best_dataframe, read_state, write_state


router = APIRouter()


class FeatureAction(BaseModel):
    type: str
    column: str | None = None
    column2: str | None = None
    params: dict = Field(default_factory=dict)
    code: str | None = None


class FeaturePayload(BaseModel):
    actions: list[FeatureAction] = Field(default_factory=list)


def _safe_series_preview(series: pd.Series) -> list:
    return [None if pd.isna(value) else (value.item() if hasattr(value, "item") else value) for value in series.dropna().head(3).tolist()]


def _build_column_summary(dataframe: pd.DataFrame) -> list[dict]:
    summary = []
    for column_name in dataframe.columns:
        series = dataframe[column_name]
        dtype_name = str(series.dtype)
        if pd.api.types.is_bool_dtype(series):
            column_type = "boolean"
        elif pd.api.types.is_numeric_dtype(series):
            column_type = "numeric"
        else:
            column_type = "categorical"
        summary.append(
            {
                "name": column_name,
                "dtype": dtype_name,
                "type": column_type,
                "unique_values": int(series.nunique(dropna=True)),
                "null_count": int(series.isna().sum()),
                "sample_values": _safe_series_preview(series),
            }
        )
    return summary


@router.get("/{project_id}/columns")
def get_columns(project_id: str):
    try:
        dataframe, source_name = load_best_dataframe(project_id)
        return {"source": source_name, "columns": _build_column_summary(dataframe)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load columns: {exc}") from exc


@router.post("/{project_id}/apply")
def apply_feature_actions(project_id: str, payload: FeaturePayload):
    try:
        dataframe, _ = load_best_dataframe(project_id)
        state = read_state(project_id)
        actions_applied: list[str] = []
        new_columns: list[str] = []
        dropped_columns: list[str] = []
        encoded_columns: list[str] = []
        scaled_columns: list[str] = []

        for action in payload.actions:
            if action.type == "one_hot_encode" and action.column in dataframe.columns:
                encoded = pd.get_dummies(dataframe[action.column], prefix=action.column)
                new_columns.extend(encoded.columns.tolist())
                dataframe = pd.concat([dataframe.drop(columns=[action.column]), encoded], axis=1)
                actions_applied.append(f"One-hot encoded '{action.column}'")
                encoded_columns.append(action.column)
            elif action.type == "label_encode" and action.column in dataframe.columns:
                encoder = LabelEncoder()
                dataframe[f"{action.column}_encoded"] = encoder.fit_transform(dataframe[action.column].astype(str))
                dataframe = dataframe.drop(columns=[action.column])
                new_columns.append(f"{action.column}_encoded")
                actions_applied.append(f"Label encoded '{action.column}'")
                encoded_columns.append(action.column)
            elif action.type == "min_max_scale" and action.column in dataframe.columns:
                scaler = MinMaxScaler()
                dataframe[[action.column]] = scaler.fit_transform(dataframe[[action.column]])
                actions_applied.append(f"Applied min-max scaling to '{action.column}'")
                scaled_columns.append(action.column)
            elif action.type == "standard_scale" and action.column in dataframe.columns:
                scaler = StandardScaler()
                dataframe[[action.column]] = scaler.fit_transform(dataframe[[action.column]])
                actions_applied.append(f"Applied standard scaling to '{action.column}'")
                scaled_columns.append(action.column)
            elif action.type == "log_transform" and action.column in dataframe.columns:
                new_column_name = f"{action.column}_log"
                dataframe[new_column_name] = np.log1p(pd.to_numeric(dataframe[action.column], errors="coerce").fillna(0).clip(lower=0))
                new_columns.append(new_column_name)
                actions_applied.append(f"Created log transform '{new_column_name}'")
            elif action.type == "drop_column" and action.column in dataframe.columns:
                dataframe = dataframe.drop(columns=[action.column])
                dropped_columns.append(action.column)
                actions_applied.append(f"Dropped column '{action.column}'")
            elif action.type == "create_interaction" and action.column in dataframe.columns and action.column2 in dataframe.columns:
                new_column_name = f"{action.column}_x_{action.column2}"
                dataframe[new_column_name] = pd.to_numeric(dataframe[action.column], errors="coerce").fillna(0) * pd.to_numeric(dataframe[action.column2], errors="coerce").fillna(0)
                new_columns.append(new_column_name)
                actions_applied.append(f"Created interaction '{new_column_name}'")
            elif action.type == "bin_column" and action.column in dataframe.columns:
                bins = int(action.params.get("bins", 4))
                labels = action.params.get("labels")
                dataframe[action.column] = pd.cut(dataframe[action.column], bins=bins, labels=labels)
                actions_applied.append(f"Binned '{action.column}' into {bins} groups")
            elif action.type == "custom_code":
                local_scope = {"df": dataframe, "pd": pd, "np": np}
                try:
                    exec(action.code or "", {"__builtins__": {}}, local_scope)
                    dataframe = local_scope["df"]
                    actions_applied.append("Executed custom code transformation")
                except Exception as exc:
                    raise HTTPException(status_code=400, detail=f"Custom code failed: {exc}") from exc

        output_path = get_project_path(project_id) / "engineered_data.csv"
        dataframe.to_csv(output_path, index=False)

        state["feature_engineering"] = {
            "actions": actions_applied,
            "new_columns": new_columns,
            "dropped_columns": dropped_columns,
            "encoded_columns": encoded_columns,
            "scaled_columns": scaled_columns,
        }
        append_step(state, 3)
        write_state(project_id, state)
        return {"actions": actions_applied, "columns": _build_column_summary(dataframe)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Feature engineering failed: {exc}") from exc


@router.post("/{project_id}/reset")
def reset_features(project_id: str):
    try:
        project_path = get_project_path(project_id)
        engineered_path = project_path / "engineered_data.csv"
        if engineered_path.exists():
            engineered_path.unlink()
        state = read_state(project_id)
        state["feature_engineering"] = {
            "actions": [],
            "new_columns": [],
            "dropped_columns": [],
            "encoded_columns": [],
            "scaled_columns": [],
        }
        write_state(project_id, state)
        return {"success": True}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to reset features: {exc}") from exc
