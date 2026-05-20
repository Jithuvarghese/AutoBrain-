from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from models.storage_utils import append_step, ensure_project_path, read_state, write_state


router = APIRouter()
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def _safe_preview_value(value):
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


@router.post("/{project_id}")
async def upload_dataset(project_id: str, file: UploadFile = File(...)):
    try:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Only .csv, .xlsx, and .xls files are supported")

        project_path = ensure_project_path(project_id)
        if suffix == ".csv":
            dataframe = pd.read_csv(file.file)
        else:
            dataframe = pd.read_excel(file.file)

        raw_path = project_path / "raw_data.csv"
        dataframe.to_csv(raw_path, index=False)

        rows, columns = dataframe.shape
        duplicate_count = int(dataframe.duplicated().sum())
        column_info = []
        null_counts = {}
        dtype_map = {}

        for column_name in dataframe.columns:
            series = dataframe[column_name]
            non_null_count = int(series.notna().sum())
            null_count = int(series.isna().sum())
            null_counts[column_name] = null_count
            dtype_map[column_name] = str(series.dtype)
            sample_values = [_safe_preview_value(value) for value in series.dropna().head(3).tolist()]
            column_info.append(
                {
                    "name": column_name,
                    "dtype": str(series.dtype),
                    "non_null_count": non_null_count,
                    "null_count": null_count,
                    "null_pct": round((null_count / rows * 100) if rows else 0, 2),
                    "sample_values": sample_values,
                }
            )

        preview = dataframe.head(100).where(pd.notnull(dataframe), None).to_dict(orient="records")

        state = read_state(project_id)
        state["upload"] = {
            "filename": file.filename,
            "rows": int(rows),
            "columns": int(columns),
            "column_names": [str(column) for column in dataframe.columns.tolist()],
            "dtypes": dtype_map,
            "null_counts": null_counts,
            "duplicate_count": duplicate_count,
            "column_info": column_info,
            "preview": preview,
        }
        append_step(state, 1)
        write_state(project_id, state)
        return state["upload"]
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc


@router.get("/{project_id}/preview")
def get_preview(project_id: str):
    try:
        state = read_state(project_id)
        return state.get("upload", {})
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read preview: {exc}") from exc