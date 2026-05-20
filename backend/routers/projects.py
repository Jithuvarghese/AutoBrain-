import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from models.storage_utils import create_initial_state, delete_project, ensure_project_path, list_all_projects, read_state, write_state


router = APIRouter()


class ProjectNamePayload(BaseModel):
    name: str


@router.get("")
def get_projects():
    try:
        return list_all_projects()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to list projects: {exc}") from exc


@router.post("")
def create_project(payload: ProjectNamePayload):
    try:
        project_id = str(uuid.uuid4())
        ensure_project_path(project_id)
        state = create_initial_state(project_id, payload.name)
        write_state(project_id, state)
        return state
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to create project: {exc}") from exc


@router.get("/{project_id}")
def get_project(project_id: str):
    try:
        return read_state(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load project: {exc}") from exc


@router.delete("/{project_id}")
def remove_project(project_id: str):
    try:
        delete_project(project_id)
        return {"success": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete project: {exc}") from exc


@router.put("/{project_id}/name")
def rename_project(project_id: str, payload: ProjectNamePayload):
    try:
        state = read_state(project_id)
        state["project_name"] = payload.name
        write_state(project_id, state)
        return state
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to rename project: {exc}") from exc