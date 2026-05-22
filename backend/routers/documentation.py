from datetime import datetime

from fastapi import APIRouter, HTTPException

from models.storage_utils import append_step, read_state, write_state


router = APIRouter()


@router.get("/{project_id}/generate")
def generate_documentation(project_id: str):
    try:
        state = read_state(project_id)
        upload = state.get("upload", {})
        preprocessing = state.get("preprocessing", {})
        features = state.get("feature_engineering", {})
        sampling = state.get("sampling", {})
        training = state.get("training", {})
        evaluation = state.get("evaluation", {})

        report = {
            "project_name": state.get("project_name"),
            "project_id": state.get("project_id"),
            "generated_at": datetime.utcnow().isoformat(),
            "pipeline_summary": {
                "dataset": {
                    "filename": upload.get("filename"),
                    "rows": upload.get("rows"),
                    "columns": upload.get("columns"),
                },
                "preprocessing": {
                    "mode": preprocessing.get("mode"),
                    "actions_count": len(preprocessing.get("actions_applied", [])),
                },
                "feature_engineering": {
                    "transformations": len(features.get("actions", [])),
                },
                "model": {
                    "algorithm": training.get("algorithm_name"),
                    "training_time": training.get("training_time_seconds"),
                },
                "performance": {
                    "accuracy": evaluation.get("accuracy"),
                    "f1_score": evaluation.get("f1_score"),
                    "r2_score": evaluation.get("r2_score"),
                    "rmse": evaluation.get("rmse"),
                },
            },
            "sections": [
                {
                    "title": "1. Dataset Overview",
                    "content": f"The dataset contains {upload.get('rows', 0)} rows and {upload.get('columns', 0)} columns from {upload.get('filename') or 'an uploaded source' }.",
                    "stats": {
                        "filename": upload.get("filename"),
                        "rows": upload.get("rows"),
                        "columns": upload.get("columns"),
                    },
                    "actions": [],
                },
                {
                    "title": "2. Preprocessing Summary",
                    "content": "Preprocessing prepared the dataset for modeling by applying cleaning operations and handling missing values.",
                    "stats": {
                        "mode": preprocessing.get("mode"),
                        "rows_before": preprocessing.get("rows_before"),
                        "rows_after": preprocessing.get("rows_after"),
                    },
                    "actions": preprocessing.get("actions_applied", []),
                },
                {
                    "title": "3. Feature Engineering",
                    "content": "Feature engineering transformed the cleaned dataset to improve model learning.",
                    "stats": {
                        "actions": len(features.get("actions", [])),
                        "new_columns": len(features.get("new_columns", [])),
                    },
                    "actions": features.get("actions", []),
                },
                {
                    "title": "4. Model Training Configuration",
                    "content": "The selected model was trained using the configured hyperparameters and the prepared train split.",
                    "stats": {
                        "algorithm": training.get("algorithm_name"),
                        "training_time_seconds": training.get("training_time_seconds"),
                        "hyperparameters": training.get("hyperparameters", {}),
                    },
                    "actions": [],
                },
                {
                    "title": "5. Model Evaluation Results",
                    "content": "The evaluation step calculated metrics and generated diagnostic outputs for the trained model.",
                    "stats": evaluation,
                    "actions": [],
                },
                {
                    "title": "6. Conclusion & Recommendations",
                    "content": "Review the selected metrics and feature importances to decide whether to refine preprocessing, tune the model, or gather more data.",
                    "stats": {
                        "problem_type": sampling.get("problem_type"),
                        "steps_completed": state.get("steps_completed", []),
                    },
                    "actions": [],
                },
            ],
            "steps_completed": state.get("steps_completed", []),
        }

        append_step(state, 7)
        write_state(project_id, state)
        report["steps_completed"] = state.get("steps_completed", [])
        return report
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate documentation: {exc}") from exc
