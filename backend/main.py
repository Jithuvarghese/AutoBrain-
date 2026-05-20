from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import documentation, evaluation, feature_engineering, preprocessing, projects, sampling, training, upload


app = FastAPI(title="ModelForge API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(projects.router, prefix="/api/projects", tags=["Projects"])
app.include_router(upload.router, prefix="/api/upload", tags=["Upload"])
app.include_router(preprocessing.router, prefix="/api/preprocessing", tags=["Preprocessing"])
app.include_router(feature_engineering.router, prefix="/api/features", tags=["Features"])
app.include_router(sampling.router, prefix="/api/sampling", tags=["Sampling"])
app.include_router(training.router, prefix="/api/training", tags=["Training"])
app.include_router(evaluation.router, prefix="/api/evaluation", tags=["Evaluation"])
app.include_router(documentation.router, prefix="/api/documentation", tags=["Documentation"])


@app.get("/")
def root():
    return {"message": "ModelForge API is running", "version": "1.0.0"}