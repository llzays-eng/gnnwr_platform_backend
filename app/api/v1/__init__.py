from fastapi import APIRouter

from app.api.v1 import auth, datasets, models, projects, reports, spatial

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(projects.router)
api_router.include_router(datasets.router)
api_router.include_router(models.router)
api_router.include_router(spatial.router)
api_router.include_router(reports.router)
