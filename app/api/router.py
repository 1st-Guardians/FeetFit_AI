from fastapi import APIRouter

from app.api.routes import (
    environment,
    photo_analysis,
    pressure_heatmaps,
    pressed_regions,
    reports,
    shoes,
)


api_router = APIRouter()
api_router.include_router(reports.router, prefix="/reports", tags=["reports"])
api_router.include_router(
    photo_analysis.router,
    prefix="/reports/photo-analysis",
    tags=["Feetfit Analysis"],
)
api_router.include_router(shoes.router, prefix="/shoes", tags=["shoes"])
api_router.include_router(
    pressed_regions.router,
    prefix="/reports/daily-foot-analysis/plantar-footprint",
    tags=["pressed-regions"],
)
api_router.include_router(
    pressure_heatmaps.router,
    prefix="/reports/daily-foot-analysis/pressure-heatmap",
    tags=["Feetfit Analysis"],
)
api_router.include_router(
    environment.router,
    prefix="/reports/daily-foot-analysis/environment",
    tags=["Feetfit Analysis"],
)
