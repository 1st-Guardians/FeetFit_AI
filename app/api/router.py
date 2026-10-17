from fastapi import APIRouter

from app.api.routes import reports, shoe_ingestion, shoes


api_router = APIRouter()
api_router.include_router(reports.router, prefix="/reports", tags=["reports"])
api_router.include_router(shoes.router, prefix="/shoes", tags=["shoes"])
api_router.include_router(
    shoe_ingestion.router,
    prefix="/admin/shoe-ingestion",
    tags=["관리자 - 신발 DB 적재"],
)
