from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorDatabase
from ..core.dependencies import get_db, get_current_user
from ..services.dashboard_service import get_dashboard_data
from ..utils.response import success_response, error_response
from typing import Dict, Any

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

@router.get("")
async def get_dashboard(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db)
):
    try:
        dashboard_data = await get_dashboard_data(current_user, db)
        return success_response(
            message="Dashboard data retrieved successfully",
            data=dashboard_data
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=error_response(message="An unexpected error occurred while fetching dashboard data")
        )
