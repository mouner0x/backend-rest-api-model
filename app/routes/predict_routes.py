from fastapi import APIRouter, Depends
from typing import Dict, Any
from motor.motor_asyncio import AsyncIOMotorDatabase
from ..core.dependencies import get_current_user, get_db
from ..services.model_service import (
    create_selected_model,
    predict_pipeline_service,
    get_model_service,
)
from ..services.train_service import train_model_service

router = APIRouter(prefix="/models", tags=["Models"])

@router.get("/{model_id}")
async def get_model(
    model_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    return await get_model_service(model_id, current_user)


@router.post("/select")
async def select_model(
    data: dict,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    return await create_selected_model(data, current_user)

@router.post("/train")
async def train_model(
    data: dict,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db)
):
    return await train_model_service(data["model_id"], current_user, db)

@router.post("/{model_id}/predict")
async def predict(
    model_id: str,
    input_data: dict,
):
    return await predict_pipeline_service(model_id, input_data)
