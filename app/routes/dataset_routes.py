from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorDatabase
from ..core.dependencies import get_db, get_current_user
from ..services.dataset_service import process_and_save_dataset, get_dataset_details, delete_user_dataset, set_target_column, get_target_options
from ..utils.response import success_response, error_response
from typing import Dict, Any

class TargetSelection(BaseModel):
    target_column: str = Field(..., description="The name of the column to set as target for regression")

router = APIRouter(
    prefix="/datasets",
    tags=["Datasets"]
)

@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db)
):
    try:
        dataset_data = await process_and_save_dataset(file, current_user, db)
        return success_response(
            message="Dataset uploaded successfully",
            data=dataset_data
        )
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content=error_response(message=e.detail)
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=error_response(message=f"An unexpected error occurred: {str(e)}")
        )

@router.get("/{dataset_id}")
async def get_dataset(
    dataset_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db)
):
    try:
        dataset_data = await get_dataset_details(dataset_id, current_user, db)
        return success_response(
            message="Dataset details retrieved successfully",
            data=dataset_data
        )
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content=error_response(message=e.detail)
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=error_response(message="An unexpected error occurred")
        )

@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db)
):
    try:
        result = await delete_user_dataset(dataset_id, current_user, db)
        return success_response(
            message=result["message"]
        )
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content=error_response(message=e.detail)
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=error_response(message="An unexpected error occurred")
        )

@router.post("/{dataset_id}/select-target")
async def select_target(
    dataset_id: str,
    payload: TargetSelection = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db)
):
    try:
        result = await set_target_column(dataset_id, payload.target_column, current_user, db)
        return success_response(
            message="Target column selected successfully",
            data=result
        )
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content=error_response(message=e.detail)
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=error_response(message="An unexpected error occurred")
        )

@router.get("/{dataset_id}/target-options")
async def target_options(
    dataset_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db)
):
    try:
        result = await get_target_options(dataset_id, current_user, db)
        return success_response(
            message="Target options retrieved successfully",
            data=result
        )
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content=error_response(message=e.detail)
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=error_response(message="An unexpected error occurred")
        )
