from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorDatabase
from ..schemas.user_schema import UserRegister, UserLogin
from ..services.auth_service import register_user, login_user
from ..core.dependencies import get_db
from ..utils.response import success_response, error_response

router = APIRouter(prefix="/auth", tags=["Auth"])

@router.post("/register")
async def register(user_data: UserRegister, db: AsyncIOMotorDatabase = Depends(get_db)):
    try:
        user_response = await register_user(user_data, db)
        return success_response(
            message="Account created successfully",
            data=user_response
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

@router.post("/login")
async def login(user_data: UserLogin, db: AsyncIOMotorDatabase = Depends(get_db)):
    try:
        user_response = await login_user(user_data, db)
        return success_response(
            message="Login successful",
            data=user_response
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
