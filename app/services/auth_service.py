from fastapi import HTTPException
from motor.motor_asyncio import AsyncIOMotorDatabase
from app.schemas.user_schema import UserRegister, UserLogin
from app.models.user_model import UserModel
from app.core.security import get_password_hash, create_access_token, verify_password

async def register_user(user_data: UserRegister, db: AsyncIOMotorDatabase) -> dict:
    # Check if user already exists
    existing_user = await db.users.find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
        
    # Hash password
    hashed_password = get_password_hash(user_data.password)
    
    # Create user model instance representation
    user_dict = user_data.model_dump(exclude={"confirm_password"})
    user_dict["hashed_password"] = hashed_password
    del user_dict["password"]
    
    # Validation/Formatting via UserModel
    new_user = UserModel(**user_dict)
    
    # Insert to DB
    result = await db.users.insert_one(new_user.model_dump(by_alias=True, exclude_none=True))
    inserted_id = str(result.inserted_id)
    
    # Generate Token
    access_token = create_access_token(data={"sub": inserted_id})
    
    response = {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": inserted_id,
            "full_name": new_user.full_name,
            "email": new_user.email
        }
    }
    from app.utils.json_sanitizer import sanitize_for_json
    response = sanitize_for_json(response)
    return response

async def login_user(user_data: UserLogin, db: AsyncIOMotorDatabase) -> dict:
    email = user_data.email.strip()
    
    # Case insensitive search using regex
    user = await db.users.find_one({"email": {"$regex": f"^{email}$", "$options": "i"}})
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
        
    if not verify_password(user_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
        
    # Generate Token
    user_id = str(user["_id"])
    access_token = create_access_token(data={"sub": user_id, "email": user["email"]})
    
    response = {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user_id,
            "full_name": user["full_name"],
            "email": user["email"]
        }
    }
    from app.utils.json_sanitizer import sanitize_for_json
    response = sanitize_for_json(response)
    return response
