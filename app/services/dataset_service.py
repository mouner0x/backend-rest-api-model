import os
import pandas as pd
from datetime import datetime, timezone
from fastapi import UploadFile, HTTPException
from motor.motor_asyncio import AsyncIOMotorDatabase
from typing import Dict, Any

MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".csv", ".txt"}
UPLOAD_DIR = "uploads"

async def process_and_save_dataset(file: UploadFile, user: Dict[str, Any], db: AsyncIOMotorDatabase) -> dict:
    from bson import ObjectId
    user_id = user["_id"]
    if isinstance(user_id, str):
        try:
            user_id = ObjectId(user_id)
        except:
            pass

    # 1. Validation
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="No filename provided")
        
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid file extension. Only .csv and .txt are allowed.")
        
    # Read file content safely
    content = await file.read()
    
    # Check max size
    file_size_bytes = len(content)
    if file_size_bytes > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE_MB}MB.")
        
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
    
    # 2. File Storage
    user_upload_dir = os.path.join(UPLOAD_DIR, str(user_id))
    os.makedirs(user_upload_dir, exist_ok=True)
    
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    safe_filename = f"{timestamp}_{filename}"
    file_path = os.path.join(user_upload_dir, safe_filename)
    
    with open(file_path, "wb") as f:
        f.write(content)
        
    # 3. Data Analysis using pandas
    try:
        # read the saved file just to be safe it's well-formed
        df = pd.read_csv(file_path)
    except Exception as e:
        # Delete the corrupted file
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"Error parsing CSV file: {str(e)}")
        
    rows_count = len(df)
    columns_count = len(df.columns)
    
    if rows_count == 0 or columns_count == 0:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail="The uploaded dataset is empty.")
        
    # Preview (first 6 rows)
    preview_df = df.head(6)
    preview = preview_df.to_dict(orient="records")
    
    # Column Analysis
    column_analysis = []
    for col in df.columns:
        dtype = df[col].dtype
        
        # Check if numeric
        col_type = "numeric" if pd.api.types.is_numeric_dtype(dtype) else "categorical"
        
        missing_values = int(df[col].isna().sum())
        
        column_analysis.append({
            "column": str(col),
            "type": col_type,
            "missing_values": missing_values
        })
        
    # 4. Database Insert
    dataset_doc = {
        "user_id": user_id,
        "filename": filename,
        "file_path": file_path,
        "file_size_mb": file_size_mb,
        "rows_count": rows_count,
        "columns_count": columns_count,
        "target_column": None,
        "created_at": datetime.now(timezone.utc)
    }
    
    result = await db.datasets.insert_one(dataset_doc)
    dataset_id = str(result.inserted_id)
    
    response = {
        "dataset_id": dataset_id,
        "file_name": filename,
        "uploaded_at": dataset_doc["created_at"].isoformat(),
        "file_size_mb": file_size_mb,
        "rows": rows_count,
        "columns": columns_count,
        "preview": preview,
        "column_analysis": column_analysis
    }
    from app.utils.json_sanitizer import sanitize_for_json
    response = sanitize_for_json(response)
    return response

async def get_dataset_details(dataset_id: str, user: Dict[str, Any], db: AsyncIOMotorDatabase) -> dict:
    from bson import ObjectId
    user_id = user["_id"]
    if isinstance(user_id, str):
        try:
            user_id = ObjectId(user_id)
        except:
            pass
            
    try:
        obj_dataset_id = ObjectId(dataset_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid dataset ID format")
        
    dataset = await db.datasets.find_one({
        "_id": obj_dataset_id,
        "user_id": user_id
    })
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found or you don't have access")
        
    # Standardize output to dict representation
    dataset["_id"] = str(dataset["_id"])
    dataset["user_id"] = str(dataset["user_id"])
    response = dataset
    from app.utils.json_sanitizer import sanitize_for_json
    response = sanitize_for_json(response)
    return response

async def delete_user_dataset(dataset_id: str, user: Dict[str, Any], db: AsyncIOMotorDatabase) -> dict:
    from bson import ObjectId
    user_id = user["_id"]
    if isinstance(user_id, str):
        try:
            user_id = ObjectId(user_id)
        except:
            pass
            
    try:
        obj_dataset_id = ObjectId(dataset_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid dataset ID format")
        
    dataset = await db.datasets.find_one({
        "_id": obj_dataset_id,
        "user_id": user_id
    })
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found or you don't have access")
        
    # Delete file from disk
    file_path = dataset.get("file_path")
    if file_path and os.path.exists(file_path):
        os.remove(file_path)
        
    # Delete from DB
    await db.datasets.delete_one({"_id": obj_dataset_id})
    response = {"message": "Dataset deleted successfully"}
    from app.utils.json_sanitizer import sanitize_for_json
    response = sanitize_for_json(response)
    return response

async def set_target_column(dataset_id: str, target_column: str, user: Dict[str, Any], db: AsyncIOMotorDatabase) -> dict:
    from bson import ObjectId
    user_id = user["_id"]
    if isinstance(user_id, str):
        try:
            user_id = ObjectId(user_id)
        except:
            pass
            
    try:
        obj_dataset_id = ObjectId(dataset_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid dataset ID format")
        
    dataset = await db.datasets.find_one({
        "_id": obj_dataset_id,
        "user_id": user_id
    })
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found or you don't have access")
        
    file_path = dataset.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=400, detail="Dataset file missing on disk")
        
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading dataset: {str(e)}")
        
    if target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{target_column}' does not exist in the dataset")
        
    if not pd.api.types.is_numeric_dtype(df[target_column].dtype):
        raise HTTPException(status_code=400, detail=f"Target column '{target_column}' must be numeric for regression")
        
    # Update document
    await db.datasets.update_one(
        {"_id": obj_dataset_id},
        {"$set": {"target_column": target_column}}
    )
    
    response = {
        "dataset_id": dataset_id,
        "target_column": target_column,
        "problem_type": "regression"
    }
    from app.utils.json_sanitizer import sanitize_for_json
    response = sanitize_for_json(response)
    return response

async def get_target_options(dataset_id: str, user: Dict[str, Any], db: AsyncIOMotorDatabase) -> dict:
    from bson import ObjectId
    user_id = user["_id"]
    if isinstance(user_id, str):
        try:
            user_id = ObjectId(user_id)
        except:
            pass
            
    try:
        obj_dataset_id = ObjectId(dataset_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid dataset ID format")
        
    dataset = await db.datasets.find_one({
        "_id": obj_dataset_id,
        "user_id": user_id
    })
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found or you don't have access")
        
    file_path = dataset.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=500, detail="Dataset file missing on disk")
        
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading dataset: {str(e)}")
        
    numeric_columns = []
    current_target = dataset.get("target_column")
    
    for col in df.columns:
        # Exclude currently selected target
        if current_target and col == current_target:
            continue
            
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(df[col].dtype):
            # Exclude columns that are entirely null
            if df[col].isna().all():
                continue
            numeric_columns.append(str(col))
            
    response = {
        "dataset_id": dataset_id,
        "numeric_columns": numeric_columns
    }
    from app.utils.json_sanitizer import sanitize_for_json
    response = sanitize_for_json(response)
    return response
