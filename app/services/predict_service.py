import os
import pandas as pd
import joblib
from fastapi import HTTPException
from motor.motor_asyncio import AsyncIOMotorDatabase
from typing import Dict, Any

async def predict_pipeline_service(model_id: str, input_data: dict, current_user: Dict[str, Any], db: AsyncIOMotorDatabase) -> float:
    from bson import ObjectId
    user_id = current_user["_id"]
    if isinstance(user_id, str):
        try:
            user_id = ObjectId(user_id)
        except:
            pass
            
    try:
        obj_model_id = ObjectId(model_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid model ID format")

    model_doc = await db.models.find_one({
        "_id": obj_model_id,
        "user_id": user_id
    })
    
    if not model_doc:
        raise HTTPException(status_code=404, detail="Model not found or you don't have access")
        
    if model_doc.get("status") != "trained":
        raise HTTPException(status_code=400, detail="Model must be in 'trained' status to make predictions")

    model_file_path = model_doc.get("model_path")
    if not model_file_path or not os.path.exists(model_file_path):
        model_file_path = f"models/{model_id}.pkl"
        if not os.path.exists(model_file_path):
            raise HTTPException(status_code=500, detail="Model pipeline file (.pkl) not found on disk")
        
    try:
        pipeline = joblib.load(model_file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading trained model pipeline: {str(e)}")

    try:
        input_df = pd.DataFrame([input_data])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error formatting input data schema: {str(e)}")

    try:
        prediction = pipeline.predict(input_df)[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error evaluating prediction pipeline: {str(e)}")

    response = {"prediction":float(prediction)}
    from app.utils.json_sanitizer import sanitize_for_json
    response = sanitize_for_json(response)
    return response
