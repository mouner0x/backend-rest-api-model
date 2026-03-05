from motor.motor_asyncio import AsyncIOMotorDatabase
from typing import Dict, Any

async def get_dashboard_data(user: Dict[str, Any], db: AsyncIOMotorDatabase) -> dict:
    from bson import ObjectId
    user_id = user["_id"]
    if isinstance(user_id, str):
        try:
            user_id = ObjectId(user_id)
        except:
            pass
            
    # greeting.name
    full_name = user.get("full_name", "")
    first_name = full_name.split(" ")[0] if full_name else "User"
    
    # stats
    datasets_uploaded = await db.datasets.count_documents({"user_id": user_id})
    models_trained = await db.models.count_documents({"user_id": user_id, "status": "trained"})
    
    # latest model
    latest_models_cursor = db.models.find({"user_id": user_id, "status": "trained"}).sort("trained_at", -1).limit(1)
    latest_models = await latest_models_cursor.to_list(length=1)
    
    latest_model_data = None
    if latest_models:
        latest_model = latest_models[0]
        metrics = latest_model.get("metrics", {})
        r2 = metrics.get("r2_score")
        if r2 is None: # fallback for old documents
            r2 = latest_model.get("r2_score")
        if r2 is not None:
            latest_model_data = {"r2_score": r2}
            
    # recent models
    recent_models_cursor = db.models.find({"user_id": user_id, "status": "trained"}).sort("trained_at", -1).limit(5)
    recent_models_list = await recent_models_cursor.to_list(length=5)
    
    recent_models = []
    for model in recent_models_list:
        model_type = model.get("model_type", "")
        if model_type == "Linear Regression":
            model_type = "Linear"
            
        dataset_name = model.get("dataset_name")
        if not dataset_name: # fallback for old models
            dataset_name = "Unknown Dataset"
            dataset_id = model.get("dataset_id")
            if dataset_id:
                dataset = await db.datasets.find_one({"_id": dataset_id})
                if dataset:
                    dataset_name = dataset.get("filename", "Unknown Dataset")
                    
        metrics = model.get("metrics", {})
        r2 = metrics.get("r2_score")
        if r2 is None:
            r2 = model.get("r2_score")
            
        recent_models.append({
            "model_id": str(model["_id"]),
            "name": model.get("model_name", "Unknown"),
            "type": model_type,
            "dataset": dataset_name,
            "r2_score": r2
        })
        
    response = {
        "greeting": {
            "name": first_name
        },
        "stats": {
            "datasets_uploaded": datasets_uploaded,
            "models_trained": models_trained,
            "latest_model": latest_model_data
        },
        "recent_models": recent_models
    }
    
    from app.utils.json_sanitizer import sanitize_for_json
    response = sanitize_for_json(response)
    return response
