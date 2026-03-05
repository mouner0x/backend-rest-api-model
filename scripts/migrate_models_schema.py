import asyncio
import os
import sys

# Modify python path to allow importing app modules if running from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

# You should replace this with your actual MongoDB URI if different, or load from .env
MONGO_DETAILS = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("MONGODB_DB_NAME", "modelix_db")

async def migrate_models_collection():
    print(f"Connecting to MongoDB at {MONGO_DETAILS}...")
    client = AsyncIOMotorClient(MONGO_DETAILS)
    db = client[DATABASE_NAME]
    
    models_collection = db["models"]
    datasets_collection = db["datasets"]
    
    # Find all models missing user_id or dataset_name or created_at
    query = {
        "$or": [
            {"user_id": {"$exists": False}},
            {"dataset_name": {"$exists": False}},
            {"created_at": {"$exists": False}}
        ]
    }
    
    cursor = models_collection.find(query)
    models_to_update = await cursor.to_list(length=None)
    
    print(f"Found {len(models_to_update)} models requiring structural migration.")
    
    updated_count = 0
    errors_count = 0
    
    for model in models_to_update:
        model_id = model["_id"]
        dataset_id = model.get("dataset_id")
        
        updates = {}
        
        # 1. Backfill user_id and dataset_name from datasets collection
        if dataset_id:
            try:
                if isinstance(dataset_id, str):
                    dataset_id = ObjectId(dataset_id)
                
                dataset = await datasets_collection.find_one({"_id": dataset_id})
                
                if dataset:
                    if "user_id" not in model:
                        updates["user_id"] = dataset.get("user_id")
                        print(f"[{model_id}] Recovered user_id from dataset mapping.")
                        
                    if "dataset_name" not in model:
                        updates["dataset_name"] = dataset.get("filename", dataset.get("file_name", "Unknown Dataset"))
                        print(f"[{model_id}] Recovered dataset_name from dataset mapping.")
                        
                    if "target_column" not in model:
                        updates["target_column"] = dataset.get("target_column")
                else:
                    print(f"[{model_id}] WARNING: Associated dataset {dataset_id} not found in DB.")
            except Exception as e:
                print(f"[{model_id}] ERROR parsing dataset {dataset_id}: {str(e)}")
        else:
            print(f"[{model_id}] WARNING: Model has no dataset_id to map from.")
            
        # 2. Enforce structural schema nulls
        if "metrics" not in model:
            updates["metrics"] = None
        if "model_path" not in model:
            updates["model_path"] = None
        if "trained_at" not in model:
            updates["trained_at"] = None
            
        if "created_at" not in model:
            # Fallback to ObjectId generation timestamp if not physically logged
            from datetime import timezone
            updates["created_at"] = model_id.generation_time.replace(tzinfo=timezone.utc)
            print(f"[{model_id}] Recovered created_at from ObjectId generation timestamp.")

        if updates:
            try:
                await models_collection.update_one(
                    {"_id": model_id},
                    {"$set": updates}
                )
                updated_count += 1
            except Exception as e:
                print(f"[{model_id}] ERROR updating model: {str(e)}")
                errors_count += 1
                
    print("\n--- Migration Complete ---")
    print(f"Models successfully migrated: {updated_count}")
    print(f"Errors encountered: {errors_count}")

if __name__ == "__main__":
    asyncio.run(migrate_models_collection())
