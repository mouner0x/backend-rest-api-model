import asyncio
import os
import sys

# Modify python path to allow importing app modules if running from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

# Re-configure these to match your actual dev/prod MongoDB URI
MONGO_DETAILS = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("MONGODB_DB_NAME", "modelix_db")

async def patch_models_collection():
    print(f"Connecting to MongoDB at {MONGO_DETAILS}...")
    client = AsyncIOMotorClient(MONGO_DETAILS)
    db = client[DATABASE_NAME]
    
    models_collection = db["models"]
    datasets_collection = db["datasets"]
    
    # Query for models that are missing any of the essential fields
    query = {
        "$or": [
            {"user_id": {"$exists": False}},
            {"dataset_name": {"$exists": False}},
            {"target_column": {"$exists": False}},
            {"metrics": {"$exists": False}},
            {"trained_at": {"$exists": False}}
        ]
    }
    
    cursor = models_collection.find(query)
    models_to_patch = await cursor.to_list(length=None)
    
    print(f"Found {len(models_to_patch)} documents requiring patching.")
    
    patched_count = 0
    errors_count = 0
    
    for model in models_to_patch:
        model_id = model["_id"]
        dataset_id = model.get("dataset_id")
        
        set_updates = {}
        
        # 1. Backfill mapping data structurally ONLY if missing
        if dataset_id:
            try:
                if isinstance(dataset_id, str):
                    try:
                        dataset_id = ObjectId(dataset_id)
                    except:
                        pass
                
                dataset = await datasets_collection.find_one({"_id": dataset_id})
                if dataset:
                    if "user_id" not in model:
                        set_updates["user_id"] = dataset.get("user_id")
                        
                    if "dataset_name" not in model:
                        set_updates["dataset_name"] = dataset.get("filename", dataset.get("file_name", "Unknown Dataset"))
                        
                    if "target_column" not in model:
                        set_updates["target_column"] = dataset.get("target_column")
                else:
                    print(f"[{model_id}] WARNING: Linked dataset {dataset_id} not found.")
            except Exception as e:
                print(f"[{model_id}] ERROR extracting dataset {dataset_id}: {str(e)}")
                
        # 2. Patch missing base schema fields with `None` safely
        if "metrics" not in model:
            set_updates["metrics"] = None
            
        if "trained_at" not in model:
            set_updates["trained_at"] = None

        if set_updates:
            try:
                await models_collection.update_one(
                    {"_id": model_id},
                    {"$set": set_updates}
                )
                patched_count += 1
                print(f"[{model_id}] Patched fields: {list(set_updates.keys())}")
            except Exception as e:
                print(f"[{model_id}] ERROR patching document: {str(e)}")
                errors_count += 1
                
    print("\n--- MongoDB Patching Complete ---")
    print(f"Documents correctly patched: {patched_count}")
    print(f"Errors encountered: {errors_count}")

if __name__ == "__main__":
    asyncio.run(patch_models_collection())
