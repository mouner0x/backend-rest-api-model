import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os
import shutil
from dotenv import load_dotenv

load_dotenv()

async def cleanup():
    mongo_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    db_name = os.getenv("DATABASE_NAME", "modelix_db")
    print(f"Connecting to {mongo_url}, database: {db_name}")
    client = AsyncIOMotorClient(mongo_url)
    db = client[db_name]
    await db.models.drop()
    if os.path.exists("models"):
        shutil.rmtree("models")
    if os.path.exists("models_storage"):
        shutil.rmtree("models_storage")
    print("Cleanup done")

if __name__ == "__main__":
    asyncio.run(cleanup())
