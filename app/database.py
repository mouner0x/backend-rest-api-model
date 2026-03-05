from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings

# Create a global database client
client = AsyncIOMotorClient(settings.MONGODB_URL)
db = client[settings.DATABASE_NAME]

async def get_database():
    return db
