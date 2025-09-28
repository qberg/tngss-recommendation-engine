from typing import Dict, Optional

from pymongo import AsyncMongoClient
from pymongo.asynchronous import database
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import ConnectionFailure

from src.config import settings
from src.utils.setup_logger import setup_logger

logger = setup_logger(__name__, "logs/database.log")


class MongoDB:
    client: Optional[AsyncMongoClient] = None
    database: Optional[AsyncDatabase] = None

    def _ensure_connected(self) -> AsyncDatabase:
        """
        Ensure database is connected and return database instance.
        Raises RuntimeError if not connected.
        """
        if self.database is None:
            raise RuntimeError("Database not connected. Call connect_to_mongo() first.")
        return self.database

    def get_database(self) -> AsyncDatabase:
        """Get database instance"""
        return self._ensure_connected()


# Global MongoDB instance
mongodb = MongoDB()


async def connect_to_mongo():
    """Create database connection"""
    try:
        mongodb.client = AsyncMongoClient(settings.MONGODB_URL)
        mongodb.database = mongodb.client[settings.DATABASE_NAME]

        # Test the connection
        await mongodb.client.admin.command("ping")
        logger.info(
            f"[SUCCESS] Successfully connected to MongoDB: {settings.DATABASE_NAME}"
        )

    except ConnectionFailure as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise e


async def close_mongo_connection():
    """Close database connection"""
    if mongodb.client:
        await mongodb.client.close()
        logger.info("Disconnected from MongoDB")


def get_database() -> AsyncDatabase:
    """Get database instance"""
    return mongodb.get_database()


async def inspect_collection_schema(collection_name: str) -> Dict[str, str]:
    """Inspect schema of a collection by sampling documents"""
    try:
        db = get_database()

        sample = await db[collection_name].find_one({})
        if not sample:
            logger.warning(f"No documents found in {collection_name}")
            return {}

        schema = {}
        for key, value in sample.items():
            if value is None:
                schema[key] = "Optional"
            else:
                schema[key] = type(value).__name__

        logger.info(f"[INFO] {collection_name}: {len(schema)} fields")
        return schema

    except Exception as e:
        logger.error(f"Schema inspection failed for {collection_name}: {e}")
        return {}


async def inspect_all_schemas() -> Dict[str, Dict[str, str]]:
    """Inspect schemas for all collections used in recommendations"""
    collections = {
        "login_info": settings.LOGIN_COLLECTION,
        "user_profile": settings.USERS_PROFILE_COLLECTION,
        "org_profile": settings.ORGANISATION_PROFILE_COLLECTION,
        "context_builder": settings.CONTEXT_BUILDER_COLLECTION,
        "events": settings.EVENTS_COLLECTION,
    }

    schemas = {}
    for name, collection_name in collections.items():
        schemas[name] = await inspect_collection_schema(collection_name)

    return schemas
