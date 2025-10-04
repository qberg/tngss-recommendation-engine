from enum import unique
from os import name
from typing import Dict, Optional

from pymongo import ASCENDING, DESCENDING, AsyncMongoClient, IndexModel
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


async def initialize_indexes():
    """
    Create all necessary indexes for the recommendation system.
    Called automatically on startup. Idempotent - safe to run multiple times.
    """
    try:
        db = mongodb.get_database()
        logger.info("[***] Initializing database indexes...")

        await db[settings.USERS_PROFILE_COLLECTION].create_index(
            [("user_id", ASCENDING)], name="user_id_1", unique=True
        )
        logger.info(
            f"[SUCCESS] Index created on {settings.USERS_PROFILE_COLLECTION}.user_id"
        )

        organisation_profile_indexes = [
            IndexModel([("profile_type", ASCENDING)], name="profile_type_1"),
            IndexModel([("user_id", ASCENDING)], name="user_id_1", unique=True),
        ]

        await db[settings.ORGANISATION_PROFILE_COLLECTION].create_indexes(
            organisation_profile_indexes
        )

        logger.info(
            f"[SUCCESS] Index created on {settings.ORGANISATION_PROFILE_COLLECTION}.profile_type"
        )

        context_builder_indexes = [
            IndexModel([("user_id", DESCENDING)], name="user_id_-1", unique=True),
            IndexModel([("sector.value", ASCENDING)], name="sector_value_1"),
            IndexModel(
                [("looking_to_connect.value", ASCENDING)],
                name="looking_to_connect_value_1",
            ),
        ]
        await db[settings.CONTEXT_BUILDER_COLLECTION].create_indexes(
            context_builder_indexes
        )

        logger.info(
            f"[SUCCESS] Indexes created on {settings.CONTEXT_BUILDER_COLLECTION}.user_id"
        )

        recommendation_indexes = [
            IndexModel(
                [
                    ("user_id", ASCENDING),
                    ("reference_type", ASCENDING),
                    ("updated_at", DESCENDING),
                ],
                name="user_ref_type_updated_idx",
            ),
            IndexModel(
                [
                    ("user_id", ASCENDING),
                    ("reference_type", ASCENDING),
                    ("score", DESCENDING),
                ],
                name="user_ref_type_score_idx",
            ),
            IndexModel(
                [
                    ("user_id", ASCENDING),
                    ("reference_id", ASCENDING),
                    ("reference_type", ASCENDING),
                ],
                name="user_ref_unique_idx",
                unique=True,
            ),
        ]

        await db[settings.RECOMMENDATIONS_COLLECTION].create_indexes(
            recommendation_indexes
        )
        logger.info(
            f"[SUCCESS] Indexes created on {settings.RECOMMENDATIONS_COLLECTION}"
        )

        user_recommendation_indexes = [
            IndexModel(
                [
                    ("user_id", ASCENDING),
                    ("score", DESCENDING),
                ],
                name="user_score_idx",
            ),
            IndexModel(
                [
                    ("user_id", ASCENDING),
                    ("matched_user_id", ASCENDING),
                ],
                name="user_matched_unique",
            ),
            IndexModel(
                [("matched_user_id", ASCENDING), ("updated_at", DESCENDING)],
                name="matched_updated_idx",
            ),
        ]

        await db[settings.USER_RECOMMENDATIONS_COLLECTION].create_indexes(
            user_recommendation_indexes
        )

        logger.info(
            f"[SUCCESS] Indexes created on {settings.USER_RECOMMENDATIONS_COLLECTION}"
        )

        for collection_name in [
            settings.USERS_PROFILE_COLLECTION,
            settings.ORGANISATION_PROFILE_COLLECTION,
            settings.CONTEXT_BUILDER_COLLECTION,
            settings.RECOMMENDATIONS_COLLECTION,
            settings.USER_RECOMMENDATIONS_COLLECTION,
        ]:
            indexes = await db[collection_name].index_information()
            logger.info(f"[INDEXES] {collection_name}: {list(indexes.keys())}")

        logger.info("[SUCCESS] All indexes initialized")
        return True

    except Exception as e:
        logger.error(f"[FAILED] Error initializing indexes: {e}")
        return False


async def connect_to_mongo():
    """Create database connection"""
    try:
        mongodb.client = AsyncMongoClient(
            settings.MONGODB_URL,
            maxPoolSize=50,
            minPoolSize=10,
            maxIdleTimeMS=45000,
            connectTimeoutMS=30000,
            serverSelectionTimeoutMS=5000,
        )
        mongodb.database = mongodb.client[settings.DATABASE_NAME]

        # Test the connection
        await mongodb.client.admin.command("ping")
        logger.info(
            f"[SUCCESS] Successfully connected to MongoDB: {settings.DATABASE_NAME}"
        )
        await initialize_indexes()

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
