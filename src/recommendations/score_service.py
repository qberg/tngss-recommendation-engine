"""Service for score calculation and database storage."""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from pymongo import ASCENDING, DESCENDING, IndexModel, UpdateOne
from pymongo.asynchronous.database import AsyncDatabase

from src.config import settings
from src.embeddings.service import EmbeddingService
from src.utils.setup_logger import setup_logger

logger = setup_logger(__name__, "logs/score_service.log")


class ScoreService:
    """Handles score calculation and database operations."""

    def __init__(self, db: AsyncDatabase):
        self.db = db
        self.embedding_service = EmbeddingService()

    async def initialize_collection(self) -> bool:
        """Initialize recommendations collection with indexes (idempotent)."""
        try:
            collection = self.db[settings.RECOMMENDATIONS_COLLECTION]

            indexes = [
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

            await collection.create_indexes(indexes)
            logger.info(
                f"[SUCCESS] Initialized {settings.RECOMMENDATIONS_COLLECTION} collection with indexes"
            )
            return True

        except Exception as e:
            logger.error(f"[FAILED] Error initializing collection: {e}")
            return False

    async def clear_all_scores(self) -> Dict[str, Any]:
        """
        Clear all documents from recommendations collection.
        WARNING: This deletes all recommendation scores for all users.
        """
        try:
            collection = self.db[settings.RECOMMENDATIONS_COLLECTION]

            logger.warning(
                f"[WARNING] Clearing all documents from {settings.RECOMMENDATIONS_COLLECTION}"
            )
            result = await collection.delete_many({})

            logger.info(
                f"[SUCCESS] Deleted {result.deleted_count} documents from collection"
            )

            return {
                "success": True,
                "deleted_count": result.deleted_count,
                "message": f"Cleared {result.deleted_count} recommendation scores",
            }

        except Exception as e:
            logger.error(f"[FAILED] Error clearing collection: {e}")
            return {
                "success": False,
                "deleted_count": 0,
                "message": f"Failed to clear collection: {str(e)}",
            }

    def normalize_scores_to_percentage(
        self, recommendation_scores: List[Dict]
    ) -> List[Dict]:
        """Normalize similarity scores to percentage range (10-95%)"""
        if not recommendation_scores:
            return recommendation_scores

        scores = [item["similarity_score"] for item in recommendation_scores]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            for item in recommendation_scores:
                item["percentage_score"] = 52.5
            return recommendation_scores

        for item in recommendation_scores:
            original_score = item["similarity_score"]
            normalized = (
                10 + ((original_score - min_score) / (max_score - min_score)) * 85
            )
            item["percentage_score"] = normalized

        return recommendation_scores

    async def get_stored_scores(
        self, user_id: str, reference_type: str = "event", max_age_hours: int = 24
    ) -> Optional[List[Dict[str, Any]]]:
        """Get stored scores from database."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

            scores = (
                await self.db[settings.RECOMMENDATIONS_COLLECTION]
                .find(
                    {
                        "user_id": user_id,
                        "reference_type": reference_type,
                        "updated_at": {"$gte": cutoff_time},
                    }
                )
                .sort("score", -1)
                .to_list(None)
            )

            if scores:
                logger.info(
                    f"[INFO] Found {len(scores)} stored scores for user {user_id[:8]}"
                )
                return scores

            return None

        except Exception as e:
            logger.error(f"[FAILED] Error fetching stored scores: {e}")
            return None

    async def store_scores_to_database(self, scores: List[Dict[str, Any]]) -> bool:
        """Store scores to database."""
        try:
            if not scores:
                logger.warning("[WARNING] No scores to store")
                return False

            operations = []
            now = datetime.now()

            for score in scores:
                doc = {
                    "user_id": score["user_id"],
                    "reference_id": score["target_id"],
                    "reference_type": "event",
                    "score": round(score["percentage_score"]),
                    "updated_at": now,
                }

                operations.append(
                    UpdateOne(
                        {
                            "user_id": doc["user_id"],
                            "reference_id": doc["reference_id"],
                            "reference_type": "event",
                        },
                        {"$set": doc, "$setOnInsert": {"created_at": now}},
                        upsert=True,
                    )
                )

            result = await self.db[settings.RECOMMENDATIONS_COLLECTION].bulk_write(
                operations
            )

            total_stored = result.upserted_count + result.modified_count
            logger.info(f"[SUCCESS] Stored {total_stored} recommendation scores")

            return True

        except Exception as e:
            logger.error(f"[FAILED] Error storing scores to database: {e}")
            return False
