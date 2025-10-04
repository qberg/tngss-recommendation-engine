"""Service for score calculation and database storage."""

from datetime import datetime, timedelta
from time import perf_counter, time
from typing import Any, Dict, List, Optional

from bson import ObjectId
from pymongo import UpdateOne
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
        self,
        user_id: str,
        reference_type: str = "event",
        max_age_hours: int = 24,
        limit: int = 100,
    ) -> Optional[List[Dict[str, Any]]]:
        """Get stored scores from database."""
        start_time = time()
        try:

            user_id_obj = ObjectId(user_id) if isinstance(user_id, str) else user_id
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

            scores = (
                await self.db[settings.RECOMMENDATIONS_COLLECTION]
                .find(
                    {
                        "user_id": user_id_obj,
                        "reference_type": reference_type,
                        "updated_at": {"$gte": cutoff_time},
                    }
                )
                .sort("score", -1)
                .limit(limit)
                .to_list(None)
            )

            if scores:
                logger.info(
                    f"[INFO] Found {len(scores)} stored scores for user {user_id}"
                )
                for score in scores:
                    score["user_id"] = str(score["user_id"])
                    score["reference_id"] = str(score["reference_id"])

                elapsed_time = time() - start_time
                logger.info(
                    f"[INFO] Found stored scores found for user {user_id} in {elapsed_time:.3f}s"
                )
                return scores

            elapsed_time = time() - start_time
            logger.info(
                f"[INFO] No stored scores found for user {user_id} in {elapsed_time:.3f}s"
            )

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

            start_time = perf_counter()
            operations = []
            now = datetime.now()

            for score in scores:
                user_id = (
                    ObjectId(score["user_id"])
                    if isinstance(score["user_id"], str)
                    else score["user_id"]
                )
                reference_id = (
                    ObjectId(score["target_id"])
                    if isinstance(score["target_id"], str)
                    else score["target_id"]
                )
                doc = {
                    "user_id": user_id,
                    "reference_id": reference_id,
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
                operations, ordered=False
            )

            elapsed = (perf_counter() - start_time) * 1000

            total_stored = result.upserted_count + result.modified_count
            logger.info(
                f"[SUCCESS] Stored {total_stored} scores in {elapsed:.2f}ms "
                f"({len(scores)/elapsed*1000:.0f} scores/sec)"
            )

            return True

        except Exception as e:
            logger.error(f"[FAILED] Error storing scores to database: {e}")
            return False
