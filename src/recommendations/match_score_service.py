"""Service for storing and retrieving user match scores."""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from bson import ObjectId
from pymongo import DESCENDING, UpdateOne
from pymongo.asynchronous.database import AsyncDatabase

from src.config import settings
from src.utils.setup_logger import setup_logger

logger = setup_logger(__name__, "logs/match_score_service.log")


class MatchScoreService:
    """Handles MongoDB operations for user matching scores."""

    def __init__(self, db: AsyncDatabase):
        self.db = db
        self.collection = db[settings.USER_RECOMMENDATIONS_COLLECTION]

    async def store_match_scores(
        self, user_id: str, matches: List[Dict[str, Any]]
    ) -> bool:
        """
        Store match scores to MongoDB using bulk upsert.
        Stores symmetric records (both A→B and B→A).
        """
        method = self.store_match_scores.__name__
        try:
            if not matches:
                logger.warning(f"[{method}] No matches to store")
                return False
            start_time = time.perf_counter()
            operations = []
            now = datetime.now()
            user_obj_id = ObjectId(user_id)

            for match in matches:
                matched_user_obj_id = ObjectId(match["matched_user_id"])

                doc = {
                    "user_id": user_obj_id,
                    "matched_user_id": matched_user_obj_id,
                    "score": match["percentage_score"],
                    "similarity_breakdown": match["similarity_breakdown"],
                    "updated_at": now,
                }

                operations.append(
                    UpdateOne(
                        {
                            "user_id": user_obj_id,
                            "matched_user_id": matched_user_obj_id,
                        },
                        {"$set": doc, "$setOnInsert": {"created_at": now}},
                        upsert=True,
                    )
                )

            result = await self.collection.bulk_write(operations, ordered=False)

            elapsed = (time.perf_counter() - start_time) * 1000
            total_stored = result.upserted_count + result.modified_count

            logger.info(
                f"[{method}] Stored {total_stored} match scores in {elapsed:.2f}ms "
                f"({len(matches)/elapsed*1000:.0f} scores/sec)"
            )
            return True

        except Exception as e:
            logger.error(f"[{method}] Error storing match scores: {e}")
            raise e

    async def get_user_matches(
        self, user_id: str, limit: int = 100, offset: int = 0
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve stored match scores for a user with pagination.
        """
        method = self.get_user_matches.__name__
        try:
            start_time = time.perf_counter()
            user_obj_id = ObjectId(user_id)

            matches = (
                await self.collection.find(
                    {"user_id": user_obj_id},
                )
                .sort("score", DESCENDING)
                .skip(offset)
                .limit(limit)
                .to_list(length=None)
            )

            if not matches:
                logger.info(f"[{method}] No matches found for user {user_id[:8]}")
                return None

            for match in matches:
                match["user_id"] = str(match["user_id"])
                match["matched_user_id"] = str(match["matched_user_id"])
                if "_id" in match:
                    match["_id"] = str(match["_id"])

            elapsed = (time.perf_counter() - start_time) * 1000
            logger.info(
                f"[{method}] Retrieved {len(matches)} matches for user {user_id[:8]} "
                f"in {elapsed:.2f}ms"
            )

            return matches

        except Exception as e:
            logger.error(f"[{method}] Error retrieving matches: {e}")
            return None


async def test():
    await connect_to_mongo()
    db = get_database()

    match_service = UserMatchingService(db)
    filter_service = UserFilterService(db)
    score_service = MatchScoreService(db)

    # Get test user
    user_ids = await filter_service.get_all_active_user_ids()
    test_user_id = user_ids[0]

    print(f"Test user: {test_user_id}")

    # Get compatible candidates
    compatible = await filter_service.filter_compatible_candidates(test_user_id)
    print(f"Compatible candidates: {len(compatible)}")

    # Calculate matches
    matches = await match_service.calculate_matches_vectorized(test_user_id, compatible)
    print(f"Calculated {len(matches)} matches")

    # Store to database
    print("\nStoring top 100 matches to MongoDB...")
    success = await score_service.store_match_scores(test_user_id, matches[:100])

    if success:
        print(" Storage successful")
    else:
        print("Storage failed")

    print("\nRetrieving stored matches...")
    stored_matches = await score_service.get_user_matches(test_user_id, limit=5)

    if stored_matches:
        print("\nTop 5 stored matches:")
        for i, match in enumerate(stored_matches, 1):
            print(f"{i}. {match['matched_user_id'][:12]} - {match['score']}%")
    else:
        print("No matches found")


if __name__ == "__main__":
    import asyncio

    from src.database import connect_to_mongo, get_database
    from src.recommendations.user_filter_service import UserFilterService
    from src.recommendations.user_matching_service import UserMatchingService

    asyncio.run(test())
