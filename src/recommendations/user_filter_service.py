"""Service for filtering compatible user candidates."""

import asyncio
import time
from typing import Dict, List, Optional

from bson import ObjectId
from pymongo.asynchronous.database import AsyncDatabase

from src.config import settings
from src.recommendations.pipelines import candidate_match_pipeline
from src.recommendations.utils import normalize_value
from src.utils.setup_logger import setup_logger

logger = setup_logger(__name__, "logs/user_filter_service.log")


class UserFilterService:
    """Handles filtering of compatible user candidates before embedding calculations."""

    def __init__(self, db: AsyncDatabase):
        self.db = db

    async def get_all_active_user_ids(self) -> List[str]:
        """
        Get all active user IDs from the system

        Returns a list of user_id strings
        """
        method = self.get_all_active_user_ids.__name__
        try:
            logger.info(f"[{method}] [START] Fetching all active user ids")

            cursor = self.db[settings.LOGIN_COLLECTION].find(
                {"is_deleted": False}, {"_id": 1}
            )

            users = await cursor.to_list(length=None)
            user_ids = [str(user["_id"]) for user in users]

            logger.info(f"[{method}] [SUCCESS] Found {len(user_ids)} active users")

            return user_ids

        except Exception as e:
            logger.error(f"[{method}] [Failed] Error fetching active user ids: {e}")
            raise e

    async def get_user_filtering_criteria(self, user_id: str) -> Optional[Dict]:
        """
        Get filtering criteria for a user.
        Returns dict with sectors, profile_type, and looking_to_connect.
        """
        method = self.get_user_filtering_criteria.__name__
        try:
            start_time = time.perf_counter()

            user_obj_id = ObjectId(user_id)

            logger.info(f"[{method}] [START] Fetching filter criterion for {user_id}")
            t1 = time.perf_counter()
            context_task = self.db[settings.CONTEXT_BUILDER_COLLECTION].find_one(
                {"user_id": user_obj_id}, {"sector": 1, "looking_to_connect": 1}
            )
            org_task = self.db[settings.ORGANISATION_PROFILE_COLLECTION].find_one(
                {"user_id": user_obj_id}, {"profile_type": 1, "sector": 1}
            )

            context, org = await asyncio.gather(context_task, org_task)
            fetch_time = (time.perf_counter() - t1) * 1000

            if not context or not org:
                logger.warning(
                    f"[{method}] [SKIP] User {user_id} has incomplete profile"
                )
                return None

            sectors = []
            if context and context.get("sector"):
                sectors = [
                    item.get("value") or item.get("label", "")
                    for item in context["sector"]
                    if isinstance(item, dict)
                ]

            looking_to_connect = []
            if context and context.get("looking_to_connect"):
                looking_to_connect = [
                    item.get("value") or item.get("label", "")
                    for item in context["looking_to_connect"]
                    if isinstance(item, dict)
                ]

            profile_type_raw = org.get("profile_type")
            profile_type = normalize_value(profile_type_raw)

            # TODO: Might add this users sector
            # user_sector_raw = org.get("sector")

            if not profile_type or not sectors:
                logger.warning(
                    f"[{method}] [SKIP] User {user_id} missing required fields: "
                    f"type={profile_type}, sectors={len(sectors)}"
                )
                return None

            elapsed = (time.perf_counter() - start_time) * 1000

            logger.info(
                f"[{method}] [SUCCESS] User {user_id} criteria in {elapsed:.2f}ms "
                f"(fetch: {fetch_time:.2f}ms) - "
                f"sectors={len(sectors)}, type={profile_type}, wants={len(looking_to_connect)}"
            )

            return {
                "sectors": sectors,
                "profile_type": profile_type,
                "looking_to_connect": looking_to_connect,
            }

        except Exception as e:
            logger.error(
                f"[{method}] [FAILED] Error getting filter criteria for {user_id}: {e}"
            )

    async def filter_compatible_candidates(self, user_id: str) -> List[str]:
        """
        Filter all users to only compatible candidates.
        Returns list of user IDs that match filtering criteria.
        """
        method = self.filter_compatible_candidates.__name__
        try:
            start_time = time.perf_counter()

            user_criteria = await self.get_user_filtering_criteria(user_id)
            if not user_criteria:
                logger.warning(
                    f"[{method}] [SKIP] User {user_id} has incomplete profile"
                )
                return []

            user_sectors = user_criteria["sectors"]
            user_type = user_criteria["profile_type"]
            user_wants = user_criteria["looking_to_connect"]

            logger.info(f"[{method}] [***] Finding compatible candidates for {user_id}")

            t1 = time.perf_counter()
            pipeline = candidate_match_pipeline(
                user_id=user_id, user_sectors=user_sectors
            )

            cursor = await self.db[settings.CONTEXT_BUILDER_COLLECTION].aggregate(
                pipeline
            )
            candidates = await cursor.to_list(length=None)

            query_time = (time.perf_counter() - t1) * 1000
            logger.info(
                f"[{method}] [TIMING] Aggregation pipeline: {query_time:.2f}ms, "
                f"found {len(candidates)} candidates"
            )

            compatible_ids = []

            for candidate in candidates:
                candidate_type = normalize_value(candidate.get("profile_type"))

                if not candidate_type:
                    continue

                candidate_wants = []
                if candidate.get("looking_to_connect"):
                    for item in candidate["looking_to_connect"]:
                        if isinstance(item, dict):
                            value = item.get("value") or item.get("label", "")
                        else:
                            value = item
                        normalized = normalize_value(value)
                        if normalized:
                            candidate_wants.append(normalized)

                is_complementary = (
                    candidate_type in user_wants or user_type in candidate_wants
                )

                if is_complementary:
                    compatible_ids.append(str(candidate["user_id"]))

            elapsed = (time.perf_counter() - start_time) * 1000

            logger.info(
                f"[{method}] [SUCCESS] Found {len(compatible_ids)} compatible candidates in {elapsed:.2f}ms"
            )

            return compatible_ids

        except Exception as e:
            logger.error(
                f"[{method}] [FAILED] Error filtering candidates for {user_id}: {e}"
            )
            raise e


async def test():

    await connect_to_mongo()
    db = get_database()
    service = UserFilterService(db)

    print("+" * 60)
    print("TEST 1: Get all active user IDs")
    print("+" * 60)
    user_ids = await service.get_all_active_user_ids()
    print(f"Total users: {len(user_ids)}")
    print(f"First 5: {user_ids[:5]}")

    print("+" * 60)
    print("TEST 2: Get filter criteria (test first 10 users)")
    print("+" * 60)

    complete_count = 0
    incomplete_count = 0

    for user_id in user_ids:
        criteria = await service.get_user_filtering_criteria(user_id)

        if criteria:
            complete_count += 1
            # print(f"{user_id}: {criteria}")
        else:
            incomplete_count += 1
            print(f"{user_id}: Skipped (incomplete)")

    print(f"\nResults: {complete_count} complete, {incomplete_count} incomplete\n")

    print("+" * 60)
    print("Test 3: Filtering compatible IDs")
    print("+" * 60)

    test_user_id = None
    for uid in user_ids:
        criteria = await service.get_user_filtering_criteria(uid)
        if criteria:
            test_user_id = uid
            print(f"\nTest user {uid}:")
            print(f"  Criteria: {criteria}")
            break

    if test_user_id:
        compatible = await service.filter_compatible_candidates(test_user_id)

        print("\nFiltering results:")
        print(f"  Total candidates: {len(user_ids)}")
        print(f"  Compatible: {len(compatible)}")
        print(f"  Reduction: {len(user_ids) - len(compatible)} users filtered out")


if __name__ == "__main__":
    import asyncio

    from src.database import connect_to_mongo, get_database

    asyncio.run(test())
