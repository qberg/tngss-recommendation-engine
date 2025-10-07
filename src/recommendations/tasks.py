"""Celery tasks for background recommendation computation."""

import asyncio
from datetime import datetime

from pymongo import AsyncMongoClient

from src.config import settings
from src.recommendations.celery_config import celery_app
from src.recommendations.service import RecommendationService
from src.utils.setup_logger import setup_logger

logger = setup_logger(__name__, "logs/celery_tasks.log")


@celery_app.task(bind=True, max_retries=3, default_retry_delay=10)
def calculate_user_recommendations(
    self,
    user_id: str,
    force_recalculate: bool = False,
    force_user_regenerate: bool = False,
):
    """Calculate recommendations - runs async code in sync Celery context."""
    try:
        # Run the async logic
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            _async_calculate_recommendations(
                self, user_id, force_recalculate, force_user_regenerate
            )
        )
        loop.close()
        return result
    except Exception as e:
        logger.error(f"[CELERY] Error: {e}", exc_info=True)
        raise self.retry(exc=e, countdown=10)


async def _async_calculate_recommendations(
    task_instance, user_id: str, force_recalculate: bool, force_user_regenerate: bool
):
    """The actual async logic."""
    logger.info(f"[CELERY] Starting calculation for user {user_id[:8]}")
    start_time = datetime.now()

    task_instance.update_state(
        state="PROGRESS",
        meta={"current": 10, "total": 100, "status": "Initializing..."},
    )

    client = AsyncMongoClient(settings.MONGODB_URL, maxPoolSize=10)
    db = client[settings.DATABASE_NAME]

    try:
        service = RecommendationService(db)

        task_instance.update_state(
            state="PROGRESS",
            meta={"current": 30, "total": 100, "status": "Generating embeddings..."},
        )

        if force_recalculate:
            logger.info(f"[CELERY] Full regeneration for {user_id[:8]}")
            scores = await service.generate_event_scores_for_user_with_cache(
                user_id, max_events=-1
            )
        else:
            logger.info(f"[CELERY] Fast mode for {user_id[:8]}")
            scores = await service.generate_event_scores_for_user_fast(
                user_id, max_events=-1, force_user_regenerate=force_user_regenerate
            )

        if not scores:
            logger.warning(f"[CELERY] No scores for {user_id[:8]}")
            return {"success": False, "user_id": user_id, "error": "No scores"}

        task_instance.update_state(
            state="PROGRESS",
            meta={"current": 90, "total": 100, "status": "Storing results..."},
        )

        success = await service.store_scores_to_database(scores)
        elapsed = (datetime.now() - start_time).total_seconds()

        if success:
            logger.info(
                f"[CELERY] Completed {user_id[:8]}: {len(scores)} scores in {elapsed:.2f}s"
            )
            return {
                "success": True,
                "user_id": user_id,
                "scores_count": len(scores),
                "elapsed_seconds": elapsed,
            }
        else:
            logger.error(f"[CELERY] Store failed for {user_id[:8]}")
            return {"success": False, "user_id": user_id, "error": "Store failed"}

    finally:
        await client.close()


@celery_app.task(bind=True, max_retries=3, default_retry_delay=10)
def calculate_user_matches(
    self, user_id: str, apply_filters: bool = True, max_matches_to_store: int = 1000
):
    """
    Calculate user-to-user matches in background.

    Args:
        user_id: User to calculate matches for
        apply_filters: If True, filter for compatible candidates. If False, match against all users.
        max_matches_to_store: Maximum number of top matches to store in MongoDB
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            _async_calculate_user_matches(
                self, user_id, apply_filters, max_matches_to_store
            )
        )
        loop.close()
        return result
    except Exception as e:
        logger.error(f"[CELERY] Error calculating user matches: {e}", exc_info=True)
        raise self.retry(exc=e, countdown=10)


async def _async_calculate_user_matches(
    task_instance, user_id: str, apply_filters: bool, max_matches_to_store: int
):
    """Async logic for user match calculation."""
    logger.info(
        f"[CELERY] Starting user match calculation for {user_id[:8]} "
        f"(filters={apply_filters}, max_store={max_matches_to_store})"
    )
    start_time = datetime.now()

    task_instance.update_state(
        state="PROGRESS",
        meta={"current": 10, "total": 100, "status": "Initializing..."},
    )

    client = AsyncMongoClient(settings.MONGODB_URL, maxPoolSize=10)
    db = client[settings.DATABASE_NAME]

    try:
        from src.recommendations.match_score_service import MatchScoreService
        from src.recommendations.user_filter_service import UserFilterService
        from src.recommendations.user_matching_service import \
            UserMatchingService

        filter_service = UserFilterService(db)
        match_service = UserMatchingService(db)
        score_service = MatchScoreService(db)

        # Get candidates (filtered or all)
        task_instance.update_state(
            state="PROGRESS",
            meta={"current": 30, "total": 100, "status": "Finding candidates..."},
        )

        if apply_filters:
            candidates = await filter_service.filter_compatible_candidates(user_id)
            logger.info(f"[CELERY] Filtered to {len(candidates)} compatible candidates")
        else:
            all_users = await filter_service.get_all_active_user_ids()
            candidates = [uid for uid in all_users if uid != user_id]
            logger.info(f"[CELERY] Matching against all {len(candidates)} users")

        if not candidates:
            logger.warning(f"[CELERY] No candidates for {user_id}")
            return {"success": False, "user_id": user_id, "error": "No candidates"}

        # Calculate matches
        task_instance.update_state(
            state="PROGRESS",
            meta={"current": 60, "total": 100, "status": "Calculating matches..."},
        )

        matches = await match_service.calculate_matches_vectorized(user_id, candidates)

        if not matches:
            logger.warning(f"[CELERY] No matches calculated for {user_id[:8]}")
            return {"success": False, "user_id": user_id, "error": "No matches"}

        # Store top N matches
        task_instance.update_state(
            state="PROGRESS",
            meta={"current": 90, "total": 100, "status": "Storing results..."},
        )

        matches_to_store = matches[:max_matches_to_store]
        success = await score_service.store_match_scores(user_id, matches_to_store)

        elapsed = (datetime.now() - start_time).total_seconds()

        if success:
            logger.info(
                f"[CELERY] Completed user matches for {user_id}: "
                f"{len(matches)} calculated, {len(matches_to_store)} stored in {elapsed:.2f}s"
            )
            return {
                "success": True,
                "user_id": user_id,
                "matches_calculated": len(matches),
                "matches_stored": len(matches_to_store),
                "elapsed_seconds": elapsed,
            }
        else:
            return {"success": False, "user_id": user_id, "error": "Storage failed"}

    finally:
        await client.close()
