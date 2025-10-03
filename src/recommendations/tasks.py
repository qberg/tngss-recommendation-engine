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
