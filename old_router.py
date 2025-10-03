"""FastAPI router for recommendation endpoints."""

from datetime import datetime
from typing import List

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import (APIRouter, BackgroundTasks, Depends, HTTPException, Path,
                     Query)
from pymongo.asynchronous.database import AsyncDatabase

from src.config import settings
from src.database import get_database
from src.recommendations.schemas import (CalculateStatusResponse,
                                         RecommendationResponse)
from src.recommendations.service import RecommendationService
from src.recommendations.tasks import calculate_user_recommendations
from src.utils.setup_logger import setup_logger

logger = setup_logger(__name__, "logs/recommendations_router.log")

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


# ============================================================================
# GET: Retrieve stored recommendations
# ============================================================================
@router.get(
    "/user/{user_id}/events",
    response_model=List[RecommendationResponse],
    summary="Get event recommendations",
)
async def get_user_event_recommendations(
    user_id: str = Path(..., description="User ID"),
    max_events: int = Query(
        default=10, ge=1, le=100, description="Max recommendations"
    ),
    db: AsyncDatabase = Depends(get_database),
):
    """Retrieve stored event recommendations from database."""
    try:
        logger.info(f"[API] GET recommendations for user {user_id}, limit={max_events}")

        # Validate user_id
        try:
            ObjectId(user_id)
        except InvalidId:
            logger.warning(f"[API] Invalid user ID format: {user_id}")
            raise HTTPException(status_code=400, detail="Invalid user ID format")

        service = RecommendationService(db)
        scores = await service.get_stored_scores(user_id, limit=max_events)

        if not scores:
            logger.warning(f"[API] No recommendations found for user {user_id[:8]}")
            raise HTTPException(
                status_code=404,
                detail="No recommendations found. Calculate first using POST /recommendations/user/{user_id}/events/calculate",
            )

        logger.info(
            f"[API] Returning {len(scores)} recommendations for user {user_id[:8]}"
        )
        return scores

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Error fetching recommendations for {user_id[:8]}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================================
# POST: Calculate and store recommendations (Background Task)
# ============================================================================
@router.post(
    "/user/{user_id}/events/calculate",
    response_model=CalculateStatusResponse,
    summary="Calculate event recommendations in background",
)
async def calculate_user_event_recommendations(
    background_tasks: BackgroundTasks,
    user_id: str = Path(..., description="User ID"),
    force_recalculate: bool = Query(
        default=False, description="Force recalculation of all events and user scores"
    ),
    force_user_regenerate: bool = Query(
        default=False, description="Force regenration of user embeddings"
    ),
    db: AsyncDatabase = Depends(get_database),
):
    """Start recommendation calculation in background, returns immediately."""
    try:
        logger.info(
            f"[API] POST calculate for user {user_id[:8]}, force={force_recalculate}"
        )

        # Validate user_id
        try:
            ObjectId(user_id)
        except InvalidId:
            logger.warning(f"[API] Invalid user ID format: {user_id}")
            raise HTTPException(status_code=400, detail="Invalid user ID format")

        # Check if user exists
        user = await db[settings.LOGIN_COLLECTION].find_one({"_id": ObjectId(user_id)})
        if not user:
            logger.warning(f"[API] User not found: {user_id}")
            raise HTTPException(status_code=404, detail="User not found")

        logger.info(f"Force recalculate: {force_recalculate}")
        service = RecommendationService(db)

        # Quick cache check
        if not force_recalculate:
            stored = await service.get_stored_scores(user_id, max_age_hours=24, limit=1)

            if stored:
                logger.info(f"[INFO] Found {len(stored)} scores for user")
                raw_data = await service.user_service.get_raw_user_data(user_id)
                user_data = await service.user_service.get_user_data(user_id, raw_data)
                current_texts = service.user_service.profile_service.create_all_texts(
                    user_data
                )
                should_regen, reason = (
                    service.user_service.vector_store.should_regenerate_user_embeddings(
                        user_id,
                        raw_data,
                        current_texts=current_texts,
                        skip_content_check=False,
                    )
                )

                logger.info(f"[INFO] Should Regen: {should_regen}, Reason: {reason}")

                if not should_regen:
                    logger.info(
                        f"[API] Using cached recommendations for user {user_id}"
                    )
                    return {
                        "success": True,
                        "message": "Recommendations already exist. Use force_recalculate=true to recalculate.",
                        "user_id": user_id,
                        "cache_hit": True,
                        "last_calculated": stored[0]
                        .get("updated_at", datetime.now())
                        .isoformat(),
                    }
                else:
                    # Profile changed, recalculate
                    logger.info(
                        f"[API] Profile changed for user {user_id}: {reason}, recalculating"
                    )
                    force_user_regenerate = True

        # Add to background tasks
        background_tasks.add_task(
            _calculate_and_store_background,
            user_id,
            db,
            force_recalculate,
            force_user_regenerate,
        )

        # Queue Celery task
        task = calculate_user_recommendations.apply_async(
            args=[user_id, force_recalculate, force_user_regenerate]
        )

        logger.info(f"[API] Background calculation started for user {user_id}")
        return {
            "success": True,
            "message": "Calculation started in background. Check GET endpoint in a few seconds.",
            "user_id": user_id,
            "cache_hit": False,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Error initiating calculation for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================================
# Background worker function
# ============================================================================
async def _calculate_and_store_background(
    user_id: str,
    db: AsyncDatabase,
    force_recalculate: bool = False,
    force_user_regenerate: bool = False,
):
    """Execute recommendation calculation in background."""
    try:
        logger.info(f"[BG] Starting calculation for user {user_id}")
        start_time = datetime.now()

        service = RecommendationService(db)
        if force_recalculate:
            logger.info(f"[BG] Force mode: regenerating all embeddings for {user_id}")
            scores = await service.generate_event_scores_for_user_with_cache(
                user_id, max_events=-1
            )
        else:
            if force_user_regenerate:
                logger.info(f"[BG] Fast mode: generating new emneddings for {user_id}")
            else:
                logger.info(f"[BG] Fast mode: using cached embeddings for {user_id}")
            scores = await service.generate_event_scores_for_user_fast(
                user_id, max_events=-1, force_user_regenerate=force_user_regenerate
            )

        if not scores:
            logger.warning(f"[BG] No scores generated for user {user_id}")
            return

        # Store in database
        success = await service.store_scores_to_database(scores)

        if success:
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"[BG] Completed for user {user_id}: "
                f"{len(scores)} scores stored in {elapsed:.2f}s"
            )
        else:
            logger.error(f"[BG] Failed to store scores for user {user_id[:8]}")

    except Exception as e:
        logger.error(
            f"[BG] Background calculation failed for {user_id[:8]}: {e}", exc_info=True
        )
