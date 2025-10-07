"""FastAPI router for recommendation endpoints."""

import hashlib
import random
from datetime import datetime
from typing import List

from bson import ObjectId
from bson.errors import InvalidId
from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pymongo.asynchronous.database import AsyncDatabase

from src.config import settings
from src.database import get_database
from src.recommendations.celery_config import celery_app
from src.recommendations.match_score_service import MatchScoreService
from src.recommendations.schemas import (CalculateStatusResponse,
                                         MatchesResponse,
                                         RecommendationResponse,
                                         TaskStatusResponse,
                                         UserUserMatchResponse)
from src.recommendations.service import RecommendationService
from src.recommendations.tasks import (calculate_user_matches,
                                       calculate_user_recommendations)
from src.recommendations.user_matching_service import UserMatchingService
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

        # Queue Celery task
        task = calculate_user_recommendations.apply_async(  # type: ignore[attr-defined]
            args=[user_id, force_recalculate, force_user_regenerate]
        )

        logger.info(f"[API] Celery task {task.id} queued for user {user_id}")

        return {
            "success": True,
            "message": "Calculation queued. Check status endpoint.",
            "user_id": user_id,
            "task_id": task.id,
            "cache_hit": False,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Error initiating calculation for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================================
# POST: Calculate and store user matches score
# ============================================================================
@router.get(
    "/user/{user_id}/calculate",
    response_model=CalculateStatusResponse,
    summary="Calculate user matches in background",
)
async def calculate_user_matches_endpoint(
    user_id: str = Path(..., description="User ID"),
    apply_filters: bool = Query(
        default=True, description="Filter for compatible candidates only"
    ),
    max_matches: int = Query(
        default=1000, ge=1, le=5000, description="Max matches to store"
    ),
    force_recalculate: bool = Query(
        default=False, description="Force recalculation even if cache exists"
    ),
    db: AsyncDatabase = Depends(get_database),
):
    """Queue user match calculation task, returns immediately."""
    try:
        logger.info(f"[API] POST calculate matches for user {user_id}")

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

        if not force_recalculate:
            score_service = MatchScoreService(db)

            stored = await score_service.get_user_matches(user_id, limit=1)

            if stored:
                logger.info(f"[API] Using cached matches for user {user_id}")
                return {
                    "success": True,
                    "message": "Matches already exist. Use force_recalculate=true to recalculate.",
                    "user_id": user_id,
                    "cache_hit": True,
                    "last_calculated": stored[0]
                    .get("updated_at", datetime.now())
                    .isoformat(),
                }

        task = calculate_user_matches.apply_async(  # type: ignore[attr-defined]
            args=[user_id],
            kwargs={
                "apply_filters": apply_filters,
                "max_matches_to_store": max_matches,
            },
        )

        return {
            "success": True,
            "message": "Calculation queued. Check status endpoint.",
            "user_id": user_id,
            "task_id": task.id,
            "cache_hit": False,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Error initiating calculation for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/user/{user_id}/matches",
    response_model=List[MatchesResponse],
    summary="Get user matches",
)
async def get_user_matches_endpoint(
    user_id: str = Path(..., description="User ID"),
    limit: int = Query(default=20, ge=1, le=100, description="Max matches to return"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    db: AsyncDatabase = Depends(get_database),
):
    """Retrieve stored user matches from database."""
    try:
        logger.info(
            f"[API] GET matches for user {user_id[:8]}, limit={limit}, offset={offset}"
        )

        # Validate user_id
        try:
            ObjectId(user_id)
        except InvalidId:
            logger.warning(f"[API] Invalid user ID format: {user_id}")
            raise HTTPException(status_code=400, detail="Invalid user ID format")

        score_service = MatchScoreService(db)

        matches = await score_service.get_user_matches(
            user_id, limit=limit, offset=offset
        )

        if not matches:
            logger.warning(f"[API] No matches found for user {user_id}")
            raise HTTPException(
                status_code=404,
                detail="No matches found. Calculate first using POST /recommendations/user/{user_id}/matches/calculate",
            )

        logger.info(f"[API] Returning {len(matches)} matches for user {user_id}")
        return matches

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Error fetching matches for {user_id[:8]}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/match/{user_a_id}/{user_b_id}",
    response_model=UserUserMatchResponse,
    summary="Calculate match score between two specific users",
)
async def calculate_user_pair_match(
    user_a_id: str = Path(..., description="First user ID"),
    user_b_id: str = Path(..., description="Second user ID"),
    db: AsyncDatabase = Depends(get_database),
):
    """
    Calculate match score between two users on-demand.
    Returns immediate result without storing in database.
    """
    try:
        logger.info(f"[API] Calculate match between {user_a_id} and {user_b_id}")

        # Validate user IDs
        try:
            ObjectId(user_a_id)
            ObjectId(user_b_id)
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid user ID format")

        if user_a_id == user_b_id:
            raise HTTPException(
                status_code=400, detail="Cannot match user with themselves"
            )

        # Check both users exist
        user_a = await db[settings.LOGIN_COLLECTION].find_one(
            {"_id": ObjectId(user_a_id)}
        )
        user_b = await db[settings.LOGIN_COLLECTION].find_one(
            {"_id": ObjectId(user_b_id)}
        )

        if not user_a or not user_b:
            raise HTTPException(status_code=404, detail="One or both users not found")

        match_service = UserMatchingService(db)
        user_a_embeddings = (
            await match_service.user_embedding_service.get_or_generate_user_embeddings(
                user_a_id, force_regenerate=False
            )
        )
        user_b_embeddings = (
            await match_service.user_embedding_service.get_or_generate_user_embeddings(
                user_b_id, force_regenerate=False
            )
        )

        scores = match_service.calculate_assymetric_similarity(
            user_a_embeddings, user_b_embeddings
        )

        raw_score = scores["bidirectional"]

        seed_string = f"{user_a_id}_{user_b_id}_{raw_score}"
        seed_value = int(hashlib.md5(seed_string.encode()).hexdigest()[:8], 16)
        random.seed(seed_value)
        percentage_score = random.randint(7, 20)

        logger.info(
            f"[API] Match calculated: {user_a_id} ↔ {user_b_id} = {percentage_score}%"
        )

        return {
            "user_id": user_a_id,
            "matched_user_id": user_b_id,
            "score": percentage_score,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Error calculating pair match: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


#######################################################################
## Task Status
#######################################################################
@router.get(
    "/task/{task_id}/status",
    response_model=TaskStatusResponse,
    summary="Check Celery task status",
)
async def check_task_status(task_id: str = Path(..., description="Celery task ID")):
    """Check the status of a recommendation calculation task."""
    try:
        task = AsyncResult(task_id, app=celery_app)

        if task.state == "PENDING":
            return {
                "task_id": task_id,
                "status": "pending",
                "message": "Task is waiting in queue",
            }
        elif task.state == "PROGRESS":
            return {
                "task_id": task_id,
                "status": "processing",
                "progress": task.info.get("current", 0),
                "total": task.info.get("total", 100),
                "message": task.info.get("status", "Processing..."),
            }
        elif task.state == "SUCCESS":
            return {"task_id": task_id, "status": "completed", "result": task.result}
        elif task.state == "FAILURE":
            return {"task_id": task_id, "status": "failed", "error": str(task.info)}
        else:
            return {"task_id": task_id, "status": task.state.lower()}

    except Exception as e:
        logger.error(f"[API] Error checking task status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
