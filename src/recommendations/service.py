"""
Core recommendation service - now a lightweight orchestrator.
"""

from typing import Any, Dict, List, Optional

from pymongo.asynchronous.database import AsyncDatabase

from src.recommendations.batch_service import BatchService
from src.recommendations.event_embedding_service import EventEmbeddingService
from src.recommendations.score_service import ScoreService
from src.recommendations.user_embedding_service import UserEmbeddingService
from src.utils.setup_logger import setup_logger

logger = setup_logger(__name__, "logs/recommendation_service.log")


class RecommendationService:
    """Orchestrates recommendation generation by delegating to specialized services."""

    def __init__(self, database: AsyncDatabase):
        self.db = database

        self.user_service = UserEmbeddingService(database)
        self.event_service = EventEmbeddingService(database)
        self.score_service = ScoreService(database)
        self.batch_service = BatchService(database)

        logger.info("[SUCCESS] Recommendation service initialized")

    async def generate_event_scores_for_user_with_cache(
        self, user_id: str, max_events: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate event recommendation scores with caching.
        Orchestrates: user embeddings → event embeddings → score calculation → normalization.
        """
        max_events = max_events or -1

        try:
            logger.info(f"[***] Generating event scores for user: {user_id[:8]}...")

            # Get user embeddings (cached or generate)
            user_embeddings = await self.user_service.get_or_generate_user_embeddings(
                user_id
            )

            # Get all events
            events_data = await self.event_service.fetch_all_events(batch_size=10)

            if not events_data:
                logger.warning("[WARNING] No events found")
                return []

            # Get event embeddings (cached or generate)
            event_embeddings = (
                await self.event_service.get_or_generate_event_embeddings(events_data)
            )

            # Calculate similarity scores
            recommendation_scores = []
            missing_count = 0

            for event_data in events_data:
                event_id = event_data["id"]
                event_embedding = event_embeddings.get(event_id)

                if event_embedding is None:
                    missing_count += 1
                    continue

                similarity_result = self.user_service.embedding_service.calculate_multi_vector_similarity(
                    user_embeddings, event_embedding
                )

                from src.events.schemas import Event

                event = Event.from_api_response(event_data)

                recommendation_scores.append(
                    {
                        "user_id": user_id,
                        "target_id": event.id,
                        "similarity_score": similarity_result.final_score,
                        "similarity_breakdown": {
                            "personal": similarity_result.personal,
                            "org": similarity_result.org,
                            "intent": similarity_result.intent,
                        },
                    }
                )

            if missing_count > 0:
                logger.warning(
                    f"[WARNING] Skipped {missing_count} events due to missing embeddings"
                )

            # Sort and limit
            recommendation_scores.sort(
                key=lambda x: x["similarity_score"], reverse=True
            )
            recommendation_scores = recommendation_scores[
                : max_events if max_events > 0 else len(recommendation_scores)
            ]

            # Normalize to percentage
            recommendation_scores = self.score_service.normalize_scores_to_percentage(
                recommendation_scores
            )

            await self.event_service.close()

            logger.info(
                f"[SUCCESS] Generated {len(recommendation_scores)} event scores"
            )

            return recommendation_scores

        except Exception as e:
            logger.error(f"[FAILED] Event recommendation generation failed: {e}")
            raise e

    # Delegate database storage methods to ScoreService
    async def get_stored_scores(self, user_id: str, max_age_hours: int = 24):
        return await self.score_service.get_stored_scores(
            user_id, "event", max_age_hours
        )

    async def store_scores_to_database(self, scores: List[Dict[str, Any]]) -> bool:
        return await self.score_service.store_scores_to_database(scores)
