"""
Core recommendation service - now a lightweight orchestrator.
"""

from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional

import numpy as np
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
            recommendation_scores = recommendation_scores[
                : max_events if max_events > 0 else len(recommendation_scores)
            ]

            # Normalize to percentage
            recommendation_scores = self.score_service.normalize_scores_to_percentage(
                recommendation_scores
            )

            await self.event_service.close()

            logger.info(
                f"[SUCCESS] Generated {len(recommendation_scores)} event scores."
            )

            return recommendation_scores

        except Exception as e:
            logger.error(f"[FAILED] Event recommendation generation failed: {e}")
            raise e

    async def generate_event_scores_for_user_fast(
        self,
        user_id: str,
        max_events: Optional[int] = None,
        force_user_regenerate: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Ultra-fast score generation - assumes event embeddings are pre-cached.
        Skips fetching events from API if embeddings exist.
        """
        max_events = max_events or -1
        start_time = perf_counter()

        try:
            logger.info(
                f"[***][RecommendationService] Fast score generation for user: {user_id}..."
            )

            t1 = perf_counter()

            # Get user embeddings (from cache or generate)
            user_embeddings = await self.user_service.get_or_generate_user_embeddings(
                user_id, force_regenerate=force_user_regenerate
            )

            user_emb_time = (perf_counter() - t1) * 1000
            logger.info(f"[TIMING] User embeddings: {user_emb_time:.2f}ms")

            # Check if we have cached event embeddings
            events_dir = Path(self.event_service.vector_store.events_embeddings_dir)
            cached_event_files = list(events_dir.glob("*.pkl"))

            if not cached_event_files:
                logger.warning(
                    "[WARNING] No cached event embeddings, falling back to full generation"
                )
                return await self.generate_event_scores_for_user_with_cache(
                    user_id, max_events
                )

            # Load all cached event embeddings without fetching from API
            t2 = perf_counter()
            event_ids = [f.stem for f in cached_event_files]
            event_embeddings = self.event_service.vector_store.get_all_event_embeddings(
                event_ids
            )

            # Filter out None values and prepare for vectorization
            valid_event_ids = []
            valid_embeddings_list = []

            for event_id, embedding in event_embeddings.items():
                if embedding is not None:
                    valid_event_ids.append(event_id)
                    valid_embeddings_list.append(embedding)

            if not valid_embeddings_list:
                logger.warning("[WARNING] No valid event embeddings found")
                return []

            load_time = (perf_counter() - t2) * 1000
            logger.info(
                f"[TIMING] Loaded {len(valid_embeddings_list)} embeddings: {load_time:.2f}ms"
            )

            # ========== VECTORIZED SIMILARITY CALCULATION ==========
            t3 = perf_counter()

            # Stack all event embeddings into matrix
            event_matrix = np.vstack(valid_embeddings_list)
            # Shape: (num_events, embedding_dim)

            # Stack user embeddings
            user_matrix = np.array(
                [
                    user_embeddings["personal"],
                    user_embeddings["org"],
                    user_embeddings["intent"],
                ]
            )
            # Shape: (3, embedding_dim)

            # Matrix multiplication: user_matrix @ event_matrix.T
            # Result shape: (3, num_events)
            similarity_matrix = user_matrix @ event_matrix.T

            # Apply weights
            weights = np.array([0.25, 0.25, 0.5])
            final_scores = weights @ similarity_matrix

            calc_time = (perf_counter() - t3) * 1000
            logger.info(
                f"[TIMING] Vectorized calculation: {calc_time:.2f}ms for {len(valid_event_ids)} events"
            )

            # Build recommendation scores
            t4 = perf_counter()
            recommendation_scores = []
            for idx, event_id in enumerate(valid_event_ids):
                recommendation_scores.append(
                    {
                        "user_id": user_id,
                        "target_id": event_id,
                        "similarity_score": float(final_scores[idx]),
                        "similarity_breakdown": {
                            "personal": float(similarity_matrix[0, idx]),
                            "org": float(similarity_matrix[1, idx]),
                            "intent": float(similarity_matrix[2, idx]),
                        },
                    }
                )

            build_time = (perf_counter() - t4) * 1000
            logger.info(f"[TIMING] Build scores list: {build_time:.2f}ms")

            # Limit results
            if max_events > 0:
                recommendation_scores = recommendation_scores[:max_events]

            # Normalize to percentage
            recommendation_scores = self.score_service.normalize_scores_to_percentage(
                recommendation_scores
            )

            elapsed_time = (perf_counter() - start_time) * 1000
            logger.info(
                f"[SUCCESS] Fast generation: {len(recommendation_scores)} scores in {elapsed_time:.2f}ms"
            )

            return recommendation_scores

        except Exception as e:
            logger.error(f"[FAILED] Fast score generation failed: {e}")
            # Fallback to normal method
            return await self.generate_event_scores_for_user_with_cache(
                user_id, max_events
            )

    # Delegate database storage methods to ScoreService
    async def get_stored_scores(
        self, user_id: str, max_age_hours: int = 24, limit: int = 100
    ):
        return await self.score_service.get_stored_scores(
            user_id, "event", max_age_hours, limit
        )

    async def store_scores_to_database(self, scores: List[Dict[str, Any]]) -> bool:
        return await self.score_service.store_scores_to_database(scores)
