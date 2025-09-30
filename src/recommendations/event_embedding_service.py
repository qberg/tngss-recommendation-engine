"""Service for event embedding operations."""

from typing import Any, Dict, List, Optional

import numpy as np
from pymongo.asynchronous.database import AsyncDatabase

from src.embeddings.service import EmbeddingService
from src.events.client import EventsClient
from src.events.service import EventService
from src.utils.setup_logger import setup_logger
from src.vector_store.service import VectorStoreService

logger = setup_logger(__name__, "logs/event_embedding_service.log")


class EventEmbeddingService:
    """Handles event-specific embedding operations."""

    def __init__(self, db: AsyncDatabase):
        self.db = db
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStoreService()
        self.events_client = EventsClient()

    async def fetch_all_events(self, batch_size: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch all public events from the events API.

        Args:
            batch_size: Number of events to fetch per request

        Returns:
            List of event dictionaries
        """
        try:
            logger.info(f"[***] Fetching all public events (batch_size={batch_size})")
            events_data = await self.events_client.get_all_public_events(
                batch_size=batch_size
            )

            if not events_data:
                logger.warning("[WARNING] No events found")
                return []

            logger.info(f"[SUCCESS] Fetched {len(events_data)} events")
            return events_data

        except Exception as e:
            logger.error(f"[FAILED] Error fetching events: {e}")
            raise e

    async def generate_event_embedding(
        self, event_data: Dict[str, Any], event_text: str
    ) -> np.ndarray:
        """
        Generate embedding for a single event.

        Args:
            event_data: Event data dictionary
            event_text: Formatted text for embedding

        Returns:
            Event embedding as numpy array
        """
        try:
            embeddings = await self.embedding_service.create_embeddings([event_text])
            return embeddings[0]

        except Exception as e:
            logger.error(f"[FAILED] Error generating event embedding: {e}")
            raise e

    async def get_or_generate_event_embeddings(
        self, events_data: List[Dict[str, Any]], force_regenerate: bool = False
    ) -> Dict[str, Optional[np.ndarray]]:
        """
        Get cached embeddings or generate new ones for events.

        Args:
            events_data: List of event dictionaries
            force_regenerate: Force regeneration even if valid cache exists

        Returns:
            Dict mapping event_id to embedding
        """
        try:
            if not events_data:
                logger.warning("[WARNING] No events provided")
                return {}

            events_texts = EventService.format_events_for_embedding(events_data)

            if not force_regenerate:
                stale_event_ids = self.vector_store.get_stale_event_ids(
                    events_data, events_texts
                )
            else:
                stale_event_ids = [e["id"] for e in events_data]
                logger.info("[INFO] Force regenerate enabled, regenerating all events")

            stale_embeddings_map = {}
            if stale_event_ids:
                logger.info(
                    f"[INFO] Regenerating {len(stale_event_ids)} stale/missing event embeddings"
                )

                stale_events = [e for e in events_data if e["id"] in stale_event_ids]
                stale_texts = [
                    events_texts[i]
                    for i, e in enumerate(events_data)
                    if e["id"] in stale_event_ids
                ]

                stale_embeddings = await self.embedding_service.create_embeddings(
                    stale_texts
                )

                self.vector_store.store_event_embeddings_batch(
                    events_data=stale_events,
                    embeddings=stale_embeddings,
                    events_texts=stale_texts,
                )

                stale_embeddings_map = {
                    event["id"]: embedding
                    for event, embedding in zip(stale_events, stale_embeddings)
                }
            else:
                logger.info("[INFO] All event embeddings are cached and fresh")

            event_ids = [e["id"] for e in events_data]
            cached_embeddings = self.vector_store.get_all_event_embeddings(event_ids)

            for event_id, embedding in stale_embeddings_map.items():
                cached_embeddings[event_id] = embedding

            missing_count = sum(1 for emb in cached_embeddings.values() if emb is None)
            if missing_count > 0:
                logger.warning(
                    f"[WARNING] {missing_count} events have missing embeddings"
                )

            logger.info(
                f"[SUCCESS] Retrieved embeddings for {len(cached_embeddings)} events "
                f"({len(stale_event_ids)} regenerated, "
                f"{len(event_ids) - len(stale_event_ids)} cached)"
            )

            return cached_embeddings

        except Exception as e:
            logger.error(f"[FAILED] Error getting/generating event embeddings: {e}")
            raise e

    async def close(self):
        """Close events client connection."""
        await self.events_client.close()
