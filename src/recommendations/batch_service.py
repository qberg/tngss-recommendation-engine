"""Service for batch operations on users and events."""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from pymongo.asynchronous.database import AsyncDatabase

from src.config import settings
from src.recommendations.event_embedding_service import EventEmbeddingService
from src.recommendations.user_embedding_service import UserEmbeddingService
from src.utils.setup_logger import setup_logger

logger = setup_logger(__name__, "logs/batch_service.log")


class BatchService:
    """Handles batch operations for all users/events."""

    def __init__(self, db: AsyncDatabase):
        self.db = db
        self.user_service = UserEmbeddingService(db)
        self.event_service = EventEmbeddingService(db)

    async def generate_all_user_embeddings(
        self,
        batch_size: int = 50,
        skip_existing: bool = True,
        force_regenerate: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate embeddings for all users in the database.

        Args:
            batch_size: Number of users to process at once
            skip_existing: Skip users with valid cached embeddings
            force_regenerate: Force regeneration even if cache is valid

        Returns:
            Statistics about the operation
        """
        start_time = time.time()

        try:
            logger.info("[***] Starting batch generation of all user embeddings")

            stats: Dict[str, Any] = {
                "total_users": 0,
                "processed": 0,
                "generated": 0,
                "cached": 0,
                "failed": 0,
                "skipped": 0,
            }

            # Get total count
            stats["total_users"] = await self.db[
                settings.LOGIN_COLLECTION
            ].count_documents({})
            logger.info(f"[INFO] Found {stats['total_users']} total users")

            if stats["total_users"] == 0:
                logger.warning("[WARNING] No users found in database")
                return stats

            # Process using cursor (memory efficient)
            cursor = (
                self.db[settings.LOGIN_COLLECTION]
                .find({}, {"_id": 1})
                .batch_size(batch_size)
            )

            batch_num = 0
            current_batch = []

            async for user in cursor:
                current_batch.append(user)

                if len(current_batch) >= batch_size:
                    batch_num += 1
                    await self._process_user_embedding_batch(
                        current_batch, stats, skip_existing, force_regenerate
                    )

                    logger.info(
                        f"[INFO] Batch {batch_num}: {stats['processed']}/{stats['total_users']} users | "
                        f"Generated: {stats['generated']}, Cached: {stats['cached']}, Failed: {stats['failed']}"
                    )

                    current_batch = []

            # Process remaining users
            if current_batch:
                batch_num += 1
                await self._process_user_embedding_batch(
                    current_batch, stats, skip_existing, force_regenerate
                )

            # Final statistics
            elapsed = time.time() - start_time
            stats["duration_seconds"] = round(elapsed, 2)
            stats["users_per_second"] = (
                round(stats["processed"] / elapsed, 2) if elapsed > 0 else 0
            )

            logger.info(
                f"[SUCCESS] Completed user embedding generation:\n"
                f"  Total: {stats['total_users']}\n"
                f"  Processed: {stats['processed']}\n"
                f"  Generated: {stats['generated']}\n"
                f"  Cached: {stats['cached']}\n"
                f"  Skipped: {stats['skipped']}\n"
                f"  Failed: {stats['failed']}\n"
                f"  Duration: {stats['duration_seconds']}s\n"
                f"  Speed: {stats['users_per_second']} users/sec"
            )

            return stats

        except Exception as e:
            logger.error(f"[FAILED] Batch embedding generation failed: {e}")
            raise e

    async def _process_user_embedding_batch(
        self,
        users: List[Dict],
        stats: Dict[str, Any],
        skip_existing: bool,
        force_regenerate: bool,
    ) -> None:
        """Process a batch of users for embedding generation."""

        for user in users:
            user_id = str(user["_id"])
            stats["processed"] += 1

            try:
                # Check if should skip
                if skip_existing and not force_regenerate:
                    if self.user_service.vector_store.user_embeddings_exists(user_id):
                        try:
                            raw_data = await self.user_service.get_raw_user_data(
                                user_id
                            )
                            should_regen, reason = (
                                self.user_service.vector_store.should_regenerate_user_embeddings(
                                    user_id, raw_data, skip_content_check=True
                                )
                            )

                            if not should_regen:
                                stats["cached"] += 1
                                continue

                            logger.info(
                                f"[INFO] User {user_id[:8]} needs regeneration: {reason}"
                            )

                        except Exception as e:
                            logger.warning(
                                f"[WARNING] Cache check failed for {user_id[:8]}, regenerating: {e}"
                            )

                # Generate embeddings
                raw_data = await self.user_service.get_raw_user_data(user_id)

                # Skip users with no data
                if not any(raw_data.values()):
                    logger.warning(
                        f"[WARNING] User {user_id[:8]} has no data, skipping"
                    )
                    stats["skipped"] += 1
                    continue

                user_data = await self.user_service.get_user_data(user_id, raw_data)
                user_embeddings = await self.user_service.generate_user_embeddings(
                    user_data
                )
                texts = self.user_service.profile_service.create_all_texts(user_data)

                self.user_service.vector_store.store_user_embeddings(
                    user_id, user_embeddings, texts, raw_data
                )

                stats["generated"] += 1

            except Exception as e:
                logger.error(f"[FAILED] Error processing user {user_id[:8]}: {e}")
                stats["failed"] += 1
                continue

    async def generate_all_event_embeddings(
        self, batch_size: int = 10, force_regenerate: bool = False
    ) -> Dict[str, Any]:
        """
        Generate embeddings for all events.

        Args:
            batch_size: Number of events to fetch per request
            force_regenerate: Force regeneration even if cache is valid

        Returns:
            Statistics about the operation
        """
        start_time = time.time()

        try:
            logger.info("[***] Starting batch generation of all event embeddings")

            # Fetch all events
            events_data = await self.event_service.fetch_all_events(
                batch_size=batch_size
            )

            if not events_data:
                logger.warning("[WARNING] No events found")
                return {
                    "total_events": 0,
                    "generated": 0,
                    "cached": 0,
                    "failed": 0,
                    "duration_seconds": 0,
                }

            # Generate/cache embeddings
            event_embeddings = (
                await self.event_service.get_or_generate_event_embeddings(
                    events_data, force_regenerate=force_regenerate
                )
            )

            # Calculate stats
            total_events = len(events_data)
            successful = sum(1 for emb in event_embeddings.values() if emb is not None)
            failed = total_events - successful

            elapsed = time.time() - start_time

            stats = {
                "total_events": total_events,
                "successful": successful,
                "failed": failed,
                "duration_seconds": round(elapsed, 2),
                "events_per_second": (
                    round(total_events / elapsed, 2) if elapsed > 0 else 0
                ),
            }

            logger.info(
                f"[SUCCESS] Completed event embedding generation:\n"
                f"  Total: {stats['total_events']}\n"
                f"  Successful: {stats['successful']}\n"
                f"  Failed: {stats['failed']}\n"
                f"  Duration: {stats['duration_seconds']}s\n"
                f"  Speed: {stats['events_per_second']} events/sec"
            )

            await self.event_service.close()

            return stats

        except Exception as e:
            logger.error(f"[FAILED] Batch event embedding generation failed: {e}")
            raise e
