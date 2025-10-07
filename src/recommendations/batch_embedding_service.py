"""Batch operations for generating embeddings for users and events."""

import asyncio
import time
from typing import Dict

from pymongo.asynchronous.database import AsyncDatabase

from src.recommendations.event_embedding_service import EventEmbeddingService
from src.recommendations.user_embedding_service import UserEmbeddingService
from src.recommendations.user_filter_service import UserFilterService
from src.utils.setup_logger import setup_logger

logger = setup_logger(__name__, "logs/batch_embedding_service.log")


class BatchEmbeddingService:
    """Handles batch generation of embeddings for users and events."""

    def __init__(self, db: AsyncDatabase):
        self.db = db
        self.user_embedding_service = UserEmbeddingService(db)
        self.event_embedding_service = EventEmbeddingService(db)
        self.filter_service = UserFilterService(db)

    async def generate_all_user_embeddings(
        self, force_regenerate: bool = False
    ) -> Dict[str, int | float]:
        """
        Generate embeddings for all users with complete profiles.

        Args:
            force_regenerate: If True, regenerate even if cache exists

        Returns dict with counts: generated, skipped, failed.
        """
        method = self.generate_all_user_embeddings.__name__
        try:
            start_time = time.perf_counter()

            all_user_ids = await self.filter_service.get_all_active_user_ids()
            logger.info(f"[{method}] Processing {len(all_user_ids)} active users")

            generated = 0
            skipped = 0
            failed = 0

            for idx, user_id in enumerate(all_user_ids, 1):
                try:
                    # Check if user has complete profile
                    criteria = await self.filter_service.get_user_filtering_criteria(
                        user_id
                    )
                    if not criteria:
                        skipped += 1
                        continue

                    # Check if embeddings need regeneration
                    if not force_regenerate:
                        if self.user_embedding_service.vector_store.user_embeddings_exists(
                            user_id
                        ):
                            raw_data = (
                                await self.user_embedding_service.get_raw_user_data(
                                    user_id
                                )
                            )
                            should_regen, _ = (
                                self.user_embedding_service.vector_store.should_regenerate_user_embeddings(
                                    user_id, raw_data, skip_content_check=True
                                )
                            )

                            if not should_regen:
                                skipped += 1
                                continue

                    # Generate embeddings
                    await self.user_embedding_service.get_or_generate_user_embeddings(
                        user_id, force_regenerate=force_regenerate
                    )
                    generated += 1

                    if idx % 50 == 0:
                        logger.info(
                            f"[{method}] Progress: {idx}/{len(all_user_ids)} "
                            f"({generated} generated, {skipped} skipped, {failed} failed)"
                        )

                except Exception as e:
                    logger.error(f"[{method}] Failed for user {user_id[:8]}: {e}")
                    failed += 1

            elapsed = (time.perf_counter() - start_time) / 60

            logger.info(
                f"[{method}] Complete in {elapsed:.2f}min: "
                f"{generated} generated, {skipped} skipped, {failed} failed"
            )

            return {
                "generated": generated,
                "skipped": skipped,
                "failed": failed,
                "total_processed": len(all_user_ids),
                "elapsed_minutes": round(elapsed, 2),
            }

        except Exception as e:
            logger.error(f"[{method}] Batch generation failed: {e}")
            raise e

    async def generate_all_event_embeddings(
        self, force_regenerate: bool = False, batch_size: int = 10
    ) -> Dict[str, int | float]:
        """
        Generate embeddings for all events.

        Args:
            force_regenerate: If True, regenerate even if cache exists
            batch_size: Number of events to fetch per API call

        Returns dict with counts: generated, skipped, failed.
        """
        method = self.generate_all_event_embeddings.__name__
        try:
            start_time = time.perf_counter()

            # Fetch all events
            events_data = await self.event_embedding_service.fetch_all_events(
                batch_size=batch_size
            )
            logger.info(f"[{method}] Processing {len(events_data)} events")

            if not events_data:
                logger.warning(f"[{method}] No events found")
                return {"generated": 0, "skipped": 0, "failed": 0, "total_processed": 0}

            # Generate embeddings (reuse existing batch method)
            event_embeddings = (
                await self.event_embedding_service.get_or_generate_event_embeddings(
                    events_data, force_regenerate=force_regenerate
                )
            )

            generated = len([e for e in event_embeddings.values() if e is not None])
            skipped = len(events_data) - generated

            elapsed = (time.perf_counter() - start_time) / 60

            logger.info(
                f"[{method}] Complete in {elapsed:.2f}min: "
                f"{generated} generated, {skipped} skipped"
            )

            return {
                "generated": generated,
                "skipped": skipped,
                "failed": 0,
                "total_processed": len(events_data),
                "elapsed_minutes": round(elapsed, 2),
            }

        except Exception as e:
            logger.error(f"[{method}] Batch event generation failed: {e}")
            raise e

    async def generate_all_embeddings(
        self, force_regenerate: bool = False
    ) -> Dict[str, Dict]:
        """
        Generate embeddings for both users and events.

        Returns dict with separate results for users and events.
        """
        method = self.generate_all_embeddings.__name__
        logger.info(f"[{method}] Starting batch generation for users and events")

        # Generate user embeddings
        logger.info(f"[{method}] === Generating user embeddings ===")
        user_results = await self.generate_all_user_embeddings(force_regenerate)

        # Generate event embeddings
        logger.info(f"[{method}] === Generating event embeddings ===")
        event_results = await self.generate_all_event_embeddings(force_regenerate)

        logger.info(f"[{method}] Batch generation complete")

        return {"users": user_results, "events": event_results}

    async def generate_all_user_event_scores(
        self, force_recalculate: bool = False
    ) -> Dict[str, int | float]:
        """
        Generate event recommendation scores for all users with embeddings.

        Args:
            force_recalculate: If True, recalculate even if scores exist

        Returns dict with counts: calculated, skipped, failed.
        """
        method = self.generate_all_user_event_scores.__name__
        try:
            start_time = time.perf_counter()

            from src.recommendations.score_service import ScoreService
            from src.recommendations.service import RecommendationService

            recommendation_service = RecommendationService(self.db)
            score_service = ScoreService(self.db)

            # Get all users with embeddings
            all_user_ids = await self.filter_service.get_all_active_user_ids()
            users_with_embeddings = [
                uid
                for uid in all_user_ids
                if self.user_embedding_service.vector_store.user_embeddings_exists(uid)
            ]

            logger.info(
                f"[{method}] Processing event scores for {len(users_with_embeddings)} users "
                f"with embeddings"
            )

            calculated = 0
            skipped = 0
            failed = 0

            for idx, user_id in enumerate(users_with_embeddings, 1):
                try:
                    # Check if scores already exist and are fresh
                    if not force_recalculate:
                        stored = await score_service.get_stored_scores(
                            user_id, "event", max_age_hours=24, limit=1
                        )
                        if stored:
                            skipped += 1
                            continue

                    # Generate event scores
                    scores = await recommendation_service.generate_event_scores_for_user_fast(
                        user_id, max_events=-1, force_user_regenerate=False
                    )

                    if scores:
                        # Store to database
                        await score_service.store_scores_to_database(scores)
                        calculated += 1

                    if idx % 50 == 0:
                        logger.info(
                            f"[{method}] Progress: {idx}/{len(users_with_embeddings)} "
                            f"({calculated} calculated, {skipped} skipped, {failed} failed)"
                        )

                except Exception as e:
                    logger.error(f"[{method}] Failed for user {user_id[:8]}: {e}")
                    failed += 1

            elapsed = (time.perf_counter() - start_time) / 60

            logger.info(
                f"[{method}] Complete in {elapsed:.2f}min: "
                f"{calculated} calculated, {skipped} skipped, {failed} failed"
            )

            return {
                "calculated": calculated,
                "skipped": skipped,
                "failed": failed,
                "total_processed": len(users_with_embeddings),
                "elapsed_minutes": round(elapsed, 2),
            }

        except Exception as e:
            logger.error(f"[{method}] Batch event score generation failed: {e}")
            raise e

    async def generate_all_user_match_scores(
        self, apply_filters: bool = True, force_recalculate: bool = False
    ) -> Dict[str, int | float]:
        """
        Generate user match scores for all users with embeddings.

        Args:
            apply_filters: If True, only match compatible candidates
            force_recalculate: If True, recalculate even if scores exist

        Returns dict with counts: calculated, skipped, failed.
        """
        method = self.generate_all_user_match_scores.__name__
        try:
            start_time = time.perf_counter()

            from src.recommendations.match_score_service import \
                MatchScoreService
            from src.recommendations.user_matching_service import \
                UserMatchingService

            match_service = UserMatchingService(self.db)
            score_service = MatchScoreService(self.db)

            # Get all users with embeddings
            all_user_ids = await self.filter_service.get_all_active_user_ids()
            users_with_embeddings = [
                uid
                for uid in all_user_ids
                if self.user_embedding_service.vector_store.user_embeddings_exists(uid)
            ]

            logger.info(
                f"[{method}] Processing user matches for {len(users_with_embeddings)} users "
                f"(filters={apply_filters})"
            )

            calculated = 0
            skipped = 0
            failed = 0

            for idx, user_id in enumerate(users_with_embeddings, 1):
                try:
                    # Check if scores already exist
                    if not force_recalculate:
                        stored = await score_service.get_user_matches(user_id, limit=1)
                        if stored:
                            skipped += 1
                            continue

                    # Get candidates
                    if apply_filters:
                        candidates = (
                            await self.filter_service.filter_compatible_candidates(
                                user_id
                            )
                        )
                    else:
                        candidates = [
                            uid for uid in users_with_embeddings if uid != user_id
                        ]

                    if not candidates:
                        skipped += 1
                        continue

                    # Calculate matches
                    matches = await match_service.calculate_matches_vectorized(
                        user_id, candidates
                    )

                    if matches:
                        # Store top 1000
                        await score_service.store_match_scores(user_id, matches[:1000])
                        calculated += 1

                    if idx % 50 == 0:
                        logger.info(
                            f"[{method}] Progress: {idx}/{len(users_with_embeddings)} "
                            f"({calculated} calculated, {skipped} skipped, {failed} failed)"
                        )

                except Exception as e:
                    logger.error(f"[{method}] Failed for user {user_id[:8]}: {e}")
                    failed += 1

            elapsed = (time.perf_counter() - start_time) / 60

            logger.info(
                f"[{method}] Complete in {elapsed:.2f}min: "
                f"{calculated} calculated, {skipped} skipped, {failed} failed"
            )

            return {
                "calculated": calculated,
                "skipped": skipped,
                "failed": failed,
                "total_processed": len(users_with_embeddings),
                "elapsed_minutes": round(elapsed, 2),
            }

        except Exception as e:
            logger.error(f"[{method}] Batch user match generation failed: {e}")
            raise e

    async def generate_all_embeddings_and_scores(
        self, force_regenerate: bool = False, calculate_scores: bool = True
    ) -> Dict[str, Dict]:
        """
        Full pipeline: Generate embeddings and calculate all scores.

        Args:
            force_regenerate: If True, regenerate embeddings
            calculate_scores: If True, also calculate event and user match scores

        Returns dict with results for embeddings and scores.
        """
        method = self.generate_all_embeddings_and_scores.__name__
        logger.info(f"[{method}] Starting full batch pipeline")

        # Step 1: Generate embeddings
        logger.info(f"[{method}] === Step 1: User embeddings ===")
        user_emb_results = await self.generate_all_user_embeddings(force_regenerate)

        logger.info(f"[{method}] === Step 2: Event embeddings ===")
        event_emb_results = await self.generate_all_event_embeddings(force_regenerate)

        results = {
            "embeddings": {"users": user_emb_results, "events": event_emb_results}
        }

        # Step 2: Calculate scores (if enabled)
        if calculate_scores:
            logger.info(f"[{method}] === Step 3: Event recommendation scores ===")
            event_score_results = await self.generate_all_user_event_scores(
                force_recalculate=force_regenerate
            )

            logger.info(f"[{method}] === Step 4: User match scores ===")
            match_score_results = await self.generate_all_user_match_scores(
                apply_filters=True, force_recalculate=force_regenerate
            )

            results["scores"] = {
                "event_recommendations": event_score_results,
                "user_matches": match_score_results,
            }

        logger.info(f"[{method}] Full pipeline complete")
        return results


if __name__ == "__main__":
    from src.database import connect_to_mongo, get_database

    async def test():
        await connect_to_mongo()
        db = get_database()
        service = BatchEmbeddingService(db)

        print("=" * 60)
        print("Batch Embedding Generation")
        print("=" * 60)

        # Test user embeddings only
        print("\n1. Generating user embeddings...")
        # user_results = await service.generate_all_user_embeddings(
        #    force_regenerate=False
        # )
        # print("\nUser Results:")
        # print(f"  Generated: {user_results['generated']}")
        # print(f"  Skipped: {user_results['skipped']}")
        # print(f"  Failed: {user_results['failed']}")
        # print(f"  Time: {user_results['elapsed_minutes']} minutes")

        # # Test event embeddings only
        # print("\n2. Generating event embeddings...")
        # event_results = await service.generate_all_event_embeddings(
        #     force_regenerate=False
        # )
        # print("\nEvent Results:")
        # print(f"  Generated: {event_results['generated']}")
        # print(f"  Skipped: {event_results['skipped']}")
        # print(f"  Failed: {event_results['failed']}")
        # print(f"  Time: {event_results['elapsed_minutes']} minutes")

        print("=" * 60)
        print("Full Batch Pipeline: Embeddings + Scores")
        print("=" * 60)

        results = await service.generate_all_embeddings_and_scores(
            force_regenerate=False, calculate_scores=True
        )

        print("\n=== Embeddings ===")
        print(f"Users: {results['embeddings']['users']}")
        print(f"Events: {results['embeddings']['events']}")

        if "scores" in results:
            print("\n=== Scores ===")
            print(f"Event Recs: {results['scores']['event_recommendations']}")
            print(f"User Matches: {results['scores']['user_matches']}")

    asyncio.run(test())
