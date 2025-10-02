"""Test all refactored recommendation services."""

import asyncio

from src.config import settings
from src.database import connect_to_mongo, get_database
from src.recommendations.service import RecommendationService
from src.utils.setup_logger import setup_logger

logger = setup_logger(__name__, "logs/test_services.log")


async def main():
    try:
        logger.info("[...] Connecting to database")
        await connect_to_mongo()
        db = get_database()

        # Initialize main service (which initializes all sub-services)
        service = RecommendationService(db)

        # Test 1: Initialize score collection
        print("\n" + "=" * 60)
        print("TEST 1: Initialize Score Collection")
        print("=" * 60)

        # Test 2: Get a test user
        print("\n" + "=" * 60)
        print("TEST 2: Fetch Test User")
        print("=" * 60)
        users = await db[settings.LOGIN_COLLECTION].find({}).limit(1).to_list(1)
        if not users:
            logger.error("[FAILED] No users found")
            return

        user_id = str(users[0]["_id"])
        logger.info(f"[INFO] Testing with user: {user_id[:8]}")

        # Test 3: User Embedding Service
        print("\n" + "=" * 60)
        print("TEST 3: User Embedding Service")
        print("=" * 60)
        user_embeddings = await service.user_service.get_or_generate_user_embeddings(
            user_id
        )
        logger.info(
            f"[SUCCESS] User embeddings shape: {user_embeddings['personal'].shape}"
        )

        # Test 4: Event Embedding Service
        print("\n" + "=" * 60)
        print("TEST 4: Event Embedding Service")
        print("=" * 60)
        events = await service.event_service.fetch_all_events(batch_size=10)
        logger.info(f"[SUCCESS] Fetched {len(events)} events")

        event_embeddings = await service.event_service.get_or_generate_event_embeddings(
            events[:5]
        )
        logger.info(
            f"[SUCCESS] Generated/retrieved {len(event_embeddings)} event embeddings"
        )

        # Test 5: Generate recommendations (full flow)
        print("\n" + "=" * 60)
        print("TEST 5: Full Recommendation Flow")
        print("=" * 60)
        scores = await service.generate_event_scores_for_user_with_cache(
            user_id, max_events=5
        )
        logger.info(f"[SUCCESS] Generated {len(scores)} recommendation scores")

        for i, score in enumerate(scores[:3]):
            logger.info(
                f"  {i+1}. Event {score['target_id']}: {score['percentage_score']:.1f}%"
            )

        # Test 6: Store scores to database
        print("\n" + "=" * 60)
        print("TEST 6: Store Scores to Database")
        print("=" * 60)
        stored = await service.store_scores_to_database(scores)
        logger.info(f"[SUCCESS] Stored scores: {stored}")

        # Test 7: Retrieve stored scores
        print("\n" + "=" * 60)
        print("TEST 7: Retrieve Stored Scores")
        print("=" * 60)
        retrieved = await service.get_stored_scores(user_id)
        logger.info(
            f"[SUCCESS] Retrieved {len(retrieved) if retrieved else 0} stored scores"
        )

        # Test 8: Batch Service - Generate 5 users
        print("\n" + "=" * 60)
        print("TEST 8: Batch Generate User Embeddings (5 users)")
        print("=" * 60)
        batch_stats = await service.batch_service.generate_all_user_embeddings(
            batch_size=5, skip_existing=True
        )
        logger.info(f"[SUCCESS] Batch stats: {batch_stats}")

        # Test 9: Batch Service - Generate all events
        print("\n" + "=" * 60)
        print("TEST 9: Batch Generate Event Embeddings")
        print("=" * 60)
        event_stats = await service.batch_service.generate_all_event_embeddings(
            batch_size=50
        )
        logger.info(f"[SUCCESS] Event stats: {event_stats}")

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        logger.error(f"[FAILED] Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
