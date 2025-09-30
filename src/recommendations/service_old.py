"""
Core recommendation service business logic for multi-vector matching system.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional

import numpy as np
from bson import ObjectId
from pymongo.asynchronous.database import AsyncDatabase

from src.config import settings
from src.database import connect_to_mongo, get_database
from src.embeddings.service import EmbeddingService
from src.events.client import EventsClient
from src.events.schemas import Event
from src.events.service import EventService
from src.matching.constants import VectorType
from src.profiles.schemas import (ContextBuilder, LoginInfo,
                                  OrganisationProfile, UserData, UserProfile)
from src.profiles.service import ProfileService
from src.utils.setup_logger import setup_logger
from src.vector_store.service import VectorStoreService

logger = setup_logger(__name__, "logs/recommendation_service.log")


class RecommendationService:
    """Core service for generating recommendations using multi-vector approach."""

    def __init__(self, database: AsyncDatabase):
        self.db = database
        self.embedding_service = EmbeddingService()
        self.profile_service = ProfileService()

        logger.info("[SUCCESS] Recommendation service initialized")

    async def check_database_collecitons(self):
        """Check database collections and returns counts"""
        try:
            logger.info("[***] Checking database collections")

            collections = {
                "login_info": settings.LOGIN_COLLECTION,
                "user_profiles": settings.USERS_PROFILE_COLLECTION,
                "org_profiles": settings.ORGANISATION_PROFILE_COLLECTION,
                "context_builder": settings.CONTEXT_BUILDER_COLLECTION,
                "events": settings.EVENTS_COLLECTION,
            }

            for name, collection in collections.items():
                count = await self.db[collection].count_documents({})
                logger.info(f"[INFO] {name}: {count} documents")

            logger.info("[SUCCESS] Database connection checked")

        except Exception as e:
            logger.error(f"[FAILED] Database check failed: {e}")

    async def get_raw_user_data(self, user_id: str) -> Dict[str, Any]:
        """Fetch user data across four collections"""
        try:
            logger.info(f"[***] Fetching data for user: {user_id}")
            start_time = time.perf_counter()

            user_obj_id = ObjectId(user_id)

            user = await self.db[settings.LOGIN_COLLECTION].find_one(
                {"_id": user_obj_id}
            )
            if not user:
                raise ValueError(f"User not found: {user_id}")

            org_id = user.get("organisation_profile_id")

            profile = await self.db[settings.USERS_PROFILE_COLLECTION].find_one(
                {"user_id": user_obj_id}
            )
            org = (
                await self.db[settings.ORGANISATION_PROFILE_COLLECTION].find_one(
                    {"_id": org_id}
                )
                if org_id
                else {}
            )
            context = await self.db[settings.CONTEXT_BUILDER_COLLECTION].find_one(
                {"user_id": user_obj_id}
            )

            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000

            logger.info(
                f"[SUCCESS] User data fetched in {execution_time:.2f}ms - Profile: {'Yes' if profile else 'No'}, Org: {'Yes' if org else 'No'}, Context: {'Yes' if context else 'No'}"
            )

            return {"login": user, "profile": profile, "org": org, "context": context}

        except Exception as e:
            logger.error(
                f"[FAILED] Fetching user data across multiple collections failed: {e}"
            )
            raise e

    async def get_user_data(
        self, user_id: str, raw_data: Optional[Dict[str, Any]] = None
    ) -> UserData:
        try:
            if not raw_data:
                raw_data = await self.get_raw_user_data(user_id)

            return UserData(
                login=LoginInfo(**raw_data["login"]) if raw_data["login"] else None,
                profile=(
                    UserProfile(**raw_data["profile"]) if raw_data["profile"] else None
                ),
                org=OrganisationProfile(**raw_data["org"]) if raw_data["org"] else None,
                context=(
                    ContextBuilder(**raw_data["context"])
                    if raw_data["context"]
                    else None
                ),
            )

        except Exception as e:
            logger.error(f"Failed to create UserData model: {e}")
            raise e

    async def generate_user_embeddings(
        self, user_data: UserData
    ) -> Dict[str, np.ndarray]:
        """Generate all 3 embeddings for a user (personal, org, intent)"""
        try:
            logger.info("[***] Generating all three embeddings of a user")

            texts = self.profile_service.create_all_texts(user_data)
            embeddings = await self.embedding_service.create_embeddings(texts)

            return {
                "personal": embeddings[0],
                "org": embeddings[1],
                "intent": embeddings[2],
            }

        except Exception as e:
            logger.error(f"[FAILED] Multi user vector emebedding failed: {e}")
            raise e

    def normalize_scores_to_percentage(self, recommendation_scores):
        """Nomalize similarity scores to percentage range (10-95%)"""
        if not recommendation_scores:
            return recommendation_scores

        scores = [item["similarity_score"] for item in recommendation_scores]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            for item in recommendation_scores:
                item["percentage_score"] = 52.5
            return recommendation_scores

        for item in recommendation_scores:
            orginal_score = item["similarity_score"]
            normalized = (
                10 + ((orginal_score - min_score) / (max_score - min_score)) * 85
            )
            item["percentage_score"] = normalized

        return recommendation_scores

    async def generate_event_scores_for_user(
        self, user_id: str, max_events: Optional[int] = None, min_score: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Generate event recommendation scores for a user based on similarity scores."""
        max_events = max_events or -1

        try:
            logger.info(
                f"[***] Generating event recommendation scores for user: {user_id[:8]}..."
            )

            user_data = await self.get_user_data(user_id)
            user_embeddings = await self.generate_user_embeddings(user_data)

            events_client = EventsClient()
            events_data = await events_client.get_all_public_events(batch_size=100)

            if not events_data:
                logger.warning("No events found")
                return []

            events_texts = EventService.format_events_for_embedding(events_data)
            events_embeddings = await self.embedding_service.create_embeddings(
                events_texts
            )

            recommendation_scores = []

            for event_data, event_embeddings in zip(events_data, events_embeddings):
                similarity_result = (
                    self.embedding_service.calculate_multi_vector_similarity(
                        user_embeddings, event_embeddings
                    )
                )

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

            recommendation_scores = recommendation_scores[:max_events]

            recommendation_scores.sort(
                key=lambda x: x["similarity_score"], reverse=True
            )

            recommendation_scores = self.normalize_scores_to_percentage(
                recommendation_scores
            )

            await events_client.close()

            logger.info(
                f"[SUCCESS] Generated {len(recommendation_scores)} scores for events"
            )

            return recommendation_scores

        except Exception as e:
            logger.error(f"[FAILED] Event recommendation generation failed: {e}")
            raise e

    async def generate_event_scores_for_user_with_cache(
        self, user_id: str, max_events: Optional[int] = None, min_score: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Generate event recommendation scores for a user with caching."""
        max_events = max_events or -1

        try:
            logger.info(f"[***] Generating event scores for user: {user_id}...")
            vector_store = VectorStoreService()

            user_embeddings = vector_store.get_user_embeddings(user_id)

            if not user_embeddings:
                logger.info(f"[INFO] No cached user embeddings, generating new ones")
                raw_data = await self.get_raw_user_data(user_id)
                user_data = await self.get_user_data(user_id, raw_data)
                user_embeddings = await self.generate_user_embeddings(user_data)

                texts = self.profile_service.create_all_texts(user_data)
                vector_store.store_user_embeddings(
                    user_id, user_embeddings, texts, raw_data
                )
            else:
                logger.info(f"[INFO] Using cached user embeddings")

            events_client = EventsClient()
            events_data = await events_client.get_all_public_events(batch_size=10)

            if not events_data:
                logger.warning("No events found")
                await events_client.close()
                return []

            events_texts = EventService.format_events_for_embedding(events_data)
            stale_event_ids = vector_store.get_stale_event_ids(
                events_data, events_texts
            )

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

                vector_store.store_event_embeddings_batch(
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
            cached_embeddings = vector_store.get_all_event_embeddings(event_ids)

            for event_id, embedding in stale_embeddings_map.items():
                cached_embeddings[event_id] = embedding

            recommendation_scores = []
            missing_embeddings = []

            for event_data in events_data:
                event_id = event_data["id"]
                event_embedding = cached_embeddings.get(event_id)

                if event_embedding is None:
                    missing_embeddings.append(event_id)
                    logger.warning(f"[WARNING] Missing embedding for event {event_id}")
                    continue

                similarity_result = (
                    self.embedding_service.calculate_multi_vector_similarity(
                        user_embeddings, event_embedding
                    )
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

            if missing_embeddings:
                logger.warning(
                    f"[WARNING] Skipped {len(missing_embeddings)} events due to missing embeddings"
                )

            recommendation_scores.sort(
                key=lambda x: x["similarity_score"], reverse=True
            )
            recommendation_scores = recommendation_scores[
                : max_events if max_events > 0 else len(recommendation_scores)
            ]

            recommendation_scores = self.normalize_scores_to_percentage(
                recommendation_scores
            )

            await events_client.close()

            logger.info(
                f"[SUCCESS] Generated {len(recommendation_scores)} event scores "
                f"({len(stale_event_ids)} regenerated, "
                f"{len(event_ids) - len(stale_event_ids)} cached)"
            )

            return recommendation_scores

        except Exception as e:
            logger.error(f"[FAILED] Event recommendation generation failed: {e}")
            raise e


async def main():
    try:
        logger.info("[...] Connecting to database")
        await connect_to_mongo()
        db = get_database()

        logger.info("[INFO] Initializing RecommendationService")
        service = RecommendationService(db)

        # Test 1: Check database collections
        await service.check_database_collecitons()

        # Test 2: Test embedding service connection
        logger.info("[***] Testing embedding service connection")
        connection_ok = await service.embedding_service.test_connection()
        if not connection_ok:
            logger.error("[FAILED] Embedding service connection failed")
            return

        # Test 3: Get a real user and test data fetching
        logger.info("[***] Testing user data fetching")
        users = await db[settings.LOGIN_COLLECTION].find({}).limit(1).to_list(1)
        if not users:
            logger.error("[FAILED] No users found in database")
            return

        user_id = str(users[0]["_id"])
        logger.info(f"[***] Testing with user: {user_id[:8]}...")

        # Test user data fetching
        user_data = await service.get_user_data(user_id)
        logger.info("[SUCCESS] User data fetched successfully")

        # Test 4: Test profile service text creation
        logger.info("[***] Testing profile text creation")
        texts = service.profile_service.create_all_texts(user_data)
        logger.info(f"[SUCCESS] Created {len(texts)} text representations")
        for i, text in enumerate(texts):
            logger.info(f"  Text {i+1}: {text[:50]}...")

        # Test 5: Test embedding generation
        logger.info("[***] Testing embedding generation")
        user_embeddings = await service.generate_user_embeddings(user_data)
        logger.info("[SUCCESS] Generated embeddings:")
        for vector_type, embedding in user_embeddings.items():
            logger.info(f"  {vector_type}: shape {embedding.shape}")

        # Test 6: Test similarity calculation
        logger.info("[***] Testing similarity calculation")
        similarity_result = service.embedding_service.calculate_multi_vector_similarity(
            user_embeddings, user_embeddings["personal"]  # Self-similarity test
        )
        logger.info("[SUCCESS] Self-similarity test:")
        logger.info(f"  Final score: {similarity_result.final_score:.3f}")
        logger.info(f"  Personal: {similarity_result.personal:.3f}")
        logger.info(f"  Org: {similarity_result.org:.3f}")
        logger.info(f"  Intent: {similarity_result.intent:.3f}")

        # Test 7: Cross-user similarity (if we have 2+ users)
        users = await db[settings.LOGIN_COLLECTION].find({}).limit(2).to_list(2)
        if len(users) >= 2:
            logger.info("[***] Testing cross-user similarity")
            user2_id = str(users[1]["_id"])
            user2_data = await service.get_user_data(user2_id)
            user2_embeddings = await service.generate_user_embeddings(user2_data)

            cross_similarity = (
                service.embedding_service.calculate_multi_vector_similarity(
                    user_embeddings, user2_embeddings["personal"]
                )
            )
            logger.info(
                f"[SUCCESS] Cross-user similarity: {cross_similarity.final_score:.3f}"
            )

        logger.info("[***] Testing event scoring for user")
        event_scores = await service.generate_event_scores_for_user(
            user_id, max_events=5, min_score=0.2
        )

        if event_scores:
            logger.info(f"[SUCCESS] Generated {len(event_scores)} event scores:")
            for i, score_data in enumerate(event_scores):
                logger.info(f"  Event {i+1}:")
                logger.info(f"    Target ID: {score_data['target_id']}")
                logger.info(f"    Final Score: {score_data['similarity_score']:.3f}")
                logger.info(
                    f"    FInal Percentage: {score_data['percentage_score']:.3f}"
                )
                logger.info(
                    f"    Personal: {score_data['similarity_breakdown']['personal']:.3f}"
                )
                logger.info(f"    Org: {score_data['similarity_breakdown']['org']:.3f}")
                logger.info(
                    f"    Intent: {score_data['similarity_breakdown']['intent']:.3f}"
                )
                logger.info("    ---")
        else:
            logger.info("[INFO] No event scores generated")

        logger.info("[***] Testing cached event scoring")
        cached_scores = await service.generate_event_scores_for_user_with_cache(
            user_id, max_events=5
        )
        logger.info(f"[SUCCESS] Generated {len(cached_scores)} cached event scores")

        logger.info("[SUCCESS] All RecommendationService tests passed!")

    except Exception as e:
        logger.error(f"RecommendationService test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
