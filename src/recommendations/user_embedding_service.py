"""Service for user embedding operations."""

import asyncio
import time
from typing import Any, Dict, Optional

import numpy as np
from bson import ObjectId
from pymongo.asynchronous.database import AsyncDatabase

from src.config import settings
from src.embeddings.service import EmbeddingService
from src.profiles.schemas import (ContextBuilder, LoginInfo,
                                  OrganisationProfile, UserData, UserProfile)
from src.profiles.service import ProfileService
from src.utils.setup_logger import setup_logger
from src.vector_store.service import VectorStoreService

logger = setup_logger(__name__, "logs/user_embedding_service.log")


class UserEmbeddingService:
    """Handles user-specific embedding operations."""

    def __init__(self, db: AsyncDatabase):
        self.db = db
        self.embedding_service = EmbeddingService()
        self.profile_service = ProfileService()
        self.vector_store = VectorStoreService()

    async def get_raw_user_data(self, user_id: str) -> Dict[str, Any]:
        """Fetch user data across four collections."""
        try:
            logger.info(
                f"[***][UserEmbeddingService - Data Fetcher] Fetching data for user: {user_id}"
            )
            start_time = time.perf_counter()

            user_obj_id = ObjectId(user_id)

            t1 = time.perf_counter()
            user = await self.db[settings.LOGIN_COLLECTION].find_one(
                {"_id": user_obj_id}
            )
            login_time = (time.perf_counter() - t1) * 1000
            logger.info(f"[TIMING] Login query: {login_time:.2f}ms")

            if not user:
                raise ValueError(f"User not found: {user_id}")

            org_id = user.get("organisation_profile_id")

            t2 = time.perf_counter()

            profile_task = self.db[settings.USERS_PROFILE_COLLECTION].find_one(
                {"user_id": user_obj_id}
            )

            context_task = self.db[settings.CONTEXT_BUILDER_COLLECTION].find_one(
                {"user_id": user_obj_id}
            )
            if org_id:
                org_task = self.db[settings.ORGANISATION_PROFILE_COLLECTION].find_one(
                    {"_id": org_id}
                )

                profile, org, context = await asyncio.gather(
                    profile_task, org_task, context_task
                )
            else:
                profile, context = await asyncio.gather(profile_task, context_task)
                org = {}

            parallel_time = (time.perf_counter() - t2) * 1000
            logger.info(
                f"[TIMING] Parallel queries (3 collections): {parallel_time:.2f}ms"
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
        """Create UserData pydantic model from raw data."""
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
        """Generate all 3 embeddings for a user (personal, org, intent)."""
        try:
            logger.info(
                "[***][UserEmbeddingService Generate User Embeddings] Generating all three embeddings of a user"
            )

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

    async def get_or_generate_user_embeddings(
        self, user_id: str, force_regenerate: bool = False
    ) -> Dict[str, np.ndarray]:
        """Get cached embeddings or generate new ones if needed."""
        method = self.get_or_generate_user_embeddings.__name__
        try:
            if not force_regenerate:
                cached_embeddings = self.vector_store.get_user_embeddings(user_id)

                if cached_embeddings:
                    raw_data = await self.get_raw_user_data(user_id)
                    should_regen, reason = (
                        self.vector_store.should_regenerate_user_embeddings(
                            user_id, raw_data, skip_content_check=True
                        )
                    )

                    if not should_regen:
                        logger.info(
                            f"[INFO] [{method}] Using cached embeddings for user {user_id}"
                        )
                        return cached_embeddings

                    logger.info(f"[INFO] Cache invalid for user {user_id}: {reason}")

            logger.info(
                f"[***] [{method}] Generating new embeddings for user {user_id}"
            )
            raw_data = await self.get_raw_user_data(user_id)

            if not any(raw_data.values()):
                raise ValueError(f"User {user_id} has no data in any collection")

            user_data = await self.get_user_data(user_id, raw_data)
            embeddings = await self.generate_user_embeddings(user_data)

            texts = self.profile_service.create_all_texts(user_data)
            self.vector_store.store_user_embeddings(
                user_id, embeddings, texts, raw_data
            )

            logger.info(
                f"[{method} - SUCCESS] Generated and cached embeddings for user {user_id}"
            )
            return embeddings

        except Exception as e:
            logger.error(
                f"[{method} - FAILED] Error getting/generating embeddings for user {user_id}: {e}"
            )
            raise e
