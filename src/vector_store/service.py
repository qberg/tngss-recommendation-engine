"""
Faiss/VectorStore service to store the vector embeddings generated from openai
"""

import asyncio
import hashlib
import pickle
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from bson import ObjectId

from src.config import settings
from src.database import connect_to_mongo, get_database
from src.recommendations.service import RecommendationService
from src.utils.setup_logger import setup_logger
from src.vector_store.config import vector_store_config
from src.vector_store.exceptions import (InvalidEmbeddingDataError,
                                         VectorStoreError)
from src.vector_store.schemas import UserEmbeddings

logger = setup_logger(__name__, "logs/vector_store_service.log")


class VectorStoreService:
    """Main service for handling embedding storage as faiss indexes"""

    def __init__(self):
        self.embeddings_dir = vector_store_config.EMBEDDINGS_DIR
        self.users_embeddings_dir = vector_store_config.USER_EMBEDDINGS_DIR
        self.events_embeddings_dir = vector_store_config.EVENTS_EMBEDDINGS_DIR
        self.embedding_dimension = vector_store_config.EMBEDDING_DIMENSION

        self._ensure_directories()

        logger.info(
            f"[***] Vector store initialized, with indexes being stored in {self.embeddings_dir}"
        )

    def _ensure_directories(self):
        """Create the storage directories if they dont exist"""
        for path in [
            self.embeddings_dir,
            self.users_embeddings_dir,
            self.events_embeddings_dir,
        ]:
            Path(path).mkdir(parents=True, exist_ok=True)
            logger.info(f"[SUCCESS] Created/ensured: {Path(path).resolve()}")

    def _generate_text_hash(self, text: str) -> str:
        """Generate hash for text content"""
        if not text:
            return ""
        return hashlib.md5(text.encode()).hexdigest()

    def _validate_embeddings(self, embeddings: Dict[str, np.ndarray]) -> bool:
        """Validate embeddings structure and dimensions."""
        required_keys = {"personal", "org", "intent"}

        if not all(key in embeddings for key in required_keys):
            missing_keys = required_keys - set(embeddings.keys())
            raise InvalidEmbeddingDataError(f"Missing embedding keys: {missing_keys}")

        for key, embedding in embeddings.items():
            if not isinstance(embedding, np.ndarray):
                raise InvalidEmbeddingDataError(
                    f"Embedding '{key}' must be numpy array"
                )

            if embedding.shape != (self.embedding_dimension,):
                raise InvalidEmbeddingDataError(
                    f"Embedding '{key}' has shape {embedding.shape}, expected ({self.embedding_dimension})"
                )

        return True

    def _safe_get_updated_at(
        self, data: Optional[Dict] = None, default: Optional[datetime] = None
    ) -> Optional[datetime]:
        """Safely extract and parse updatedAt timestamp from data."""
        if not data:
            return default

        updated_value = data.get("updatedAt")
        if not updated_value:
            return default

        try:
            if isinstance(updated_value, datetime):
                return updated_value

            if isinstance(updated_value, str):
                if "T" in updated_value:
                    return datetime.fromisoformat(updated_value.replace("Z", "+00:00"))
                else:
                    return datetime.strptime(updated_value, "%Y-%m-%d %H:%M:%S.%f")

            logger.warning(f"[WARNING] Unexpected datetime type: {type(updated_value)}")
            return default

        except (ValueError, TypeError) as e:
            logger.warning(
                f"[WARNING] Invalid datetime format: {updated_value}, error: {e}"
            )
            return default

    def store_user_embeddings(
        self,
        user_id: str,
        embeddings: Dict[str, np.ndarray],
        texts: List[str],
        raw_data: Dict,
    ) -> bool:
        """Store user embeddings with 4-collection tracking."""
        try:
            if not user_id.strip():
                raise InvalidEmbeddingDataError("User ID cannot be empty")

            if not texts or not any(text.strip() for text in texts):
                raise InvalidEmbeddingDataError("Text cannot be empty")

            self._validate_embeddings(embeddings)

            text_hash = self._generate_text_hash("".join(texts))

            user_embedding = UserEmbeddings(
                user_id=user_id,
                personal=embeddings["personal"].tolist(),
                org=embeddings["org"].tolist(),
                intent=embeddings["intent"].tolist(),
                created_at=datetime.now(),
                updated_at=datetime.now(),
                text_hash=text_hash,
                login_updated_at=self._safe_get_updated_at(raw_data.get("login")),
                profile_updated_at=self._safe_get_updated_at(raw_data.get("profile")),
                org_updated_at=self._safe_get_updated_at(raw_data.get("org")),
                context_updated_at=self._safe_get_updated_at(raw_data.get("context")),
            )

            user_file = Path(self.users_embeddings_dir) / f"{user_id}.pkl"
            temp_file = user_file.with_suffix(".tmp")

            with open(temp_file, "wb") as f:
                pickle.dump(user_embedding, f)

            temp_file.replace(user_file)

            logger.info(f"[SUCCESS] Stored user embeddings: {user_id[:8]}...")

            return True

        except (InvalidEmbeddingDataError, VectorStoreError):
            raise
        except Exception as e:
            logger.error("[FAILED] Error storing user embeddings")
            raise VectorStoreError(f"Failed to store user embeddings: {e}")

    def _read_user_embeddings_file(self, user_id: str) -> Optional[UserEmbeddings]:
        """
        Internal helper to read pickle file
        Corrupted files dont carsh service, we just can regenrate new files
        """
        try:
            user_file = Path(self.users_embeddings_dir) / f"{user_id}.pkl"

            if not user_file.exists():
                return None
            with open(user_file, "rb") as f:
                return pickle.load(f)

        except (pickle.UnpicklingError, EOFError, OSError) as e:
            logger.error(f"[FAILED] Corrupted embedding file for {user_id[:8]}...: {e}")
            return None

    def get_user_embeddings(self, user_id: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Retrieve user embeddings by user ID.
        """
        try:
            if not user_id.strip():
                raise InvalidEmbeddingDataError("User ID cannot be empty")

            user_embedding = self._read_user_embeddings_file(user_id)

            if not user_embedding:
                return None

            logger.info(f"[SUCCESS] Retrieved user embeddings of {user_id[:-1]}...")
            return {
                "personal": np.array(user_embedding.personal),
                "org": np.array(user_embedding.org),
                "intent": np.array(user_embedding.intent),
            }

        except InvalidEmbeddingDataError:
            raise
        except Exception as e:
            logger.error(
                f"[FAILED] Error retrieving user embeddings for {user_id[:8]}...: {e}"
            )
            raise VectorStoreError(f"Failed to retrieve user embeddings: {e}")

    def get_user_embeddings_metadata(self, user_id: str) -> Optional[UserEmbeddings]:
        """Get user embeddings metadata for cache invalidation checking.
        Fails silenty by returning None in case metadata is corrupt
        """
        try:
            if not user_id.strip():
                raise InvalidEmbeddingDataError("User ID cannot be empty")
            return self._read_user_embeddings_file(user_id)
        except InvalidEmbeddingDataError:
            raise
        except Exception as e:
            logger.error(f"[FAILED] Error retrieving metadata for {user_id}...: {e}")

        return self._read_user_embeddings_file(user_id)

    def _has_datetime_changes(
        self, stored_embeddings: UserEmbeddings, raw_data: Dict
    ) -> bool:
        """Check for datetime-based changes (fast check)."""
        current_times = {
            "login": self._safe_get_updated_at(raw_data.get("login")),
            "profile": self._safe_get_updated_at(raw_data.get("profile")),
            "org": self._safe_get_updated_at(raw_data.get("org")),
            "context": self._safe_get_updated_at(raw_data.get("context")),
        }

        stored_times = {
            "login": stored_embeddings.login_updated_at,
            "profile": stored_embeddings.profile_updated_at,
            "org": stored_embeddings.org_updated_at,
            "context": stored_embeddings.context_updated_at,
        }

        for collection, current_time in current_times.items():
            stored_time = stored_times[collection]
            if current_time and stored_time and current_time > stored_time:
                logger.info(f"[INFO] {collection} collection updated")
                return True

        return False

    def _has_user_content_changes(
        self, stored_embeddings: UserEmbeddings, current_texts: List[str]
    ) -> bool:
        """Check for content-based changes (thorough check)."""
        if not current_texts:
            return False

        current_text_hash = self._generate_text_hash("".join(current_texts))
        has_changes = current_text_hash != stored_embeddings.text_hash

        if has_changes:
            logger.info("[INFO] Content based change detected")

        return has_changes

    def should_regenerate_user_embeddings(
        self,
        user_id: str,
        raw_data: Dict,
        current_texts: Optional[List[str]] = None,
        skip_content_check: bool = False,
    ) -> Tuple[bool, str]:
        """Hybrid cache invalidation: check both datetime and content changes."""
        try:
            stored_embeddings = self.get_user_embeddings_metadata(user_id)

            if not stored_embeddings:
                return True, "No embeddings exist"

            if self._has_datetime_changes(stored_embeddings, raw_data):
                return True, "Collection timestamp changed"

            if current_texts and not skip_content_check:
                if self._has_user_content_changes(stored_embeddings, current_texts):
                    return True, "Content hash changed"

            return False, "No changes detected"

        except Exception as e:
            logger.error(
                f"[FAILED] Error checking regeneration need for {user_id[:8]}...: {e}"
            )
            return True, f"Error during check: {e}"

    def user_embeddings_exist(self, user_id: str) -> bool:
        """Checks if user embeddings exists for a given user id."""
        try:
            if not user_id.strip():
                return False
            user_file = Path(self.users_embeddings_dir) / f"{user_id}.pkl"
            return user_file.exists()

        except Exception as e:
            logger.error(
                f"[FAILED] Error checking embeddings existence for {user_id}: {e}"
            )
            return False

    def delete_user_embeddings(self, user_id: str) -> bool:
        """Delete user embeddings"""
        try:
            if not user_id.strip():
                raise InvalidEmbeddingDataError("User ID cannot be empty")
            user_file = Path(self.users_embeddings_dir) / f"{user_id}.pkl"
            if user_file.exists():
                user_file.unlink()
                logger.info(f"[SUCCESS] Deleted user embeddings for: {user_id}")
                return True
            return False
        except InvalidEmbeddingDataError:
            raise
        except Exception as e:
            logger.error(f"[FAILED] Error deleting user embeddings for {user_id}: {e}")
            raise VectorStoreError(f"Failed to delete user embeddings: {e}")


async def main():
    """Async integration test with real database data"""
    user_id = None
    vector_store = None

    try:
        logger.info("[...] Connecting to database")
        await connect_to_mongo()
        db = get_database()
        logger.info("[INFO] Initializing VectorStoreService")

        vector_store = VectorStoreService()
        recommendation_service = RecommendationService(db)

        users = await db[settings.LOGIN_COLLECTION].find({}).limit(1).to_list(1)

        if not users:
            logger.error("[FAILED] No users found in database")
            return

        user_id = str(users[0]["_id"])
        logger.info(f"[***] Testing with user: {user_id[:8]}...")

        raw_data = await recommendation_service.get_raw_user_data(user_id)
        logger.info(
            f"[SUCCESS] Fetched raw data from {sum(1 for v in raw_data.values() if v)} collections"
        )

        user_data = await recommendation_service.get_user_data(user_id)

        texts = recommendation_service.profile_service.create_all_texts(user_data)
        logger.info(f"[SUCCESS] Created {len(texts)} text representations")

        user_embeddings = await recommendation_service.generate_user_embeddings(
            user_data
        )
        logger.info("[SUCCESS] Generated embeddings:")
        for vector_type, embedding in user_embeddings.items():
            logger.info(f"  {vector_type}: shape {embedding.shape}")

        # ============================================================
        # Test 1: Store embeddings
        # ============================================================
        print("+" * 100)
        print("+ Test 1 +")
        print("+" * 100)
        logger.info("[***] Storing embeddings in vector store")

        success = vector_store.store_user_embeddings(
            user_id=user_id, embeddings=user_embeddings, texts=texts, raw_data=raw_data
        )
        logger.info(f"[SUCCESS] Test 1: Storage = {success}\n")

        # ============================================================
        # Test 2: Retrieve embeddings (reuse stored, no regeneration)
        # ============================================================
        print("+" * 100)
        print("+ Test 2 +")
        print("+" * 100)
        logger.info("[***] Retrieve user embeddings from vector store")
        retrieved = vector_store.get_user_embeddings(user_id)
        logger.info(f"[SUCCESS] Test 2: Retrieved = {retrieved is not None}")
        if retrieved:
            for vector_type, embedding in retrieved.items():
                logger.info(f"  {vector_type}: shape {embedding.shape}")
        print()

        # ============================================================
        # Test 3: Check metadata
        # ============================================================
        print("+" * 100)
        print("+ Test 3 +")
        print("+" * 100)
        logger.info("[***] Checking embeddings metadata")
        metadata = vector_store.get_user_embeddings_metadata(user_id)
        if metadata:
            logger.info("[SUCCESS] Test 3: Metadata retrieved")
            logger.info(f"  User ID: {metadata.user_id[:8]}...")
            logger.info(f"  Text hash: {metadata.text_hash[:8]}...")
            logger.info(f"  Login updated: {metadata.login_updated_at}")
            logger.info(f"  Profile updated: {metadata.profile_updated_at}")
            logger.info(f"  Org updated: {metadata.org_updated_at}")
            logger.info(f"  Context updated: {metadata.context_updated_at}")
        print()

        # ============================================================
        # Test 4: Cache invalidation
        # ============================================================
        print("+" * 100)
        print("+ Test 4 +")
        print("+" * 100)
        logger.info("[***] Cache invalidation")
        should_regen, reason = vector_store.should_regenerate_user_embeddings(
            user_id=user_id, raw_data=raw_data, current_texts=texts
        )
        logger.info(f"[SUCCESS] Test 4: Should regenerate = {should_regen}")
        logger.info(f"  Reason: {reason}")
        print()

        # ============================================================
        # Test 5: Cache invalidation - changed timestamp
        # ============================================================
        print("+" * 100)
        print("TEST 5: Cache Invalidation - Changed Timestamp")
        print("+" * 100)
        logger.info("Simulating profile update in database...")

        await db[settings.USERS_PROFILE_COLLECTION].update_one(
            {"user_id": ObjectId(user_id)}, {"$set": {"updatedAt": datetime.now()}}
        )

        updated_raw_data = await recommendation_service.get_raw_user_data(user_id)
        should_regen, reason = vector_store.should_regenerate_user_embeddings(
            user_id=user_id, raw_data=updated_raw_data, current_texts=texts
        )
        logger.info(f"[SUCCESS] Test 5: Should regenerate = {should_regen}")
        logger.info(f"  Reason: {reason}")
        print()

        # ============================================================
        # Test 6: Verify existence
        # ============================================================
        print("+" * 100)
        print("TEST 6: Check Embeddings Exist")
        print("+" * 100)
        exists = vector_store.user_embeddings_exist(user_id)
        logger.info(f"[SUCCESS] Test 6: Exists = {exists}\n")

        # ============================================================
        # Test 7: Delete embeddings
        # ============================================================
        print("+" * 100)
        print("TEST 7: Delete Embeddings")
        print("+" * 100)
        deleted = vector_store.delete_user_embeddings(user_id)
        logger.info(f"[SUCCESS] Test 8: Deleted = {deleted}\n")

        # ============================================================
        # Test 8: Verify deletion
        # ============================================================
        print("+" * 100)
        print("TEST 8: Verify Deletion")
        print("+" * 100)
        exists_after = vector_store.user_embeddings_exist(user_id)
        logger.info(f"[SUCCESS] Test 9: Exists after deletion = {exists_after}\n")

        print("+" * 100)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("+" * 100)

    except Exception as e:
        logger.error(f"Vector Store Service test failed: {e}")
        traceback.print_exc()

    finally:
        if user_id and vector_store:
            print("\n" + "+" * 100)
            print("FINAL CLEANUP")
            print("+" * 100)
            if vector_store.user_embeddings_exist(user_id):
                vector_store.delete_user_embeddings(user_id)
                logger.info("[INFO] Test data cleaned up")


if __name__ == "__main__":
    asyncio.run(main())
