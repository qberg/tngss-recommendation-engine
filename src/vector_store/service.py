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
from src.events.client import EventsClient
from src.events.service import EventService
from src.utils.setup_logger import setup_logger
from src.vector_store.config import vector_store_config
from src.vector_store.exceptions import (InvalidEmbeddingDataError,
                                         VectorStoreError)
from src.vector_store.schemas import EventEmbedding, UserEmbeddings

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

    def store_event_embedding(
        self,
        event_id: str,
        embedding: np.ndarray,
        event_text: str,
        api_updated_at: Optional[datetime] = None,
    ) -> bool:
        """Store single event embedding as pkl file"""
        try:
            if not event_id.strip():
                raise InvalidEmbeddingDataError("Event ID cannot be empty")

            if not event_text.strip():
                raise InvalidEmbeddingDataError("Event text content cannot be empty")

            if embedding.shape != (self.embedding_dimension,):
                raise InvalidEmbeddingDataError(
                    f"Embedding has shape {embedding.shape}, expected {self.embedding_dimension}"
                )

            text_hash = self._generate_text_hash(event_text)

            event_embedding = EventEmbedding(
                event_id=event_id,
                embedding=embedding.tolist(),
                created_at=datetime.now(),
                updated_at=datetime.now(),
                text_hash=text_hash,
                api_updated_at=api_updated_at,
            )

            event_file = Path(self.events_embeddings_dir) / f"{event_id}.pkl"
            temp_file = event_file.with_suffix(".tmp")

            with open(temp_file, "wb") as f:
                pickle.dump(event_embedding, f)

            temp_file.replace(event_file)

            logger.info(f"[SUCCESS] Stored event embedding: {event_id}")

            return True

        except (InvalidEmbeddingDataError, VectorStoreError):
            raise
        except Exception as e:
            logger.error(f"[FAILED] Error storing event embedding for {event_id}: {e}")
            raise VectorStoreError(f"Failed to store event emebedidng: {e}")

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

    def _read_event_embedding_file(self, event_id: str) -> Optional[EventEmbedding]:
        """
        Internal helper to read pickle file
        Corrupted files dont carsh service, we just can regenrate new files
        """
        try:
            event_file = Path(self.events_embeddings_dir) / f"{event_id}.pkl"

            if not event_file.exists():
                return None
            with open(event_file, "rb") as f:
                return pickle.load(f)

        except (pickle.UnpicklingError, EOFError, OSError) as e:
            logger.error(f"[FAILED] Corrupted embedding file for {event_id}: {e}")
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

    def get_event_embedding(self, event_id: str) -> Optional[np.ndarray]:
        """Retrieve event embedding by event ID"""
        try:
            if not event_id.strip():
                raise InvalidEmbeddingDataError("Event ID cannot be empty")

            event_embedding = self._read_event_embedding_file(event_id)

            if not event_embedding:
                return None

            logger.info(f"[SUCCESS] Retrieved event embedding of {event_id}")
            return np.array(event_embedding.embedding)
        except InvalidEmbeddingDataError:
            raise
        except Exception as e:
            logger.error(f"[FAILED] Error retrieving event embedding for {event_id}")
            raise VectorStoreError(f"Failed to retrieve event embedding: {e}")

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
            return None

    def get_event_embedding_metadata(self, event_id: str) -> Optional[EventEmbedding]:
        """Get event embedding metadata for cache validation."""
        try:
            if not event_id.strip():
                raise InvalidEmbeddingDataError("Event ID cannot be empty")
            return self._read_event_embedding_file(event_id)
        except InvalidEmbeddingDataError:
            raise
        except Exception as e:
            logger.error(f"[FAILED] Error retrieving metadata for {event_id}: {e}")
            return None

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

    def should_regenerate_event_embedding(
        self,
        event_id: str,
        event_data: dict,
        current_text: Optional[str] = None,
        skip_content_check: bool = False,
    ) -> Tuple[bool, str]:
        """Check if event embedding needs regeneration."""
        try:
            stored_embedding = self.get_event_embedding_metadata(event_id)

            if not stored_embedding:
                return True, "No embedding exists"

            api_updated_current = event_data.get("updatedAt")

            api_updated_current_time = self._safe_get_updated_at(
                {"updatedAt": api_updated_current}
            )
            if api_updated_current_time:
                api_updated_stored_time = stored_embedding.api_updated_at
                if (
                    api_updated_stored_time
                    and api_updated_current_time > api_updated_stored_time
                ):
                    logger.info(
                        "[INFO] Need regeneration of embedding since the cms data has been updated"
                    )
                    return True, "Event collection timestamp changed"

            if current_text and not skip_content_check:
                current_text_hash = self._generate_text_hash(current_text)
                stored_text_hash = stored_embedding.text_hash
                if stored_text_hash != current_text_hash:
                    logger.info(
                        "[INFO] Content bases change detected, need to generate new embedding"
                    )
                    return True, "Content has changed"

            return False, "No changes detected"

        except Exception as e:
            logger.error(
                f"[FAILED] Error checking regeneration need for {event_id}: {e}"
            )
            return True, f"Error during check: {e}"

    def user_embeddings_exists(self, user_id: str) -> bool:
        """Checks if user embeddings exists for a given user id."""
        try:
            if not user_id.strip():
                return False
            user_file = Path(self.users_embeddings_dir) / f"{user_id}.pkl"
            return user_file.exists()

        except Exception as e:
            logger.error(
                f"[FAILED] Error checking embeddings existence for user with id {user_id}: {e}"
            )
            return False

    def event_embedding_exists(self, event_id: str) -> bool:
        """Check if event embedding exists."""
        try:
            if not event_id.strip():
                return False
            event_file = Path(self.events_embeddings_dir) / f"{event_id}.pkl"
            return event_file.exists()
        except Exception as e:
            logger.error(
                f"[FAILED] Error checking embedding existence for event with id {event_id}: {e}"
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

    def delete_event_embedding(self, event_id: str) -> bool:
        """Delete event embedding"""
        try:
            if not event_id.strip():
                raise InvalidEmbeddingDataError("Event ID cannot be empty")
            event_file = Path(self.events_embeddings_dir) / f"{event_id}.pkl"
            if event_file.exists():
                event_file.unlink()
                logger.info(f"[SUCCESS] Deleted event embedding for: {event_id}")
                return True
            return False
        except InvalidEmbeddingDataError:
            raise
        except Exception as e:
            logger.error(f"[FAILED] Error deleting event embedding for {event_id}: {e}")
            raise VectorStoreError(f"Failed to delete event embedding: {e}")

    def store_event_embeddings_batch(
        self,
        events_data: List[dict],
        embeddings: List[np.ndarray],
        events_texts: List[str],
    ) -> Dict[str, bool]:
        """Store multiple event embeddings at once."""
        try:
            if not (len(events_data) == len(embeddings) == len(events_texts)):
                raise InvalidEmbeddingDataError(
                    f"Length mismatch: events={len(events_data)}, "
                    f"embeddings={len(embeddings)}, texts={len(events_texts)}"
                )

            if not events_data:
                logger.warning(
                    "[WARNING] Empty batch provided to store_event_embeddings_batch"
                )
                return {}

            results = {}
            success_count = 0
            failure_count = 0

            for event, embedding, text in zip(events_data, embeddings, events_texts):
                event_id = event.get("id")
                if not event_id:
                    logger.warning("[WARNING] Event missing id field, skipping")
                    failure_count += 1
                    continue

                try:
                    success = self.store_event_embedding(
                        event_id=event_id,
                        embedding=embedding,
                        event_text=text,
                        api_updated_at=event.get("updatedAt"),
                    )

                    results[event_id] = success
                    if success:
                        success_count += 1
                    else:
                        failure_count += 1

                except Exception as e:
                    logger.error(f"[FAILED] Error storing event {event_id}: {e}")
                    results[event_id] = False
                    failure_count += 1

            logger.info(
                f"[SUCCESS] Batch store complete: {success_count} successful, "
                f"{failure_count} failed out of {len(events_data)} total"
            )

            return results

        except InvalidEmbeddingDataError:
            raise
        except Exception as e:
            logger.error(f"[FAILED] Batch store operation failed: {e}")
            raise VectorStoreError(f"Failed to store event embeddings batch: {e}")

    def get_stale_event_ids(
        self, events_data: List[dict], texts: Optional[List[str]] = None
    ) -> List[str]:
        """Return list of event IDs that need regeneration."""
        try:
            stale_ids = []
            if not texts:
                logger.info(
                    "[INFO] No texts provided, checking only timestamp-based staleness"
                )

            if texts and (len(events_data) != len(texts)):
                raise InvalidEmbeddingDataError(
                    f"Length mismatch: events={len(events_data)}, texts={len(texts)}"
                )

            for index, event_data in enumerate(events_data):
                event_id = event_data.get("id")
                if not event_id:
                    logger.warning("[WARNING] Event missing 'id' field, skipping")
                    continue

                current_text = texts[index] if texts else None

                should_regen, _ = self.should_regenerate_event_embedding(
                    event_id=event_id, event_data=event_data, current_text=current_text
                )

                if should_regen:
                    stale_ids.append(event_id)

            logger.info(
                f"[INFO] Found {len(stale_ids)} stale events that need regenration"
            )
            return stale_ids

        except InvalidEmbeddingDataError:
            raise

        except Exception as e:
            logger.error(f"[FAILED] Error getting stale event IDs: {e}")
            return []

    def get_all_event_embeddings(
        self, event_ids: List[str]
    ) -> Dict[str, Optional[np.ndarray]]:
        """Retrieve multiple event embeddings efficiently."""
        try:
            if not event_ids:
                logger.warning("[WARNING] Empty event_ids list provided")
                return {}
            results = {}
            found_count = 0
            missing_count = 0

            for event_id in event_ids:
                try:
                    embedding = self.get_event_embedding(event_id)
                    results[event_id] = embedding

                    if embedding is not None:
                        found_count += 1
                    else:
                        missing_count += 1
                except Exception as e:
                    logger.error(f"[FAILED] Error retrieving event {event_id}: {e}")
                    results[event_id] = None
                    missing_count += 1

            logger.info(
                f"[SUCCESS] Batch retrieve complete: {found_count} found, "
                f"{missing_count} missing out of {len(event_ids)} total"
            )

            return results

        except Exception as e:
            logger.error(f"[FAILED] Batch retrieve operation failed: {e}")
            raise VectorStoreError(f"Failed to retrieve event embeddings batch: {e}")
