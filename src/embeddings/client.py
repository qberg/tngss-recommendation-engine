"""OpenAI API client wrapper for embedding operations."""

import asyncio
from typing import List, Optional

import numpy as np
from openai import AsyncOpenAI

from src.config import settings
from src.embeddings.config import embedding_config
from src.utils.setup_logger import setup_logger

logger = setup_logger(__name__, "logs/embedding_client.log")


class OpenAIClient:
    """Wrapper for OpenAI API embedding operations."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = embedding_config.EMBEDDING_MODEL
        self.batch_size = embedding_config.EMBEDDING_BATCH_SIZE

        logger.info(
            f"[***] OpenAI embedding client initialized with model: {self.model}"
        )

    async def test_connection(self) -> bool:
        """Test OpenAI API connection"""
        try:
            logger.info("[...] Testing OpenAI Connection")

            response = await self.client.embeddings.create(
                model="text-embedding-3-small", input=["Test Connection"]
            )

            if response.data and len(response.data) > 0:
                embedding = response.data[0].embedding
                logger.info("[SUCCESS] OpenAI connection sucessfull")
                logger.info(f"[INFO] Embedding Dimensions: {len(embedding)}")

                return True
            else:
                logger.error("[FAILED] OpenAI API returned empty response")
                return False

        except Exception as e:
            logger.error(f"[FAILED] OpenAI connection test failed: {e}")
            raise e

    async def _create_embeddings_single(self, texts: List[str]) -> List[np.ndarray]:
        """Single API call for small batches"""
        response = await self.client.embeddings.create(model=self.model, input=texts)

        return [np.array(data.embedding) for data in response.data]

    async def _create_embeddings_batched(
        self, texts: List[str], batch_size: int
    ) -> List[np.ndarray]:
        """Create embeddings in multiple batches."""
        logger.info(f"[INFO] Processing {len(texts)} texts in batches of {batch_size}")

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            logger.info(f"[...] Processing batch {i//batch_size + 1}")

            batch_embeddings = await self._create_embeddings_single(batch)
            all_embeddings.extend(batch_embeddings)

            logger.info(f"[SUCCESS] Created {len(all_embeddings)} embeddings total")
        return all_embeddings

    async def create_embeddings(
        self, texts: List[str], batch_size: Optional[int] = None
    ):
        """Create embeddings with automatic batching for large inputs."""
        batch_size = batch_size or self.batch_size
        try:
            if isinstance(texts, str):
                texts = [texts]

            logger.info(f"[***] Creating embeddings for {len(texts)} texts")

            if len(texts) <= batch_size:
                return await self._create_embeddings_single(texts)
            else:
                return await self._create_embeddings_batched(texts, batch_size)

        except Exception as e:
            logger.error(f"[FAILED] Failed to create embeddings: {e}")
            raise e


async def main():
    """Test the OpenAI embedding client."""
    try:
        logger.info("[***] Starting OpenAI client test")

        client = OpenAIClient()

        connection_ok = await client.test_connection()

        if connection_ok:
            logger.info("[...] Testing embedding creation")
            test_texts = ["Hello world", "This is a test"]
            embeddings = await client.create_embeddings(test_texts)
            logger.info(
                f"[SUCCESS] Created embeddings with shape: {len(embeddings)}x{len(embeddings[0])}"
            )
            logger.info("[SUCCESS] All tests passed!")

        else:
            logger.error("[FAILED] Connection test failed")

    except Exception as e:
        logger.error(f"[FAILED] Client test failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
