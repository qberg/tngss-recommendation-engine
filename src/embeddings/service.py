"""Main embedding service for generating and processing embeddings."""

import asyncio
from typing import Dict, List, Optional

import numpy as np

from src.embeddings.client import OpenAIClient
from src.embeddings.schemas import SimilarityResult
from src.embeddings.utils import cosine_similarity, truncate_text
from src.utils.setup_logger import setup_logger

logger = setup_logger(__name__, "logs/embedding_service.log")


class EmbeddingService:
    """Main service for handling embedding operations."""

    def __init__(self):
        self.client = OpenAIClient()
        logger.info("[***] Embedding service initialized")

    async def test_connection(self) -> bool:
        """Test the embedding service connection."""
        return await self.client.test_connection()

    async def create_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Create embeddings for multiple texts."""
        processed_texts = [truncate_text(text) for text in texts]

        embeddings = await self.client.create_embeddings(processed_texts)

        logger.info(
            f"[SUCCESS][Pure EmbeddingService] Created {len(embeddings)} embeddings"
        )
        return embeddings

    def calculate_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """Calculate cosine similarity between two embeddings."""
        return cosine_similarity(embedding1, embedding2)

    def calculate_multi_vector_similarity(
        self,
        user_embeddings: Dict[str, np.ndarray],
        target_embedding: np.ndarray,
        weights: Optional[Dict[str, float]] = None,
    ) -> SimilarityResult:
        """Calculate weighted similarity between multi-vector user profile and target."""

        default_weights = {"personal": 0.25, "org": 0.25, "intent": 0.5}
        weights = weights or default_weights

        logger.info("[***] Calculating multi-vector similarity")

        similarities = {}
        for vector_type, embedding in user_embeddings.items():
            similarities[vector_type] = self.calculate_similarity(
                embedding, target_embedding
            )

        final_score = (
            weights["personal"] * similarities["personal"]
            + weights["org"] * similarities["org"]
            + weights["intent"] * similarities["intent"]
        )

        logger.info(f"[SUCCESS] Multi-vector similarity: {final_score:.3f}")

        return SimilarityResult(
            final_score=final_score,
            personal=similarities.get("personal", 0.0),
            org=similarities.get("org", 0.0),
            intent=similarities.get("intent", 0.0),
            weights={str(k): v for k, v in weights.items()},
        )


async def main():
    """Test the embedding service."""
    try:
        logger.info("[***] Testing embedding service")

        # Initialize service
        service = EmbeddingService()

        # Test connection
        if await service.test_connection():
            # Test embedding creation
            texts = ["Hello everyone", "Hello all", "Another example"]
            embeddings = await service.create_embeddings(texts)

            # Test similarity
            similarity = service.calculate_similarity(embeddings[0], embeddings[1])
            logger.info(
                f"[SUCCESS] Similarity between first two texts: {similarity:.3f}"
            )

            logger.info("[SUCCESS] All embedding service tests passed!")

    except Exception as e:
        logger.error(f"[FAILED] Embedding service test failed: {e}")


if __name__ == "__main__":

    asyncio.run(main())
