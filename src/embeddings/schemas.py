from typing import Dict

from pydantic import BaseModel


class SimilarityResult(BaseModel):
    final_score: float
    personal: float
    org: float
    intent: float
    weights: Dict[str, float]


class EmbeddingResponse(BaseModel):
    embeddings: list
    count: int
    dimensions: int
