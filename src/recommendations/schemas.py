from typing import Dict, Optional

from pydantic import BaseModel, Field


class RecommendationResponse(BaseModel):
    """Response format for stored recommendations"""

    user_id: str
    reference_id: str
    reference_type: str
    score: float = Field(..., ge=10, le=95)


class MatchesResponse(BaseModel):
    """Response format for user matches"""

    user_id: str
    matched_user_id: str
    similarity_breakdown: Dict[str, float]
    score: float = Field(..., ge=10, le=95)


class UserUserMatchResponse(BaseModel):
    """Resposne format for user user matches"""

    user_id: str
    matched_user_id: str
    score: float = Field(..., ge=5, le=95)


class CalculateRecommendationsResponse(BaseModel):
    """Response for calculation endpoint"""

    success: bool
    message: str
    user_id: str
    scores_count: int
    calculated_at: str
    cache_hit: bool


class CalculateStatusResponse(BaseModel):
    """Response for calculation status endpoint."""

    success: bool
    message: str
    user_id: str
    task_id: Optional[str] = None
    cache_hit: bool
    last_calculated: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Calculation started in background",
                "user_id": "507f1f77bcf86cd799439011",
                "cache_hit": False,
            }
        }


class TaskStatusResponse(BaseModel):
    """Response for task status check."""

    task_id: str
    status: str
    message: Optional[str] = None
    progress: Optional[int] = None
    total: Optional[int] = None
    result: Optional[dict] = None
    error: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "abc-123-def",
                "status": "processing",
                "progress": 50,
                "total": 100,
                "message": "Generating embeddings...",
            }
        }
