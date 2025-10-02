from typing import Optional

from pydantic import BaseModel, Field


class RecommendationResponse(BaseModel):
    """Response format for stored recommendations"""

    user_id: str
    reference_id: str
    reference_type: str
    score: float = Field(..., ge=10, le=95)


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
