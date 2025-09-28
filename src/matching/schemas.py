from datetime import datetime
from typing import Annotated, List, Literal, Optional

from bson import ObjectId
from pydantic import BaseModel, Field


class TriggerEventRecommendationsRequest(BaseModel):
    """Request to trigger recommendation calculation for a new event"""

    event_id: str = Field(..., description="Event ID to calculate recommendations for")


class TriggerUserRecommendationsRequest(BaseModel):
    """Request to trigger recommendation calculation for a new user"""

    user_id: str = Field(..., description="User ID to calculate recommendations for")


class TriggerRecommendationsResponse(BaseModel):
    """Response for triggered recommendation calculation"""

    job_id: str = Field(..., description="Unique job identifier for tracking")
    trigger_type: Literal["event", "user"] = Field(..., description="Type of trigger")
    target_id: str = Field(..., description="Event ID or User ID")
    message: str = Field(..., description="Status message")
    estimated_processing_time: str = Field(
        ..., description="Estimated time to complete"
    )
    total_calculations: int = Field(
        ..., description="Number of calculations to perform"
    )


class UserEventRecommendation(BaseModel):
    """Stored recommendation document in MongoDB"""

    user_id: str = Field(..., description="User ID")
    event_id: str = Field(..., description="Event ID")
    score: float = Field(..., ge=0.0, le=1.0, description="Recommendation score")
    generated_at: datetime = Field(
        default_factory=datetime.now, description="When calculated"
    )


class PaginationInfo(BaseModel):
    """Pagination metadata"""

    current_page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Items per page")
    total_records: int = Field(..., description="Total records matching criteria")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether next page exists")
    has_previous: bool = Field(..., description="Whether previuos page exists")


#######################
# DB Models
#######################

PyObjectId = Annotated[ObjectId, Field(json_schema_extra={"type": "string"})]


class LoginInfo(BaseModel):
    id: PyObjectId = Field(alias="_id")
    first_name: str
    last_name: str
    email_id: str
    phone_number: str
    organisation_profile_id: PyObjectId
    status: str
    is_deleted: bool = False
    createdAt: datetime
    updatedAt: datetime

    class Config:
        validate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class UserProfile(BaseModel):
    id: PyObjectId = Field(alias="_id")
    user_id: PyObjectId
    designation: str
    bio: str
    city: str
    country: str
    linkedin_url: str
    focused_sector: str
    organization_name: str
    is_deleted: bool = False

    class Config:
        validate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class OrganisationProfile(BaseModel):
    id: PyObjectId = Field(alias="_id")
    user_id: PyObjectId
    profile_type: str
    sub_type: str
    organisation_name: str
    sector: str
    city: str
    country: str
    about_organisation: str
    product_offering_details: List[str] = []
    looking_for: List[str] = []
    bussiness_model: List[str] = []
    is_deleted: bool = False

    # Startup specific fields
    stage: Optional[str] = None
    team_size: Optional[int] = None
    fund_rise_till_date: Optional[str] = None
    revenue: Optional[str] = None

    # Investor specific fields
    funding_stage: Optional[str] = None
    investment_instrument: Optional[str] = None
    revenue_stage: Optional[str] = None
    ticket_size: Optional[str] = None

    class Config:
        validate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class ContextBuilder(BaseModel):
    id: PyObjectId = Field(alias="_id")
    user_id: PyObjectId
    looking_to_connect: List[str] = []
    looking_to_meet: List[str] = []
    sector: List[str] = []
    is_deleted: bool = False

    class Config:
        validate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class Event(BaseModel):
    id: PyObjectId = Field(alias="_id")
    title: str
    about: str
    tags: List[str] = []
    speakers: List[str] = []
    format: PyObjectId
    isPublic: bool = True

    class Config:
        validate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class UserData(BaseModel):
    login: Optional[LoginInfo]
    profile: Optional[UserProfile]
    org: Optional[OrganisationProfile]
    context: Optional[ContextBuilder]

    class Config:
        arbitrary_types_allowed = True
