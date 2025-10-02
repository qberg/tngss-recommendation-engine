from datetime import datetime
from typing import Annotated, List, Optional

from bson import ObjectId
from pydantic import BaseModel, Field, field_validator

PyObjectId = Annotated[ObjectId, Field(json_schema_extra={"type": "string"})]


class LoginInfo(BaseModel):
    id: PyObjectId = Field(alias="_id")
    first_name: str
    last_name: Optional[str] = None
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
    gender: str
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

    @field_validator("looking_to_connect", "looking_to_meet", "sector", mode="before")
    @classmethod
    def extract_values(cls, v):
        if not v:
            return []

        if isinstance(v, list) and all(isinstance(item, str) for item in v):
            return v

        if isinstance(v, list) and all(isinstance(item, dict) for item in v):
            return [item.get("value", item.get("label", "")) for item in v]

        return v

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
