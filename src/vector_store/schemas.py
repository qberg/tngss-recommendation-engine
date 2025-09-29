from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field


class UserEmbeddings(BaseModel):
    user_id: str
    personal: List[float]
    org: List[float]
    intent: List[float]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    login_updated_at: Optional[datetime] = None
    profile_updated_at: Optional[datetime] = None
    org_updated_at: Optional[datetime] = None
    context_updated_at: Optional[datetime] = None
    text_hash: str


class EventEmbedding(BaseModel):
    event_id: str
    embedding: List[float]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    api_updated_at: Optional[datetime] = None
    text_hash: str
