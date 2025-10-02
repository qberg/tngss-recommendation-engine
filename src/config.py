import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MONGODB_URL: str = "mongodb://localhost:27017"
    DATABASE_NAME: str = "tngss_database"

    LOGIN_COLLECTION: str = "fake_login_info"
    USERS_PROFILE_COLLECTION: str = "fake_profile_management"
    ORGANISATION_PROFILE_COLLECTION: str = "fake_organisation_profile_management"
    CONTEXT_BUILDER_COLLECTION: str = "fake_context_builder_management"
    EVENTS_COLLECTION: str = "events"
    RECOMMENDATIONS_COLLECTION: str = "fake_ai_score"

    PAYLOAD_CMS_URL: str = "https://cms.tngss.startuptn.in/api"

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    DEFAULT_RECOMMENDATION_LIMIT: int = 10
    MIN_SCORE_THRESHOLD: float = 0.1

    OPENAI_API_KEY: str = ""

    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    class Config:
        env_file = (
            ".env.development" if os.getenv("ENVIRONMENT") == "development" else ".env"
        )
        case_sensitive = True
        extra = "ignore"


settings = Settings()
