from pydantic_settings import BaseSettings


class EmbeddingConfig(BaseSettings):
    """Embedding service configuration"""

    EMBEDDING_MODEL: str = "text-embedding-3-small"
    MAX_TOKENS: int = 8000
    EMBEDDING_BATCH_SIZE: int = 100

    class Config:
        env_file = ".env"
        env_prefix = "EMBEDDING_"
        extra = "ignore"


embedding_config = EmbeddingConfig()
