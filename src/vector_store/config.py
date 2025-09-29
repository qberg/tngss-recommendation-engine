from pydantic_settings import BaseSettings


class VectorStoreConfig(BaseSettings):
    """Embedding service configuration"""

    EMBEDDINGS_DIR: str = ""
    USER_EMBEDDINGS_DIR: str = ""
    EVENTS_EMBEDDINGS_DIR: str = ""
    EMBEDDING_DIMENSION: int = 1536

    class Config:
        env_file = ".env"
        env_prefix = "FAISS_"
        extra = "ignore"


vector_store_config = VectorStoreConfig()
