class VectorStoreError(Exception):
    """Base exception for vector store operations."""

    pass


class EmbeddingNotFoundError(VectorStoreError):
    """Raised when requested embedding is not found."""

    pass


class InvalidEmbeddingDataError(VectorStoreError):
    """Raised when embedding data is invalid or malformed."""

    pass


class StorageError(VectorStoreError):
    """Raised when storage operations fail."""

    pass


class ValidationError(VectorStoreError):
    """Raised when data validation fails."""

    pass
