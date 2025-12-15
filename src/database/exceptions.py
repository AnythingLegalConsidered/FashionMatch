"""Custom exceptions for vector store operations."""


class VectorStoreError(Exception):
    """Base exception for vector store operations."""
    pass


class EmbeddingDimensionError(VectorStoreError):
    """Raised when embedding dimensions don't match expected dimensions."""
    
    def __init__(self, message: str, expected_dim: int = None, actual_dim: int = None):
        self.expected_dim = expected_dim
        self.actual_dim = actual_dim
        super().__init__(message)


class ItemNotFoundError(VectorStoreError):
    """Raised when item ID doesn't exist in the vector store."""
    
    def __init__(self, message: str, item_id: str = None):
        self.item_id = item_id
        super().__init__(message)


class CollectionError(VectorStoreError):
    """Raised for collection creation or access issues."""
    pass


class BatchInsertError(VectorStoreError):
    """Raised for batch operation failures."""
    
    def __init__(self, message: str, failed_ids: list[str] = None):
        self.failed_ids = failed_ids or []
        super().__init__(message)
