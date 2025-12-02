"""
Custom exception hierarchy for FashionMatch.

Provides a structured exception hierarchy for different error scenarios:
- AppException: Base for all application errors
- ConfigError: Configuration-related errors
- ScraperError: Web scraping errors
- EncoderError: AI model encoding errors
- DatabaseError: Storage/retrieval errors
- ValidationError: Input validation errors

Each exception includes:
- Descriptive message
- Optional error code for programmatic handling
- Optional context dictionary for debugging

Example:
    >>> from src.utils.exceptions import ScraperError
    >>> raise ScraperError("Failed to fetch page", code="SCRAPER_001", context={"url": url})
"""

from __future__ import annotations

from typing import Any, Dict, Optional


# ============================================
# Base Exception
# ============================================


class AppException(Exception):
    """
    Base exception for all FashionMatch application errors.
    
    All custom exceptions inherit from this class, allowing:
    - Catch-all handling of application errors
    - Consistent error structure across the app
    - Error code and context support
    
    Attributes:
        message: Human-readable error description.
        code: Optional error code for programmatic handling.
        context: Optional dictionary with debugging context.
        
    Example:
        >>> try:
        ...     raise AppException("Something went wrong", code="APP_001")
        ... except AppException as e:
        ...     print(f"Error {e.code}: {e.message}")
    """

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the exception.
        
        Args:
            message: Human-readable error description.
            code: Optional error code (e.g., "CONFIG_001").
            context: Optional dict with additional debugging info.
        """
        self.message = message
        self.code = code or self._default_code()
        self.context = context or {}
        super().__init__(self.message)

    def _default_code(self) -> str:
        """Generate default error code from class name."""
        # Convert CamelCase to UPPER_SNAKE_CASE
        name = self.__class__.__name__
        code = ""
        for i, char in enumerate(name):
            if char.isupper() and i > 0:
                code += "_"
            code += char.upper()
        return code

    def __str__(self) -> str:
        """String representation with code if available."""
        if self.code:
            return f"[{self.code}] {self.message}"
        return self.message

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"code={self.code!r}, "
            f"context={self.context!r})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "code": self.code,
            "context": self.context,
        }


# ============================================
# Configuration Errors
# ============================================


class ConfigError(AppException):
    """
    Base exception for configuration-related errors.
    
    Raised when there are issues with:
    - Loading configuration files
    - Parsing YAML/JSON
    - Validating configuration values
    """

    pass


class ConfigFileNotFoundError(ConfigError):
    """
    Raised when a required configuration file is not found.
    
    Example:
        >>> raise ConfigFileNotFoundError(
        ...     "Configuration file not found",
        ...     context={"path": "/path/to/config.yaml"}
        ... )
    """

    def __init__(
        self,
        message: str = "Configuration file not found",
        path: Optional[str] = None,
        **kwargs,
    ) -> None:
        context = kwargs.pop("context", {})
        if path:
            context["path"] = path
        super().__init__(message, code="CONFIG_FILE_NOT_FOUND", context=context, **kwargs)


class ConfigurationError(ConfigError):
    """
    Raised when configuration is invalid or cannot be parsed.
    
    Example:
        >>> raise ConfigurationError(
        ...     "Invalid fusion weights: must sum to 1",
        ...     context={"weights": {"clip": 0.6, "dino": 0.6}}
        ... )
    """

    def __init__(
        self,
        message: str = "Invalid configuration",
        **kwargs,
    ) -> None:
        super().__init__(message, code="CONFIG_INVALID", **kwargs)


class ConfigValidationError(ConfigError):
    """
    Raised when configuration values fail validation.
    
    Example:
        >>> raise ConfigValidationError(
        ...     "embedding_dim must be positive",
        ...     field="models.clip.embedding_dim",
        ...     value=-1
        ... )
    """

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Any = None,
        **kwargs,
    ) -> None:
        context = kwargs.pop("context", {})
        if field:
            context["field"] = field
        if value is not None:
            context["value"] = value
        super().__init__(message, code="CONFIG_VALIDATION", context=context, **kwargs)


# ============================================
# Scraper Errors
# ============================================


class ScraperError(AppException):
    """
    Base exception for web scraping errors.
    
    Raised when there are issues with:
    - HTTP requests
    - Rate limiting
    - Page parsing
    - Image downloading
    """

    pass


class NetworkError(ScraperError):
    """
    Raised when a network request fails.
    
    Example:
        >>> raise NetworkError(
        ...     "Connection timeout",
        ...     url="https://vinted.fr/items/123",
        ...     status_code=None
        ... )
    """

    def __init__(
        self,
        message: str = "Network request failed",
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs,
    ) -> None:
        context = kwargs.pop("context", {})
        if url:
            context["url"] = url
        if status_code:
            context["status_code"] = status_code
        super().__init__(message, code="NETWORK_ERROR", context=context, **kwargs)


class RateLimitError(ScraperError):
    """
    Raised when rate limit is exceeded.
    
    Example:
        >>> raise RateLimitError(
        ...     "Too many requests",
        ...     retry_after=60
        ... )
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        **kwargs,
    ) -> None:
        context = kwargs.pop("context", {})
        if retry_after:
            context["retry_after_seconds"] = retry_after
        super().__init__(message, code="RATE_LIMIT", context=context, **kwargs)


class PageParsingError(ScraperError):
    """
    Raised when page content cannot be parsed.
    
    Example:
        >>> raise PageParsingError(
        ...     "Failed to extract price",
        ...     selector=".item-price",
        ...     url="https://vinted.fr/items/123"
        ... )
    """

    def __init__(
        self,
        message: str = "Failed to parse page content",
        url: Optional[str] = None,
        selector: Optional[str] = None,
        **kwargs,
    ) -> None:
        context = kwargs.pop("context", {})
        if url:
            context["url"] = url
        if selector:
            context["selector"] = selector
        super().__init__(message, code="PAGE_PARSE", context=context, **kwargs)


class ImageDownloadError(ScraperError):
    """Raised when image download fails."""

    def __init__(
        self,
        message: str = "Failed to download image",
        image_url: Optional[str] = None,
        **kwargs,
    ) -> None:
        context = kwargs.pop("context", {})
        if image_url:
            context["image_url"] = image_url
        super().__init__(message, code="IMAGE_DOWNLOAD", context=context, **kwargs)


# ============================================
# Encoder Errors
# ============================================


class EncoderError(AppException):
    """
    Base exception for AI model encoding errors.
    
    Raised when there are issues with:
    - Loading models
    - Processing images
    - Generating embeddings
    """

    pass


class ModelNotLoadedError(EncoderError):
    """
    Raised when trying to use an encoder before the model is loaded.
    
    Example:
        >>> raise ModelNotLoadedError(
        ...     "CLIP model not loaded",
        ...     model_name="ViT-B/32"
        ... )
    """

    def __init__(
        self,
        message: str = "Model not loaded",
        model_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        context = kwargs.pop("context", {})
        if model_name:
            context["model_name"] = model_name
        super().__init__(message, code="MODEL_NOT_LOADED", context=context, **kwargs)


class ModelLoadError(EncoderError):
    """Raised when model loading fails."""

    def __init__(
        self,
        message: str = "Failed to load model",
        model_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        context = kwargs.pop("context", {})
        if model_name:
            context["model_name"] = model_name
        super().__init__(message, code="MODEL_LOAD", context=context, **kwargs)


class ImageProcessingError(EncoderError):
    """
    Raised when image preprocessing fails.
    
    Example:
        >>> raise ImageProcessingError(
        ...     "Invalid image format",
        ...     image_path="/path/to/image.xyz"
        ... )
    """

    def __init__(
        self,
        message: str = "Failed to process image",
        image_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        context = kwargs.pop("context", {})
        if image_path:
            context["image_path"] = image_path
        super().__init__(message, code="IMAGE_PROCESS", context=context, **kwargs)


class EmbeddingError(EncoderError):
    """Raised when embedding generation fails."""

    def __init__(
        self,
        message: str = "Failed to generate embedding",
        **kwargs,
    ) -> None:
        super().__init__(message, code="EMBEDDING_ERROR", **kwargs)


class EncodingError(EncoderError):
    """
    Raised when the encoding process fails.
    
    Example:
        >>> raise EncodingError(
        ...     "Failed to encode image with CLIP",
        ...     model_name="ViT-B/32"
        ... )
    """

    def __init__(
        self,
        message: str = "Failed to encode",
        model_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        context = kwargs.pop("context", {})
        if model_name:
            context["model_name"] = model_name
        super().__init__(message, code="ENCODING_ERROR", context=context, **kwargs)


# ============================================
# Database Errors
# ============================================


class DatabaseError(AppException):
    """
    Base exception for database/storage errors.
    
    Raised when there are issues with:
    - ChromaDB operations
    - Vector storage/retrieval
    - Collection management
    """

    pass


class ItemNotFoundError(DatabaseError):
    """
    Raised when an item is not found in the database.
    
    Example:
        >>> raise ItemNotFoundError(
        ...     "Item not found",
        ...     item_id="abc-123"
        ... )
    """

    def __init__(
        self,
        message: str = "Item not found",
        item_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        context = kwargs.pop("context", {})
        if item_id:
            context["item_id"] = item_id
        super().__init__(message, code="ITEM_NOT_FOUND", context=context, **kwargs)


class EmbeddingMismatchError(DatabaseError):
    """
    Raised when embedding dimensions don't match expected values.
    
    Example:
        >>> raise EmbeddingMismatchError(
        ...     "CLIP embedding dimension mismatch",
        ...     expected=512,
        ...     actual=768
        ... )
    """

    def __init__(
        self,
        message: str = "Embedding dimension mismatch",
        expected: Optional[int] = None,
        actual: Optional[int] = None,
        **kwargs,
    ) -> None:
        context = kwargs.pop("context", {})
        if expected:
            context["expected_dim"] = expected
        if actual:
            context["actual_dim"] = actual
        super().__init__(message, code="EMBEDDING_MISMATCH", context=context, **kwargs)


class CollectionError(DatabaseError):
    """Raised when collection operations fail."""

    def __init__(
        self,
        message: str = "Collection operation failed",
        collection_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        context = kwargs.pop("context", {})
        if collection_name:
            context["collection_name"] = collection_name
        super().__init__(message, code="COLLECTION_ERROR", context=context, **kwargs)


# ============================================
# Validation Errors
# ============================================


class ValidationError(AppException):
    """
    Base exception for input validation errors.
    
    Raised when user input or data fails validation.
    """

    pass


class InvalidInputError(ValidationError):
    """Raised when input data is invalid."""

    def __init__(
        self,
        message: str = "Invalid input",
        field: Optional[str] = None,
        value: Any = None,
        **kwargs,
    ) -> None:
        context = kwargs.pop("context", {})
        if field:
            context["field"] = field
        if value is not None:
            context["value"] = str(value)[:100]  # Limit value length
        super().__init__(message, code="INVALID_INPUT", context=context, **kwargs)


class InvalidURLError(ValidationError):
    """Raised when URL is invalid."""

    def __init__(
        self,
        message: str = "Invalid URL",
        url: Optional[str] = None,
        **kwargs,
    ) -> None:
        context = kwargs.pop("context", {})
        if url:
            context["url"] = url
        super().__init__(message, code="INVALID_URL", context=context, **kwargs)


class InvalidImageError(ValidationError):
    """Raised when image file is invalid."""

    def __init__(
        self,
        message: str = "Invalid image file",
        path: Optional[str] = None,
        reason: Optional[str] = None,
        **kwargs,
    ) -> None:
        context = kwargs.pop("context", {})
        if path:
            context["path"] = path
        if reason:
            context["reason"] = reason
        super().__init__(message, code="INVALID_IMAGE", context=context, **kwargs)


# ============================================
# UI Errors
# ============================================


class UIError(AppException):
    """Base exception for UI-related errors."""

    pass


class SessionError(UIError):
    """Raised when session state is invalid."""

    def __init__(
        self,
        message: str = "Session error",
        **kwargs,
    ) -> None:
        super().__init__(message, code="SESSION_ERROR", **kwargs)


# ============================================
# Convenience Aliases (for backward compatibility)
# ============================================

# Alias for common import pattern
FashionMatchError = AppException
