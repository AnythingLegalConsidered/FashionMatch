"""Custom exceptions for the Vinted scraper module."""


class ScraperError(Exception):
    """Base exception for scraper-related errors."""
    pass


class ParsingError(ScraperError):
    """Failed to parse HTML/JSON content."""
    pass


class NavigationError(ScraperError):
    """Failed to navigate to a page."""
    pass


class DownloadError(ScraperError):
    """Failed to download an image."""
    pass
