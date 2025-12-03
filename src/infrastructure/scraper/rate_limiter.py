"""
Rate limiter for respectful scraping.

Provides configurable rate limiting with random delays
and exponential backoff for retry scenarios.

Example:
    >>> limiter = RateLimiter(min_delay=1.0, max_delay=3.0)
    >>> await limiter.wait()  # Waits 1-3 seconds randomly
    >>> await limiter.wait()  # Waits again before next request
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RateLimiterConfig:
    """
    Configuration for the rate limiter.
    
    Attributes:
        min_delay: Minimum delay between requests in seconds.
        max_delay: Maximum delay between requests in seconds.
        backoff_factor: Multiplier for exponential backoff on errors.
        max_backoff: Maximum backoff delay in seconds.
    """
    min_delay: float = 1.0
    max_delay: float = 3.0
    backoff_factor: float = 2.0
    max_backoff: float = 60.0


class RateLimiter:
    """
    Basic rate limiter with random delays.
    
    Enforces a minimum time between requests with random jitter
    to avoid detection patterns.
    
    Attributes:
        config: Rate limiter configuration.
        last_request_time: Timestamp of the last request.
        
    Example:
        >>> limiter = RateLimiter(min_delay=1.0, max_delay=3.0)
        >>> await limiter.wait()
        >>> # Make request
        >>> await limiter.wait()
        >>> # Make next request
    """
    
    def __init__(
        self,
        min_delay: float = 1.0,
        max_delay: float = 3.0,
        config: Optional[RateLimiterConfig] = None,
    ):
        """
        Initialize the rate limiter.
        
        Args:
            min_delay: Minimum delay in seconds.
            max_delay: Maximum delay in seconds.
            config: Full configuration object (overrides min/max_delay).
        """
        if config:
            self.config = config
        else:
            self.config = RateLimiterConfig(
                min_delay=min_delay,
                max_delay=max_delay,
            )
        
        self.last_request_time: float = 0.0
        logger.debug(
            f"RateLimiter initialized: {self.config.min_delay}-{self.config.max_delay}s delay"
        )
    
    def _get_random_delay(self) -> float:
        """Generate a random delay between min and max."""
        return random.uniform(self.config.min_delay, self.config.max_delay)
    
    async def wait(self) -> float:
        """
        Wait before making the next request.
        
        Returns:
            The actual delay in seconds.
        """
        delay = self._get_random_delay()
        
        # Calculate time since last request
        elapsed = time.time() - self.last_request_time
        
        # Only wait if not enough time has passed
        if elapsed < delay:
            actual_delay = delay - elapsed
            logger.debug(f"Rate limiting: waiting {actual_delay:.2f}s")
            await asyncio.sleep(actual_delay)
        else:
            actual_delay = 0.0
        
        self.last_request_time = time.time()
        return actual_delay
    
    def reset(self) -> None:
        """Reset the last request time."""
        self.last_request_time = 0.0


class AdaptiveRateLimiter(RateLimiter):
    """
    Rate limiter with exponential backoff on errors.
    
    Automatically increases delays when errors occur and
    gradually reduces them on success.
    
    Attributes:
        current_backoff: Current backoff multiplier.
        consecutive_errors: Number of consecutive errors.
        
    Example:
        >>> limiter = AdaptiveRateLimiter()
        >>> await limiter.wait()
        >>> try:
        ...     response = await make_request()
        ...     limiter.on_success()
        ... except Exception:
        ...     limiter.on_error()
    """
    
    def __init__(
        self,
        min_delay: float = 1.0,
        max_delay: float = 3.0,
        backoff_factor: float = 2.0,
        max_backoff: float = 60.0,
        config: Optional[RateLimiterConfig] = None,
    ):
        """
        Initialize the adaptive rate limiter.
        
        Args:
            min_delay: Minimum delay in seconds.
            max_delay: Maximum delay in seconds.
            backoff_factor: Multiplier for exponential backoff.
            max_backoff: Maximum backoff delay.
            config: Full configuration (overrides other params).
        """
        if config:
            super().__init__(config=config)
        else:
            super().__init__(
                config=RateLimiterConfig(
                    min_delay=min_delay,
                    max_delay=max_delay,
                    backoff_factor=backoff_factor,
                    max_backoff=max_backoff,
                )
            )
        
        self.current_backoff: float = 1.0
        self.consecutive_errors: int = 0
    
    def _get_random_delay(self) -> float:
        """Generate delay with current backoff applied."""
        base_delay = super()._get_random_delay()
        return min(
            base_delay * self.current_backoff,
            self.config.max_backoff,
        )
    
    def on_success(self) -> None:
        """
        Call after a successful request.
        
        Gradually reduces the backoff multiplier.
        """
        self.consecutive_errors = 0
        # Gradually reduce backoff
        if self.current_backoff > 1.0:
            self.current_backoff = max(1.0, self.current_backoff / 2)
            logger.debug(f"Backoff reduced to {self.current_backoff:.2f}x")
    
    def on_error(self) -> None:
        """
        Call after a failed request.
        
        Increases the backoff multiplier exponentially.
        """
        self.consecutive_errors += 1
        self.current_backoff = min(
            self.current_backoff * self.config.backoff_factor,
            self.config.max_backoff / self.config.max_delay,
        )
        logger.warning(
            f"Error #{self.consecutive_errors}, backoff increased to {self.current_backoff:.2f}x"
        )
    
    def reset(self) -> None:
        """Reset the limiter to initial state."""
        super().reset()
        self.current_backoff = 1.0
        self.consecutive_errors = 0


def create_rate_limiter_from_config() -> AdaptiveRateLimiter:
    """
    Create a rate limiter from the application config.
    
    Returns:
        AdaptiveRateLimiter configured from config.yaml.
    """
    from src.utils.config import get_settings
    
    settings = get_settings()
    
    return AdaptiveRateLimiter(
        min_delay=settings.scraper.request_delay.min_seconds,
        max_delay=settings.scraper.request_delay.max_seconds,
        backoff_factor=2.0,
        max_backoff=60.0,
    )
