"""
UserPreference entity for tracking user feedback and preferences.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class FeedbackType(Enum):
    """Types of user feedback."""

    LIKE = "like"
    DISLIKE = "dislike"
    SKIP = "skip"


@dataclass
class UserPreference:
    """
    Represents a user's preference/feedback on a clothing item.
    
    Used for learning and adjusting recommendation weights.
    """

    item_id: str
    feedback: FeedbackType
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Scores at the time of feedback (for weight optimization)
    clip_score: Optional[float] = None
    dino_score: Optional[float] = None
    hybrid_score: Optional[float] = None
    
    # Optional context
    session_id: Optional[str] = None
    reference_image_id: Optional[str] = None

    def is_positive(self) -> bool:
        """Check if feedback is positive."""
        return self.feedback == FeedbackType.LIKE

    def is_negative(self) -> bool:
        """Check if feedback is negative."""
        return self.feedback == FeedbackType.DISLIKE
