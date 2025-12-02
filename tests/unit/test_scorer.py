"""
Unit tests for scoring functions.
"""

import pytest

from src.core.scoring.weighted_scorer import FusionWeights, WeightedScorer


class TestFusionWeights:
    """Tests for FusionWeights dataclass."""

    def test_default_weights(self):
        """Test default weights are equal."""
        weights = FusionWeights()
        assert weights.clip == 0.5
        assert weights.dino == 0.5

    def test_custom_weights(self):
        """Test custom weights."""
        weights = FusionWeights(clip=0.7, dino=0.3)
        assert weights.clip == 0.7
        assert weights.dino == 0.3

    def test_weights_must_sum_to_one(self):
        """Test that weights must sum to 1."""
        with pytest.raises(ValueError, match="must sum to 1"):
            FusionWeights(clip=0.6, dino=0.6)

    def test_weights_in_valid_range(self):
        """Test weights must be in [0, 1]."""
        with pytest.raises(ValueError):
            FusionWeights(clip=1.5, dino=-0.5)


class TestWeightedScorer:
    """Tests for WeightedScorer class."""

    def test_compute_score_equal_weights(self):
        """Test score computation with equal weights."""
        scorer = WeightedScorer()
        score = scorer.compute_score(clip_sim=0.8, dino_sim=0.6)
        assert score == pytest.approx(0.7)

    def test_compute_score_custom_weights(self):
        """Test score computation with custom weights."""
        weights = FusionWeights(clip=0.8, dino=0.2)
        scorer = WeightedScorer(weights)
        score = scorer.compute_score(clip_sim=1.0, dino_sim=0.0)
        assert score == pytest.approx(0.8)

    def test_update_weights(self):
        """Test weight update."""
        scorer = WeightedScorer()
        new_weights = FusionWeights(clip=0.3, dino=0.7)
        scorer.update_weights(new_weights)
        assert scorer.weights.clip == 0.3
        assert scorer.weights.dino == 0.7

    def test_get_weights(self):
        """Test getting current weights."""
        weights = FusionWeights(clip=0.6, dino=0.4)
        scorer = WeightedScorer(weights)
        retrieved = scorer.get_weights()
        assert retrieved.clip == 0.6
        assert retrieved.dino == 0.4
