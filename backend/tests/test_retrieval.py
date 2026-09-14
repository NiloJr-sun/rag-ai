"""Unit tests for cosine similarity.

Pure arithmetic, so these need neither Ollama nor a mock transport.
"""

from __future__ import annotations

import math

import pytest

from app.rag.retrieval import (
    SimilarityError,
    cosine_similarity,
    dot_product,
    magnitude,
    rank_by_similarity,
)


def test_identical_vectors_score_one() -> None:
    vector = [1.0, 2.0, 3.0]
    assert cosine_similarity(vector, vector) == pytest.approx(1.0)


def test_opposite_vectors_score_minus_one() -> None:
    assert cosine_similarity([1.0, 2.0], [-1.0, -2.0]) == pytest.approx(-1.0)


def test_perpendicular_vectors_score_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_magnitude_is_ignored() -> None:
    """The whole point: scaling a vector must not change its direction."""
    a = [1.0, 2.0, 3.0]
    scaled = [value * 17.0 for value in a]
    assert cosine_similarity(a, scaled) == pytest.approx(1.0)


def test_known_value() -> None:
    # cos of the angle between (1,0) and (1,1) is 1/sqrt(2)
    assert cosine_similarity([1.0, 0.0], [1.0, 1.0]) == pytest.approx(1 / math.sqrt(2))


def test_mismatched_dimensions_are_rejected() -> None:
    with pytest.raises(SimilarityError, match="different dimensions"):
        cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])


def test_zero_vector_is_rejected() -> None:
    with pytest.raises(SimilarityError, match="no direction"):
        cosine_similarity([0.0, 0.0], [1.0, 2.0])


def test_empty_vector_is_rejected() -> None:
    with pytest.raises(SimilarityError, match="empty"):
        cosine_similarity([], [])


def test_magnitude_and_dot_product() -> None:
    assert magnitude([3.0, 4.0]) == pytest.approx(5.0)
    assert dot_product([1.0, 2.0], [3.0, 4.0]) == pytest.approx(11.0)


def test_ranking_orders_best_first() -> None:
    query = [1.0, 0.0]
    ranked = rank_by_similarity(
        query,
        [
            ("perpendicular", [0.0, 1.0]),
            ("identical", [1.0, 0.0]),
            ("diagonal", [1.0, 1.0]),
        ],
    )
    assert [label for label, _ in ranked] == ["identical", "diagonal", "perpendicular"]
    assert ranked[0][1] == pytest.approx(1.0)
