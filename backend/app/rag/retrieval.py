"""Comparing embeddings by direction (T1.2).

Cosine similarity is implemented by hand rather than with numpy: the whole
point of the ticket is watching the formula do the work.

    cos(a, b) = dot(a, b) / (||a|| * ||b||)

Dividing by both magnitudes is what makes this measure *direction only*.
That matters because Ollama's vectors are not normalised -- see
docs/rag/embeddings.md -- so raw distance would confuse "different meaning"
with "different length of text".
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


class SimilarityError(ValueError):
    """Two vectors cannot meaningfully be compared."""


def magnitude(vector: Sequence[float]) -> float:
    """Euclidean length of ``vector``.

    math.fsum rather than sum(): 768 float additions accumulate rounding
    error, and fsum keeps the running total exact.
    """
    return math.sqrt(math.fsum(value * value for value in vector))


def dot_product(a: Sequence[float], b: Sequence[float]) -> float:
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Return how closely two vectors point the same way, in -1.0 .. 1.0.

    1.0 means identical direction, 0.0 unrelated, -1.0 opposite. In practice
    text embeddings rarely go negative, so treat the useful range as 0..1.
    """
    if len(a) != len(b):
        raise SimilarityError(
            f"Vectors have different dimensions ({len(a)} vs {len(b)}); "
            "they probably came from different embedding models"
        )
    if not a:
        raise SimilarityError("Cannot compare empty vectors")

    magnitude_a = magnitude(a)
    magnitude_b = magnitude(b)
    if magnitude_a == 0.0 or magnitude_b == 0.0:
        raise SimilarityError("Cannot compare a zero vector: it has no direction")

    return dot_product(a, b) / (magnitude_a * magnitude_b)


def rank_by_similarity(
    query_vector: Sequence[float],
    candidates: Iterable[tuple[str, Sequence[float]]],
) -> list[tuple[str, float]]:
    """Score every candidate against the query, best match first.

    ``candidates`` pairs a label with its vector. This is retrieval in
    miniature -- T1.9 replaces the in-memory loop with a pgvector query, but
    the ranking it performs is the same.
    """
    scored = [
        (label, cosine_similarity(query_vector, vector)) for label, vector in candidates
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored
