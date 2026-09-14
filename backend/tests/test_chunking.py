"""Unit tests for chunking. Pure string handling -- no Ollama needed."""

from __future__ import annotations

import pytest

from app.rag.chunking import (
    ChunkingError,
    chunk_by_sentence,
    chunk_fixed,
    split_sentences,
)


def test_short_text_is_one_chunk() -> None:
    assert chunk_fixed("hello", size=100) == ["hello"]


def test_empty_text_yields_no_chunks() -> None:
    assert chunk_fixed("   ") == []
    assert chunk_by_sentence("   ") == []


def test_fixed_chunks_respect_size() -> None:
    text = "abcdefghij" * 10  # 100 chars
    chunks = chunk_fixed(text, size=30, overlap=0)
    assert [len(c) for c in chunks] == [30, 30, 30, 10]
    assert "".join(chunks) == text


def test_overlap_repeats_the_boundary() -> None:
    text = "0123456789"
    chunks = chunk_fixed(text, size=5, overlap=2)
    assert chunks == ["01234", "34567", "6789"]
    # the tail of one chunk is the head of the next
    assert chunks[0][-2:] == chunks[1][:2]


def test_no_chunk_is_wholly_contained_in_its_predecessor() -> None:
    chunks = chunk_fixed("abcdefg", size=5, overlap=4)
    assert chunks[-1] not in chunks[-2]


def test_invalid_parameters_are_rejected() -> None:
    with pytest.raises(ChunkingError, match="size must be positive"):
        chunk_fixed("text", size=0)
    with pytest.raises(ChunkingError, match="overlap must be in"):
        chunk_fixed("text", size=10, overlap=10)
    with pytest.raises(ChunkingError, match="max_size must be positive"):
        chunk_by_sentence("text", max_size=0)


def test_split_sentences() -> None:
    assert split_sentences("One. Two! Three?  Four.") == [
        "One.",
        "Two!",
        "Three?",
        "Four.",
    ]


def test_sentence_chunks_never_split_a_sentence() -> None:
    text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota."
    for chunk in chunk_by_sentence(text, max_size=40):
        assert chunk.endswith((".", "!", "?"))


def test_sentence_chunks_pack_up_to_the_limit() -> None:
    text = "Aaa. Bbb. Ccc. Ddd."
    assert chunk_by_sentence(text, max_size=9) == ["Aaa. Bbb.", "Ccc. Ddd."]


def test_a_long_sentence_exceeds_max_size_rather_than_being_split() -> None:
    """The documented trade-off: whole ideas win over a guaranteed ceiling."""
    sentence = "word " * 50 + "end."
    chunks = chunk_by_sentence(sentence, max_size=20)
    assert len(chunks) == 1
    assert len(chunks[0]) > 20
