"""Unit tests for structured chunks and batch embedding (T1.6, T1.7).

No database and no Ollama: chunking is pure, and embedding runs on a mock
transport.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from app.config import DEFAULT_TOP_K, ConfigError, top_k
from app.rag.chunking import (
    Chunk,
    chunk_document,
    chunk_id_for,
    content_hash,
    document_id_for,
)
from app.rag.embeddings import EmbeddingError, embed_chunks

TEXT = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota. Kappa lambda mu."
Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: Handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


# --- T1.6 -------------------------------------------------------------------


def test_chunks_carry_their_origin() -> None:
    chunks = chunk_document(TEXT, source="data/samples/sample.txt", max_size=40)
    assert chunks
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.source == "data/samples/sample.txt" for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_ids_are_stable_across_runs() -> None:
    """Re-ingesting the same file must reuse ids, or rows would duplicate."""
    first = chunk_document(TEXT, source="a.txt", max_size=40)
    second = chunk_document(TEXT, source="a.txt", max_size=40)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert first[0].document_id == second[0].document_id


def test_different_sources_get_different_ids() -> None:
    a = chunk_document(TEXT, source="a.txt", max_size=40)
    b = chunk_document(TEXT, source="b.txt", max_size=40)
    assert a[0].document_id != b[0].document_id


def test_chunk_ids_are_positional_not_content_derived() -> None:
    """Editing a document must overwrite chunk 3, not mint a new id for it."""
    original = chunk_document(TEXT, source="a.txt", max_size=40)
    edited = chunk_document(TEXT.replace("Alpha", "Omega"), source="a.txt", max_size=40)
    assert original[0].chunk_id == edited[0].chunk_id
    assert original[0].text != edited[0].text


def test_id_helpers() -> None:
    document_id = document_id_for("a.txt")
    assert len(document_id) == 16
    assert chunk_id_for(document_id, 7) == f"{document_id}-0007"


def test_content_hash_changes_with_content() -> None:
    assert content_hash("one") == content_hash("one")
    assert content_hash("one") != content_hash("two")


def test_empty_document_yields_no_chunks() -> None:
    assert chunk_document("   ", source="a.txt") == []


# --- T1.7 -------------------------------------------------------------------


def test_embed_chunks_attaches_vectors_and_times_the_run() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embedding": [0.1, 0.2, 0.3]})

    chunks = chunk_document(TEXT, source="a.txt", max_size=40)
    with _client(handler) as client:
        run = embed_chunks(chunks, client=client)

    assert len(run.embedded) == len(chunks)
    assert run.failed == []
    assert [item.chunk for item in run.embedded] == chunks
    assert all(item.embedding == [0.1, 0.2, 0.3] for item in run.embedded)
    assert run.seconds >= 0.0
    assert run.seconds_per_chunk >= 0.0


def test_embed_chunks_raises_on_failure_by_default() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    chunks = chunk_document(TEXT, source="a.txt", max_size=40)
    with _client(handler) as client, pytest.raises(EmbeddingError):
        embed_chunks(chunks, client=client)


def test_embed_chunks_can_skip_failures() -> None:
    """Opt-in partial ingestion: record what failed, keep the rest."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 2:
            return httpx.Response(500, text="boom")
        return httpx.Response(200, json={"embedding": [1.0]})

    chunks = chunk_document(TEXT, source="a.txt", max_size=40)
    with _client(handler) as client:
        run = embed_chunks(chunks, client=client, skip_failures=True)

    assert len(run.failed) == 1
    assert len(run.embedded) == len(chunks) - 1
    assert "boom" in run.failed[0][1]


# --- T1.9 -------------------------------------------------------------------


def test_top_k_defaults_and_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RETRIEVAL_TOP_K", raising=False)
    assert top_k() == DEFAULT_TOP_K

    monkeypatch.setenv("RETRIEVAL_TOP_K", "12")
    assert top_k() == 12


def test_a_nonsense_top_k_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIEVAL_TOP_K", "not-a-number")
    with pytest.raises(ConfigError, match="must be an integer"):
        top_k()

    monkeypatch.setenv("RETRIEVAL_TOP_K", "0")
    with pytest.raises(ConfigError, match="must be positive"):
        top_k()
