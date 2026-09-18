"""Integration tests for the storage layer (T1.8).

These need a real Postgres with pgvector and the migration applied, so they
skip unless TEST_DATABASE_URL is set. CI has no database, and that is fine:
the logic worth unit-testing lives in chunking and embeddings.

TEST_DATABASE_URL deliberately, not DATABASE_URL: app.config loads the
repo-root .env, so DATABASE_URL is almost always present now and using it
here would silently point these tests at whatever database you actually
work against. They insert and delete rows.

    docker run -d --rm --name ragpg -e POSTGRES_PASSWORD=test \
      -p 55432:5432 pgvector/pgvector:pg16
    psql postgresql://postgres:test@127.0.0.1:55432/postgres \
      -f backend/migrations/0001_init.sql
    TEST_DATABASE_URL=postgresql://postgres:test@127.0.0.1:55432/postgres \
      pytest backend/tests
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import psycopg
import pytest

from app.rag.chunking import chunk_document
from app.rag.embeddings import EmbeddedChunk
from app.storage import supabase

DSN = os.environ.get("TEST_DATABASE_URL")
TEXT = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota. Kappa lambda mu."
SOURCE = "tests/fixture-document.txt"

pytestmark = pytest.mark.skipif(not DSN, reason="TEST_DATABASE_URL is not set")


def _embedded(source: str = SOURCE, max_size: int = 40) -> list[EmbeddedChunk]:
    """Chunks with stand-in vectors, so these tests need no Ollama.

    Deliberately never zero: a zero vector has no direction, so pgvector's
    cosine index cannot place it and search would not return it. Real
    embeddings are never zero -- see test_a_zero_vector_is_unsearchable.
    """
    return [
        EmbeddedChunk(chunk=chunk, embedding=[float(chunk.chunk_index + 1)] * 768)
        for chunk in chunk_document(TEXT, source=source, max_size=max_size)
    ]


@pytest.fixture
def conn() -> Iterator[psycopg.Connection]:
    connection = supabase.connect(DSN)
    try:
        yield connection
    finally:
        # Never truncate: DATABASE_URL may point at a real project.
        for source in (SOURCE, "tests/other-document.txt"):
            with connection.cursor() as cur:
                cur.execute("delete from documents where source = %s", (source,))
        connection.commit()
        connection.close()


def test_document_and_chunks_round_trip(conn: psycopg.Connection) -> None:
    embedded = _embedded()
    document_id = embedded[0].chunk.document_id

    supabase.upsert_document(
        conn, document_id=document_id, source=SOURCE, content_hash="abc123"
    )
    stored = supabase.replace_chunks(conn, document_id, embedded)
    conn.commit()

    assert stored == len(embedded)
    assert supabase.count_chunks(conn, document_id) == len(embedded)


def test_stored_vector_comes_back_unchanged(conn: psycopg.Connection) -> None:
    embedded = _embedded()
    document_id = embedded[0].chunk.document_id
    supabase.upsert_document(
        conn, document_id=document_id, source=SOURCE, content_hash="abc123"
    )
    supabase.replace_chunks(conn, document_id, embedded)
    conn.commit()

    first = embedded[0]
    read_back = supabase.fetch_embedding(conn, first.chunk.chunk_id)
    assert read_back is not None
    assert len(read_back) == 768
    assert read_back == pytest.approx(first.embedding)


def test_reingesting_replaces_rather_than_duplicates(conn: psycopg.Connection) -> None:
    """The point of deriving ids from the source path."""
    embedded = _embedded()
    document_id = embedded[0].chunk.document_id

    for _ in range(3):
        supabase.upsert_document(
            conn, document_id=document_id, source=SOURCE, content_hash="abc123"
        )
        supabase.replace_chunks(conn, document_id, embedded)
        conn.commit()

    assert supabase.count_chunks(conn, document_id) == len(embedded)


def test_a_shorter_document_drops_its_surplus_chunks(conn: psycopg.Connection) -> None:
    """Delete-then-insert, so stale chunks cannot survive an edit."""
    embedded = _embedded()
    document_id = embedded[0].chunk.document_id
    supabase.upsert_document(
        conn, document_id=document_id, source=SOURCE, content_hash="abc123"
    )
    supabase.replace_chunks(conn, document_id, embedded)
    conn.commit()

    supabase.replace_chunks(conn, document_id, embedded[:1])
    conn.commit()
    assert supabase.count_chunks(conn, document_id) == 1


def test_deleting_a_document_cascades_to_its_chunks(conn: psycopg.Connection) -> None:
    embedded = _embedded()
    document_id = embedded[0].chunk.document_id
    supabase.upsert_document(
        conn, document_id=document_id, source=SOURCE, content_hash="abc123"
    )
    supabase.replace_chunks(conn, document_id, embedded)
    conn.commit()

    supabase.delete_document(conn, document_id)
    conn.commit()
    assert supabase.count_chunks(conn, document_id) == 0


def test_missing_chunk_has_no_embedding(conn: psycopg.Connection) -> None:
    assert supabase.fetch_embedding(conn, "does-not-exist") is None


def test_connect_without_a_dsn_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    with pytest.raises(supabase.StorageError, match="DATABASE_URL"):
        supabase.connect()


def test_search_returns_closest_first(conn: psycopg.Connection) -> None:
    """Rows come back ordered by similarity, with citation fields populated."""
    embedded = _embedded()
    document_id = embedded[0].chunk.document_id
    supabase.upsert_document(
        conn, document_id=document_id, source=SOURCE, content_hash="abc123"
    )
    supabase.replace_chunks(conn, document_id, embedded)
    conn.commit()

    target = embedded[1].embedding
    matches = supabase.search_chunks(conn, target, top_k=50)

    # An exact-match vector must win outright, whatever else is in the table.
    assert matches[0].chunk_id == embedded[1].chunk.chunk_id
    assert matches[0].similarity == pytest.approx(1.0)
    assert matches[0].source == SOURCE
    assert matches[0].chunk_index == 1
    scores = [m.similarity for m in matches]
    assert scores == sorted(scores, reverse=True)


def test_search_honours_top_k(conn: psycopg.Connection) -> None:
    embedded = _embedded()
    document_id = embedded[0].chunk.document_id
    supabase.upsert_document(
        conn, document_id=document_id, source=SOURCE, content_hash="abc123"
    )
    supabase.replace_chunks(conn, document_id, embedded)
    conn.commit()

    assert len(supabase.search_chunks(conn, embedded[1].embedding, top_k=1)) == 1
    # top_k caps the result count; the table may hold rows beyond the fixture's.
    wide = supabase.search_chunks(conn, embedded[1].embedding, top_k=2)
    assert len(wide) == 2
    assert supabase.count_chunks(conn, document_id) == len(embedded)


def test_search_rejects_a_useless_top_k(conn: psycopg.Connection) -> None:
    with pytest.raises(supabase.StorageError, match="top_k must be positive"):
        supabase.search_chunks(conn, [0.0] * 768, top_k=0)


def test_a_zero_vector_is_unsearchable(conn: psycopg.Connection) -> None:
    """pgvector's cosine index cannot place a vector with no direction.

    Such a row is stored happily and then never returned by a search, at any
    top_k. app.rag.retrieval.cosine_similarity rejects zero vectors outright
    for the same reason.
    """
    embedded = _embedded()
    document_id = embedded[0].chunk.document_id
    zeroed = [
        EmbeddedChunk(chunk=embedded[0].chunk, embedding=[0.0] * 768),
        embedded[1],
    ]
    supabase.upsert_document(
        conn, document_id=document_id, source=SOURCE, content_hash="abc123"
    )
    supabase.replace_chunks(conn, document_id, zeroed)
    conn.commit()

    assert supabase.count_chunks(conn, document_id) == 2
    matches = supabase.search_chunks(conn, embedded[1].embedding, top_k=50)
    ours = [m.chunk_index for m in matches if m.document_id == document_id]
    assert ours == [1], "the zeroed chunk must be stored yet unreachable"
