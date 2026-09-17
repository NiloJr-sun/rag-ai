"""Persisting documents, chunks and vectors in Postgres/pgvector (T1.8).

Talks to Postgres directly with psycopg rather than through the Supabase
client library: pgvector's distance operators are SQL, and PostgREST would
only get in the way of them.

Schema lives in backend/migrations/0001_init.sql and must be applied first.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

import psycopg

from app.rag.embeddings import EmbeddedChunk


class StorageError(RuntimeError):
    """The database could not be reached or a write did not apply."""


@dataclass(frozen=True)
class ChunkMatch:
    """A retrieved chunk, carrying everything needed to cite it (T2.11)."""

    chunk_id: str
    document_id: str
    chunk_index: int
    content: str
    source: str
    similarity: float


def connect(dsn: str | None = None) -> psycopg.Connection:
    """Open a connection, defaulting to ``DATABASE_URL``."""
    dsn = dsn or os.environ.get("DATABASE_URL")
    if not dsn:
        raise StorageError("DATABASE_URL is not set (see .env.example)")
    try:
        return psycopg.connect(dsn)
    except psycopg.Error as exc:
        raise StorageError(f"Could not connect to the database: {exc}") from exc


def upsert_document(
    conn: psycopg.Connection,
    *,
    document_id: str,
    source: str,
    content_hash: str,
) -> None:
    """Insert the document, or refresh it if this source was ingested before.

    Upsert rather than insert because document ids are derived from the
    source path, so re-ingesting a file must update rather than collide.
    """
    with conn.cursor() as cur:
        cur.execute(
            "insert into documents (id, source, content_hash) values (%s, %s, %s)"
            " on conflict (id) do update"
            " set source = excluded.source,"
            "     content_hash = excluded.content_hash,"
            "     ingested_at = now()",
            (document_id, source, content_hash),
        )


def replace_chunks(
    conn: psycopg.Connection,
    document_id: str,
    embedded: Sequence[EmbeddedChunk],
) -> int:
    """Replace every chunk of a document with the given ones.

    Delete-then-insert rather than upsert per row: if an edited document
    produces *fewer* chunks than before, the surplus rows must disappear or
    retrieval would keep returning stale text.
    """
    with conn.cursor() as cur:
        cur.execute(
            "delete from document_chunks where document_id = %s", (document_id,)
        )
        for item in embedded:
            cur.execute(
                "insert into document_chunks"
                " (id, document_id, chunk_index, content, embedding)"
                " values (%s, %s, %s, %s, %s)",
                (
                    item.chunk.chunk_id,
                    item.chunk.document_id,
                    item.chunk.chunk_index,
                    item.chunk.text,
                    str(item.embedding),
                ),
            )
    return len(embedded)


def count_chunks(conn: psycopg.Connection, document_id: str | None = None) -> int:
    with conn.cursor() as cur:
        if document_id is None:
            cur.execute("select count(*) from document_chunks")
        else:
            cur.execute(
                "select count(*) from document_chunks where document_id = %s",
                (document_id,),
            )
        row = cur.fetchone()
    return int(row[0]) if row else 0


def fetch_embedding(conn: psycopg.Connection, chunk_id: str) -> list[float] | None:
    """Read a stored vector back, for verifying a write actually landed."""
    with conn.cursor() as cur:
        cur.execute("select embedding from document_chunks where id = %s", (chunk_id,))
        row = cur.fetchone()
    if row is None or row[0] is None:
        return None
    # pgvector hands back its own text form, "[0.1,0.2,...]".
    return [float(part) for part in str(row[0]).strip("[]").split(",")]


def delete_document(conn: psycopg.Connection, document_id: str) -> None:
    """Remove a document; its chunks go with it via the cascade."""
    with conn.cursor() as cur:
        cur.execute("delete from documents where id = %s", (document_id,))


def reset(conn: psycopg.Connection) -> None:
    """Empty both tables. Destructive, and meant for development only."""
    with conn.cursor() as cur:
        cur.execute("truncate document_chunks, documents")


def search_chunks(
    conn: psycopg.Connection,
    embedding: Sequence[float],
    *,
    top_k: int,
) -> list[ChunkMatch]:
    """Return the ``top_k`` chunks closest to ``embedding``.

    Ordering by the `<=>` operator is what lets Postgres use the HNSW index;
    `<->` (L2) or `<#>` (inner product) would still return correct rows but
    fall back to a full scan, because the index was built for cosine.

    Similarity is reported as ``1 - distance`` so it matches
    app.rag.retrieval.cosine_similarity, where 1.0 is identical.
    """
    if top_k <= 0:
        raise StorageError(f"top_k must be positive, got {top_k}")

    vector = str(list(embedding))
    with conn.cursor() as cur:
        cur.execute(
            "select c.id, c.document_id, c.chunk_index, c.content, d.source,"
            "       1 - (c.embedding <=> %s) as similarity"
            " from document_chunks c"
            " join documents d on d.id = c.document_id"
            " where c.embedding is not null"
            " order by c.embedding <=> %s"
            " limit %s",
            (vector, vector, top_k),
        )
        rows = cur.fetchall()

    return [
        ChunkMatch(
            chunk_id=row[0],
            document_id=row[1],
            chunk_index=row[2],
            content=row[3],
            source=row[4],
            similarity=float(row[5]),
        )
        for row in rows
    ]
