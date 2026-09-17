"""End-to-end ingestion: file -> chunks -> embeddings -> Supabase.

Ties together T1.6, T1.7 and T1.8. Reading the file happens here rather than
in chunking.py so the chunkers stay pure functions over strings -- the text
will arrive from an upload at T2.2 and from Google Drive at T5.3, neither of
which has a path on disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx
import psycopg

from app.rag.chunking import DEFAULT_CHUNK_SIZE, chunk_document, content_hash
from app.rag.embeddings import embed_chunks
from app.storage import supabase

SUPPORTED_SUFFIXES = {".txt", ".md"}


@dataclass
class IngestResult:
    source: str
    document_id: str
    chunks_stored: int
    chunks_failed: int
    seconds: float

    def __str__(self) -> str:
        failed = f", {self.chunks_failed} failed" if self.chunks_failed else ""
        return (
            f"{self.source}: {self.chunks_stored} chunks{failed} in {self.seconds:.2f}s"
        )


def ingest_file(
    path: Path,
    *,
    conn: psycopg.Connection,
    client: httpx.Client | None = None,
    max_size: int = DEFAULT_CHUNK_SIZE,
    skip_failures: bool = False,
) -> IngestResult:
    """Chunk, embed and store one text file."""
    text = path.read_text()
    source = str(path)

    chunks = chunk_document(text, source=source, max_size=max_size)
    run = embed_chunks(chunks, client=client, skip_failures=skip_failures)

    document_id = chunks[0].document_id if chunks else ""
    if chunks:
        supabase.upsert_document(
            conn,
            document_id=document_id,
            source=source,
            content_hash=content_hash(text),
        )
        supabase.replace_chunks(conn, document_id, run.embedded)
        # One transaction per document, so a failure part-way through a
        # directory leaves earlier documents intact and this one absent.
        conn.commit()

    return IngestResult(
        source=source,
        document_id=document_id,
        chunks_stored=len(run.embedded),
        chunks_failed=len(run.failed),
        seconds=run.seconds,
    )


def find_documents(target: Path) -> list[Path]:
    """Every supported file at ``target``, which may be a file or a directory."""
    if target.is_file():
        return [target] if target.suffix.lower() in SUPPORTED_SUFFIXES else []
    return sorted(
        p
        for p in target.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


def ingest_path(
    target: Path,
    *,
    conn: psycopg.Connection,
    client: httpx.Client | None = None,
    max_size: int = DEFAULT_CHUNK_SIZE,
    skip_failures: bool = False,
) -> list[IngestResult]:
    """Ingest every supported document at ``target``."""
    return [
        ingest_file(
            path,
            conn=conn,
            client=client,
            max_size=max_size,
            skip_failures=skip_failures,
        )
        for path in find_documents(target)
    ]
