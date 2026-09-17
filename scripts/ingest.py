"""Ingest local documents into Supabase.

    export DATABASE_URL='postgresql://...'
    python scripts/ingest.py data/samples
    python scripts/ingest.py data/samples --reset

Requires the schema from backend/migrations/0001_init.sql and a running
Ollama with the embedding model pulled.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx
from app.pipeline import find_documents, ingest_file
from app.rag.chunking import DEFAULT_CHUNK_SIZE
from app.rag.embeddings import DEFAULT_TIMEOUT_SECONDS
from app.storage import supabase


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, help="file or directory to ingest")
    parser.add_argument("--max-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--reset", action="store_true", help="empty both tables before ingesting"
    )
    parser.add_argument(
        "--skip-failures",
        action="store_true",
        help="store what embedded successfully instead of aborting",
    )
    args = parser.parse_args()

    documents = find_documents(args.target)
    if not documents:
        raise SystemExit(f"No .txt or .md files found at {args.target}")

    with (
        supabase.connect() as conn,
        httpx.Client(timeout=DEFAULT_TIMEOUT_SECONDS) as client,
    ):
        if args.reset:
            supabase.reset(conn)
            conn.commit()
            print("reset: both tables emptied")

        total = 0
        for path in documents:
            result = ingest_file(
                path,
                conn=conn,
                client=client,
                max_size=args.max_size,
                skip_failures=args.skip_failures,
            )
            total += result.chunks_stored
            print(result)

        print(f"\n{len(documents)} documents, {total} chunks stored")
        print(f"chunks in database: {supabase.count_chunks(conn)}")


if __name__ == "__main__":
    main()
