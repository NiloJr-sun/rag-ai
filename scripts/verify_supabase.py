"""T1.4 -- prove the schema stores and retrieves vectors correctly.

Applies nothing: run backend/migrations/0001_init.sql first, then

    export DATABASE_URL='postgresql://...'      # Supabase: Settings -> Database
    python scripts/verify_supabase.py

Ingests data/samples/sample.txt, asks pgvector a question, and checks that
the ranking matches the one app/rag/retrieval.py computes in Python. It
cleans up after itself, so it is safe to run against a real project.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import httpx
import psycopg
from app.rag.chunking import chunk_by_sentence
from app.rag.embeddings import embed_text
from app.rag.retrieval import rank_by_similarity

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "sample.txt"
QUESTION = "How does cosine similarity work?"
DOCUMENT_ID = "verify-sample"


def main() -> None:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise SystemExit("DATABASE_URL is not set (see .env.example)")

    text = SAMPLE.read_text()
    chunks = chunk_by_sentence(text, max_size=500)
    print(f"{SAMPLE.name}: {len(text)} chars -> {len(chunks)} chunks")

    with httpx.Client(timeout=120.0) as http:
        vectors = [embed_text(chunk, client=http) for chunk in chunks]
        question_vector = embed_text(QUESTION, client=http)

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        # Clean slate; the cascade removes any chunks from a previous run.
        cur.execute("delete from documents where id = %s", (DOCUMENT_ID,))
        cur.execute(
            "insert into documents (id, source, content_hash) values (%s, %s, %s)",
            (
                DOCUMENT_ID,
                str(SAMPLE),
                hashlib.sha256(text.encode()).hexdigest(),
            ),
        )
        for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
            cur.execute(
                "insert into document_chunks (id, document_id, chunk_index, content, embedding)"
                " values (%s, %s, %s, %s, %s)",
                (f"{DOCUMENT_ID}-{index}", DOCUMENT_ID, index, chunk, str(vector)),
            )
        conn.commit()

        cur.execute(
            "select count(*) from document_chunks where document_id = %s",
            (DOCUMENT_ID,),
        )
        row = cur.fetchone()
        print(f"stored {row[0] if row else 0} chunks\n")

        # <=> is cosine DISTANCE, so similarity is 1 - distance.
        cur.execute(
            "select content, 1 - (embedding <=> %s) as similarity"
            " from document_chunks where document_id = %s"
            " order by embedding <=> %s limit 3",
            (str(question_vector), DOCUMENT_ID, str(question_vector)),
        )
        print(f"pgvector, for {QUESTION!r}:")
        db_ranking = []
        for content, similarity in cur.fetchall():
            db_ranking.append(content)
            print(f"  {similarity:.4f}  {' '.join(content.split())[:60]}...")

        print("\nsame query ranked in Python:")
        python_ranking = []
        for content, score in rank_by_similarity(question_vector, zip(chunks, vectors))[
            :3
        ]:
            python_ranking.append(content)
            print(f"  {score:.4f}  {' '.join(content.split())[:60]}...")

        assert db_ranking == python_ranking, (
            "pgvector and Python disagree on the ranking"
        )
        print("\nOK: pgvector and app/rag/retrieval.py agree")

        cur.execute("delete from documents where id = %s", (DOCUMENT_ID,))
        conn.commit()
        cur.execute(
            "select count(*) from document_chunks where document_id = %s",
            (DOCUMENT_ID,),
        )
        row = cur.fetchone()
        print(
            f"cleaned up; chunks remaining: {row[0] if row else 0} (cascade delete works)"
        )


if __name__ == "__main__":
    main()
