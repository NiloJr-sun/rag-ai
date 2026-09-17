"""T1.9 -- retrieve from the database, and see what k costs you.

export DATABASE_URL='postgresql://...'
python scripts/ingest.py data/samples
python scripts/search_demo.py
python scripts/search_demo.py "why split a document?"
"""

from __future__ import annotations

import sys

import httpx
from app.rag.embeddings import DEFAULT_TIMEOUT_SECONDS, embed_text
from app.storage import supabase

DEFAULT_QUESTIONS = [
    "How does cosine similarity work?",
    "Why should a document be split into smaller pieces?",
    "How do I cook pasta?",
]

K_VALUES = [1, 3, 5, 10]


def main() -> None:
    questions = sys.argv[1:] or DEFAULT_QUESTIONS

    with (
        supabase.connect() as conn,
        httpx.Client(timeout=DEFAULT_TIMEOUT_SECONDS) as client,
    ):
        stored = supabase.count_chunks(conn)
        print(f"{stored} chunks in the database\n")
        if not stored:
            raise SystemExit(
                "Nothing stored. Run: python scripts/ingest.py data/samples"
            )

        for question in questions:
            vector = embed_text(question, client=client)
            print(f"query: {question}")
            for k in K_VALUES:
                matches = supabase.search_chunks(conn, vector, top_k=k)
                scores = ", ".join(f"{m.similarity:.3f}" for m in matches)
                print(f"  k={k:<3} {len(matches)} chunk(s): {scores}")

            print("  top match:")
            for match in supabase.search_chunks(conn, vector, top_k=1):
                preview = " ".join(match.content.split())[:70]
                print(
                    f"    {match.similarity:.4f}  {match.source} #{match.chunk_index}"
                )
                print(f"    {preview}...")
            print()


if __name__ == "__main__":
    main()
