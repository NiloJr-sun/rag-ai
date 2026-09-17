"""T1.11 -- ask questions about the ingested documents.

    export DATABASE_URL='postgresql://...'
    python scripts/ingest.py data/samples
    python scripts/ask.py "How does cosine similarity work?"
    python scripts/ask.py                 # interactive; Ctrl-D to quit

Needs the schema from backend/migrations/0001_init.sql, a running Ollama
with both models pulled, and documents already ingested.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator

import httpx
import psycopg
from app.pipeline import answer_question
from app.rag.embeddings import EmbeddingError
from app.rag.generation import DEFAULT_TIMEOUT_SECONDS, GenerationError
from app.rag.retrieval import SimilarityError
from app.storage import supabase

# Everything a bad question can raise: a wrong model, an unreachable Ollama,
# a nonsensical --top-k. None of them should end an interactive session.
ASK_ERRORS = (
    EmbeddingError,
    GenerationError,
    SimilarityError,
    supabase.StorageError,
)


def ask(
    question: str,
    *,
    conn: psycopg.Connection,
    client: httpx.Client,
    top_k: int | None,
    show_chunks: bool,
) -> None:
    answer = answer_question(question, conn=conn, client=client, top_k=top_k)

    print(f"\n{answer.text}\n")

    if answer.sources:
        print("sources:")
        for number, match in enumerate(answer.sources, start=1):
            print(
                f"  [{number}] {match.source} #{match.chunk_index}  ({match.similarity:.3f})"
            )
            if show_chunks:
                print(f"      {' '.join(match.content.split())[:100]}...")
    else:
        print("sources: none retrieved")

    print(
        f"\n{answer.seconds:.1f}s | {answer.prompt_tokens} prompt tokens,"
        f" {answer.response_tokens} response tokens"
    )


def questions_from_stdin() -> Iterator[str]:
    """Read questions until EOF, so the model stays loaded between them."""
    while True:
        try:
            line = input("\nquestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if line:
            yield line


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="*", help="omit to ask interactively")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--show-chunks", action="store_true", help="print the retrieved text too"
    )
    args = parser.parse_args()

    with (
        supabase.connect() as conn,
        httpx.Client(timeout=DEFAULT_TIMEOUT_SECONDS) as client,
    ):
        if not supabase.count_chunks(conn):
            raise SystemExit(
                "Nothing stored. Run: python scripts/ingest.py data/samples"
            )

        if args.question:
            try:
                ask(
                    " ".join(args.question),
                    conn=conn,
                    client=client,
                    top_k=args.top_k,
                    show_chunks=args.show_chunks,
                )
            except ASK_ERRORS as exc:
                raise SystemExit(f"error: {exc}") from exc
            return

        for question in questions_from_stdin():
            # One bad question must not end the session: the loop exists to
            # keep the model warm between questions, and a timeout or a
            # restarted Ollama should cost you that question, not the rest.
            try:
                ask(
                    question,
                    conn=conn,
                    client=client,
                    top_k=args.top_k,
                    show_chunks=args.show_chunks,
                )
            except ASK_ERRORS as exc:
                print(f"\nerror: {exc}")
            except KeyboardInterrupt:
                print("\ninterrupted")


if __name__ == "__main__":
    main()
