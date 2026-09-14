"""T1.3 -- compare chunking strategies by how well they actually retrieve.

python scripts/chunking_demo.py
"""

from __future__ import annotations

from pathlib import Path

import httpx
from app.rag.chunking import chunk_by_sentence, chunk_fixed
from app.rag.embeddings import embed_text
from app.rag.retrieval import rank_by_similarity

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "sample.txt"

QUERIES = [
    "How does cosine similarity work?",
    "Why should a document be split into smaller pieces?",
    "How do I cook pasta?",
]

STRATEGIES = [
    ("whole document", lambda t: [t]),
    ("fixed 200 / 0", lambda t: chunk_fixed(t, size=200, overlap=0)),
    ("fixed 200 / 50", lambda t: chunk_fixed(t, size=200, overlap=50)),
    ("fixed 500 / 50", lambda t: chunk_fixed(t, size=500, overlap=50)),
    ("fixed 1000 / 100", lambda t: chunk_fixed(t, size=1000, overlap=100)),
    ("sentences <= 500", lambda t: chunk_by_sentence(t, max_size=500)),
]


def main() -> None:
    text = SAMPLE.read_text()
    print(f"{SAMPLE.name}: {len(text)} chars\n")

    with httpx.Client(timeout=120.0) as client:
        query_vectors = {q: embed_text(q, client=client) for q in QUERIES}

        for name, strategy in STRATEGIES:
            chunks = strategy(text)
            sizes = [len(c) for c in chunks]
            vectors = [(c, embed_text(c, client=client)) for c in chunks]

            print(
                f"{name:<18} {len(chunks):>3} chunks, avg {sum(sizes) // len(sizes):>4} chars"
            )
            for query in QUERIES:
                best, score = rank_by_similarity(query_vectors[query], vectors)[0]
                preview = " ".join(best.split())[:58]
                print(f"    {score:.4f}  {query[:44]:<44}  {preview}...")
            print()


if __name__ == "__main__":
    main()
