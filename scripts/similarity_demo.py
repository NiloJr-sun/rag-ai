"""T1.2 -- rank text by meaning, not by keyword.

ollama pull nomic-embed-text
python scripts/similarity_demo.py
"""

from __future__ import annotations

import httpx
from app.rag.embeddings import embed_text
from app.rag.retrieval import cosine_similarity, rank_by_similarity

QUERY = "Where was the cat resting?"

DOCUMENTS = [
    "The cat sat on the mat.",
    "A feline rested on the rug.",  # same meaning, almost no shared words
    "The dog slept in its basket.",  # related topic, different subject
    "Kubernetes schedules containers across a cluster.",  # unrelated
    "cat",  # keyword match only
]


def main() -> None:
    with httpx.Client(timeout=60.0) as client:
        query_vector = embed_text(QUERY, client=client)
        vectors = {text: embed_text(text, client=client) for text in DOCUMENTS}

        print(f"query: {QUERY}\n")
        for text, score in rank_by_similarity(query_vector, vectors.items()):
            print(f"  {score:.4f}  {text}")

        a, b = DOCUMENTS[0], DOCUMENTS[1]
        print(
            f"\nsame meaning, different words: {cosine_similarity(vectors[a], vectors[b]):.4f}"
        )
        print(f"  {a!r} vs {b!r}")


if __name__ == "__main__":
    main()
